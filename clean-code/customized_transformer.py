import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


IS_CUDA = True
H = 8
BATCH_SIZE = 32
MODEL_DIM = 512


class GradSaver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sequentialOutput, saver):
        # print('in forward in grad saver')
        ctx.save_for_backward(x, saver, sequentialOutput)
        
        # print('saving stuff in ctx from gradsaver')
        return sequentialOutput.clone().detach()
    
    @staticmethod
    def backward(ctx, gradients):
        global lst_sqs_sum
        
        x, saver, s = ctx.saved_tensors
        g = gradients
        saver.g = gradients.clone()
        
        return gradients, None, None 

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class EncoderDecoder_split1(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder_split1, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        startTime = time
        x = self.encode(src, src_mask)
        return x

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        return self.encoder(self.src_embed(src), src_mask)
    
class EncoderDecoder_split2(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder_split2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        a = self.encoder((src), src_mask)
        # a.backward(gradient = torch.ones((32, 72, 512)).to('cuda:0'))
        return a

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class MNet_Transformer(torch.nn.Module):
    def __init__(self, sequentialLayers, output_size):
        print('Creating MNet_Transformer')
        super(MNet_Transformer, self).__init__()
        self.layers = []
        for l in sequentialLayers:
            self.layers.append(l)
        # self.layers.requires_grad = False
        self.input_x = None
        self.layersOutput = []
        self.saver = torch.ones(output_size, dtype = self.layers[0].norm.a_2.data.dtype, requires_grad=True).to('cuda:0').detach() # check dims of this (batch, output of M) 
        self.gradDiverge = GradSaver.apply
        self.sequentialOutput = None

        # Switch to enable M ---------- NOTE for hybrid method
        self.use_M = True
        self.h = H
        self.d_k = MODEL_DIM // self.h
        self.linears = clones(nn.Linear(MODEL_DIM, MODEL_DIM), 4)
        self.dropout = nn.Dropout(p=0.1)
        self.self_attn_func = MultiHeadedAttention(H, MODEL_DIM)

    def getLayersOutput(self, x, mask,):
        self.layersOutput = []

        if self.use_M:
            x_clone = x.clone().detach()
            x_clone.requires_grad = False
            self.input_x = x_clone
        else:
            self.input_x = x

        sublayer_ct = 0
        lo = self.input_x
        for l in self.layers:
            # print(l)
            # print(type(l))

            if 'sublayer' in str(type(l)).lower() and sublayer_ct == 0:
                # lo = l(lo, self.custom_multihead_attn(lo,lo,lo, mask))
                lo = l(lo, lambda x: self.self_attn_func(x, x, x, mask))
                self.attn = l.attn_mat
                sublayer_ct += 1
            elif 'sublayer' in str(type(l)).lower() and sublayer_ct > 0:
                lo = l(lo, feed_forward_layer)
                sublayer_ct = 0
            elif 'feed' in str(type(l)).lower():
                feed_forward_layer = l # save for second sublayer
            else:
                print('ERROR: unexpected layer: ', l)

        return lo

    def forward(self, x, mask,):
        if self.use_M:
            x_clone = x.clone().detach()
            x_clone.requires_grad = False
            self.sequentialOutput = self.getLayersOutput(x_clone, mask,)
        
            t = self.gradDiverge(x, self.sequentialOutput.clone().detach(), self.saver)
        else:
            # self.sequentialOutput = self.getLayersOutput(x, mask,)
            # t = self.sequentialOutput
            pass
        # print('t, ', t.grad)

        # t.backward(gradient = torch.ones(t.shape).to('cuda:0'))
        return t

    def backwardHidden(self):
        self.sequentialOutput.backward(gradient = self.saver.g.clone().detach())
        
    def get_parameters(self):
        ps = []
        ps.append(self.layers[0].out_proj.weight) # self attn
        ps.append(self.layers[0].out_proj.bias)

        for l in self.layers:
            if hasattr(l, 'weight') and hasattr(l, 'bias'):
                ps.append(l.weight)
                ps.append(l.bias)
        return ps

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class CustomSequential(nn.Module):
    def __init__(self, encoder_layers) -> None:
        super(CustomSequential, self).__init__()
        self.encoder_layers = encoder_layers
        modules = []
        for i in range(len(encoder_layers)):
            # modules.append(encoder_layers[i].self_attn)
            modules.append(encoder_layers[i].sublayer[0])
            modules.append(encoder_layers[i].feed_forward)
            modules.append(encoder_layers[i].sublayer[1])
            # try to add the layernorms too
        # print(modules)
        # print(len(modules))
        self.custom_sequential = nn.Sequential(*modules)


class Encoder_MNet_split1(nn.Module):
    "Core encoder is a stack of N layers"
    "Just the top N=6/2=3 encoder layers, no M net"

    def __init__(self, layer, N):
        super(Encoder_MNet_split1, self).__init__()
        self.layers = clones(layer, N)
        # self.layers = [layer for _ in range(N)]
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for idx, layer in enumerate(self.layers):
                x = layer(x, mask)
        return self.norm(x)




class Encoder_MNet_split2(nn.Module):
    "Core encoder is a stack of N layers"
    "The bottom half of the encoder N=6/2=3 layers, uses M net exclusively"

    def __init__(self, layer, N):
        super(Encoder_MNet_split2, self).__init__()
        self.layers = clones(layer, N)
        # self.layers = [layer for _ in range(N)]
        self.norm = LayerNorm(layer.size)
            
        # mnet is the only part of this split encoder
        self.mnet1 = MNet_Transformer(CustomSequential(self.layers).custom_sequential, (BATCH_SIZE, MODEL_DIM))


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.mnet1(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # try:
        #     return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # except RuntimeError:
        #     return self.a_2.cuda('cuda:0') * (x - mean) / (std + self.eps) + self.b_2.cuda('cuda:0')
        

class LayerNorm_CUDA(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm_CUDA, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features)).to('cuda:0')
        self.b_2 = nn.Parameter(torch.zeros(features)).to('cuda:0')
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# softmax_func_M = nn.Softmax(dim=-1)
# # layerwisenorm_M = nn.LayerNorm(m_shape, eps=1e-6, elementwise_affine=True).to('cuda:0')
# layerwisenorm_M = LayerNorm_CUDA(m_shape)
# # layerwisenorm_M = LayerNorm(m_shape)
# # dropout_M = nn.Dropout(p=0.1)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.attn_mat = None

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."

        imd = sublayer(self.norm(x))
        if type(imd) is tuple:
            self.attn_mat = imd[-1] # grab second output from MHAttn
        else:
            imd = [imd, -1]
        return x + self.dropout(imd[0]) # regular opeartion

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class CustomSequentialDecoder(nn.Module):
    def __init__(self, decoder_layers) -> None:
        super(CustomSequential, self).__init__()
        self.decoder_layers = decoder_layers
        modules = []
        for i in range(len(decoder_layers)):
            # modules.append(encoder_layers[i].self_attn)
            modules.append(decoder_layers[i].sublayer[0])
            modules.append(decoder_layers[i].sublayer[1])
            modules.append(decoder_layers[i].feed_forward)
            modules.append(decoder_layers[i].sublayer[2])
            # try to add the layernorms too
        # print(modules)
        # print(len(modules))
        self.custom_sequential = nn.Sequential(*modules)


class Decoder_MNet(nn.Module):
    "Generic N layer decoder with masking."
    # FIXME: M abstraction does not depend on memory directly -- could be a large problem

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        self.layer_idx_M = [3] # just choose one encoder layer to abstract
        self.mnet = MNet_TransformerDecoder(CustomSequentialDecoder([layer]).custom_sequential, (BATCH_SIZE, MODEL_DIM))

    def forward(self, x, memory, src_mask, tgt_mask):
        for idx, layer in enumerate(self.layers):
            if idx in self.layer_idx_M:
                # print('passing input thru MNet')
                x = self.mnet(x, memory, src_mask, tgt_mask)
            else:
                x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # this is the output of softmax (before multiplied by V) ------------------ NOTE
        # self.attn
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x), self.attn
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
        # try:
        #     return self.w_2(self.dropout(self.w_1(x).relu()))
        # except RuntimeError:
        #     return (self.w_2.cuda('cuda:0'))(self.dropout(self.w_1.cuda('cuda:0')(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        startTime = time.time()
        a = self.lut(x)
        print('embedding lut time', time.time()-startTime)
        # print('self.lut.weight',self.lut.weight)
        b = math.sqrt(self.d_model)
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model1(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder_split1(
        Encoder_MNet_split1(EncoderLayer(d_model, c(attn), c(ff), dropout), N//2),
        None,
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)).to(torch.device("cuda:0")),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)).to(torch.device("cuda:0")),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def make_model2(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder_split2(
        Encoder_MNet_split2(EncoderLayer(d_model, c(attn), c(ff), dropout), N//2),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        None,
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

def P1(batch, model1, write_queue, read_queue):
    
    startTime = time.time()
    batch.src = batch.src * 1
    out1 = model1.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
    print('p1 forward time', time.time()-startTime)
    startTime = time.time()
    write_queue.put(out1.clone().detach())
    print('put on queue time', time.time()-startTime)
    out1.backward(gradient = read_queue.get())
    

def P2(batch, model2, loss_compute, write_queue, read_queue):
    
    while True:
        if read_queue.empty():
            continue
        
        out1_clone = read_queue.get()
        batch.tgt = batch.tgt * 1
        startTime = time.time()
        out1_clone.requires_grad = True
        out = model2.forward(out1_clone, batch.tgt, batch.src_mask, batch.tgt_mask)

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss_node.backward() 

        write_queue.put(model2.encoder.mnet1.saver.g)

        # finish executing this process
        model2.encoder.mnet1.backwardHidden() # this will update weights in Mnet (which is contained in model2)
        print('p2 take', time.time()-startTime)
        break

def run_epoch_parallel(
    data_iter,
    model1,
    model2,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    use_M=True
):
    global training_loss
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    mmp = mp.get_context('spawn')
    p1_write_queue = mmp.SimpleQueue()
    p2_write_queue = mmp.SimpleQueue()
    
    for i, batch in enumerate(data_iter):
        # startTime = time.time()
        # out1 = model1.forward(
        #         batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        #     )
        # out1_clone = out1.clone().detach()
        
        # out1_clone.requires_grad = True
        # out = model2.forward(out1_clone, batch.tgt, batch.src_mask, batch.tgt_mask)

        # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node.backward() 
        # out1.backward(gradient = model2.encoder.mnet1.saver.g)
        # model2.encoder.mnet1.backwardHidden() # this will update weights in Mnet (which is contained in model2)
        # print('main thread both p finish time', time.time()-startTime)
        






        model1.share_memory()
        p1 = mmp.Process(target=P1, args=(batch, model1, p1_write_queue,p2_write_queue))
        p2 = mmp.Process(target=P2, args=(batch, model2, loss_compute, p2_write_queue,p1_write_queue))
        queueStartTime = time.time()
        p1.start()
        p2.start()
        print('queueStartTake', time.time()-queueStartTime)
        p1.join()
        p2.join()
        print('queueStartTake', time.time()-queueStartTime)

        if mode == "train" or mode == "train+log":
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()
        
        total_loss += 0
        LOSS = []
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, -1, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        # del loss
        # del loss_node
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])



def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader



def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    time_start = time.time()
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(0)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model1(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(0)
    # print('model.encoder.mnet.use_M ', model.encoder.mnet.use_M)
    module1 = model

    model = make_model2(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(0)

    module2 = model
    


    is_main_process = True
    # if is_distributed:
    #     dist.init_process_group(
    #         "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
    #     )
    #     model = DDP(model, device_ids=[gpu])
    #     module = model.module
    #     is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        list(module1.parameters()) + list(module2.parameters()), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9 # NOTE: potentially need custom get param funcs 
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()

            
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        s = time.time()
        _, train_state = run_epoch_parallel(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            module1,
            module2,
            SimpleLossCompute(module2.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
            use_M = False,
        )
        print(f"Epoch training time: {time.time() - s}")

        GPUtil.showUtilization()
        # if is_main_process:
        #     file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
        #     torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            module1,
            module2,
            SimpleLossCompute(module2.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        timestamp = time.time()
        otherinfo = {
            'experiment name' : 'parallelized transformer N=6',
            'total wallclock training time' : time.time()-time_start,
            'config_info' : config
        }
        file_path = f"%sfinal_{timestamp}_module1.pt" % config["file_prefix"]
        torch.save(training_loss, f'saved_{timestamp}_train_losses_for.pt')
        torch.save(val_loss, f'saved_{timestamp}_val_losses_for.pt')
        torch.save(otherinfo, f'saved_{timestamp}_otherinfo_for.pt')
        torch.save(module1.state_dict(), file_path)
        file_path = f"%sfinal_{timestamp}_module2.pt" % config["file_prefix"]
        torch.save(module1.state_dict(), file_path)

def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        # USE GPU1
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )

def load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en):
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 1,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path) or True: # default to always train
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    return None
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model

s = time.time()
if is_interactive_notebook():
    model = load_trained_model()
print(f"\nTotal wallclock training time: {time.time()-s} seconds")