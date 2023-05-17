import torch

def P1(t,QUEUE):
    src = torch.randint(2, (32, 72), dtype=torch.int64).cuda(0)
    t = t * 2
    print('231233', src.get_device())
    # t = t * 2
    # data_iter = data_gen(11, 11, 20)
    # print('multed1213')
    # print(next(data_iter))


    # batch.src torch.Size([32, 72]) torch.int64
    # batch.tgt torch.Size([32, 71]) torch.int64
    # batch.src_mask torch.Size([32, 1, 72]) torch.bool
    # batch.tgt_mask torch.Size([32, 71, 71]) torch.bool
    # try:
    #     src = torch.randint(2, (32, 72), dtype=torch.int64, device=torch.device('cuda:0'))
    #     print('src', src.get_device())
    #     tgt = torch.randint(2, (32, 72), dtype=torch.int64, device=torch.device('cuda:0'))
    #     print('tgt', tgt.get_device())
    #     src_mask = (torch.rand((32, 1, 72), device=torch.device('cuda:0')) < 0.9)
    #     print('src_mask', src_mask.get_device())
    #     tgt_mask = (torch.rand((32, 71, 71), device=torch.device('cuda:0')) < 0.9)
    #     print('tgt_mask', tgt_mask.get_device())
    # except Exception as e:
    #     print(1231231231231221)
    #     print(e)


    print('P1 start')

    # out1 = model1.forward(
    #         src, tgt, src_mask, tgt_mask
    #     )
    # out1 = model1.forward(
    #         batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
    #     )
    # print('P1 1')
    # # return
    # # save imd output to QUEUE
    # QUEUE.put(out1.clone().detach())
    # # flag to know when to get something

    # # pull off the gradients
    # out1.backward(gradients = QUEUE.get()) # this will pull gradients off QUEUE and then compute net1's backward
    



def P2(batch, model2, loss_compute, QUEUE):
    print('P2 start')

    out1_clone = QUEUE.get()
    print('P2 1')
    # return
    # tgt = torch.randint((32, 72), dtype=torch.int64, device=torch.device('cuda:0'))
    # src_mask = torch.rand((32, 1, 72), device=torch.device('cuda:0')) < 0.9
    # tgt_mask = torch.rand((32, 71, 71), device=torch.device('cuda:0')) < 0.9
    # out = model2.forward(out1_clone, tgt, src_mask, tgt_mask)

    out = model2.forward(out1_clone, batch.tgt, batch.src_mask, batch.tgt_mask)
    print('P2 2')
    # loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
    # training_loss.append(loss)
    
    # loss_node.backward() # this will put gradients on QUEUE (see Mnet and gradsaver classes)

    

    # # finish executing this process
    # model2.encoder.mnet1.backwardHidden() # this will update weights in Mnet (which is contained in model2)
    