
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

# class GradSaver(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, sequentialOutput, saver):
#         ctx.save_for_backward(x, saver, sequentialOutput)
#         return sequentialOutput.clone().detach()

#     @staticmethod
#     def backward(ctx, gradients):
#         # old method using least squares 
#         # x, saver, sequentialOutput, = ctx.saved_tensors
#         # m = torch.linalg.lstsq(x, sequentialOutput).solution
#         # saver.grad = gradients.clone()
#         # return torch.matmul(gradients, torch.transpose(m, 0, 1)), None, None
        
#         x, saver, sequentialOutput, = ctx.saved_tensors
#         # s = time.time()
#         t = 1e-9
#         x_T = torch.transpose(x, 0, 1)
#         I = torch.eye(x.shape[0]).to('cuda:1')
#         try:
#             pinv = torch.mm(x_T, torch.inverse(torch.mm(x, x_T) + t * I)) 
#         except Exception:
#             pinv = torch.mm(x_T, torch.inverse(torch.mm(x, x_T) + I)) 

#         # print(pinv.shape)
#         m = torch.mm(pinv, sequentialOutput) # torch.transpose(pinv, -1, 1)
#         # print(m.shape)


#         # print(x.shape)
#         # print(sequentialOutput.shape)
#         # print('saver dtype ', saver.dtype, saver.shape, saver.device)
#         # print('gradients dtype ', gradients.dtype, gradients.shape, gradients.device)
#         saver.grad = gradients.clone()#.cpu()
#         # print('in grad saver ', saver.grad)


#         # print(gradients.shape, m.shape)
#         # a = 
    
#         # z = torch.mm(gradients, m)#, None, None
#         # z = torch.bmm(gradients, m)
#         # lst_sqs_sum += (time.time() -s )
#         z = torch.matmul(gradients, torch.transpose(m, 0, 1))#, None, None
#         # print('matmul result in grad saver: ', z)
#         return z, None, None

class GradSaver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sequentialOutput, saver):
        ctx.save_for_backward(x, saver, sequentialOutput)
        return sequentialOutput.clone().detach()

    @staticmethod
    def backward(ctx, gradients):
        x, saver, sequentialOutput, = ctx.saved_tensors
        m = torch.linalg.lstsq(x, sequentialOutput).solution
        saver.grad = gradients.clone()
        return torch.matmul(gradients, torch.transpose(m, 0, 1)), None, None


class TeacherNet(nn.Module):
    def __init__(self, total_image_pixel = 784):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(total_image_pixel, 392)
        self.distillable = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1')
        self.fc3 = nn.Linear(49, 10)

        self.distill = False

        self.distilledLayer = nn.Linear(392, 49)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 28*28)).to('cuda:1')
        
        x = F.relu(self.fc1(x))
        
        # x = F.dropout(x, training=self.training)

        if self.distill:
            x = F.relu(self.distilledLayer(x))
        else:
            x = self.distillable(x)

        x = self.fc3(x)
        return F.log_softmax(x)


class MNet(torch.nn.Module):
    def __init__(self, sequentialLayers):
        super(MNet, self).__init__()
        self.layers = []
        for l in sequentialLayers:
            if not isinstance(l, nn.ReLU):
                # l.requires_grad = True
                self.layers.append(l)
        # self.layers.requires_grad = False
        self.input_x = None
        self.layersOutput = []
        self.saver = torch.ones((64,49), dtype = self.layers[1].weight.dtype, requires_grad=True).to('cuda:1') # check dims of this (batch, output of M) -- TODO
        self.gradDiverge = GradSaver.apply

    def getLayersOutput(self, x):
        self.layersOutput = []

        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        self.input_x = x_clone

        lo = x_clone
        # print('x_clone.device', x_clone.device)
        # print('layer', self.layers[0])
        for l in self.layers:
            lo = l(lo)
            # print('lo.device', lo.device)
            lo = F.relu(lo)
        return lo

    # def weightUpdate(self, correctness, lr = 0.001):
    #     for i in range(len(self.layersOutput)):
    #         output = self.layersOutput[i]
    #         layer = self.layers[i]
    #         maxIndexs = torch.argmax(output, dim = -1)
    #         weightChange = layer.weight.abs() * (1 - layer.weight.abs()) * (-1)
    #         for j in range(len(maxIndexs)):
    #             if not correctness[j]:
    #                 index = maxIndexs[j]
    #                 deltaWeight = weightChange.clone()
    #                 deltaWeight[int(index)] = deltaWeight[int(index)] * (-1)
    #                 layer.weight = torch.nn.Parameter(layer.weight + deltaWeight * lr)

    def forward(self, x):
        # print('in forward')
        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        self.sequentialOutput = self.getLayersOutput(x_clone).to('cuda:1')
        # self.saver = sequentialOutput.clone().detach()
        if self.saver.shape != self.sequentialOutput.shape:
            self.saver = torch.ones(self.sequentialOutput.shape).to('cuda:1')
        # print(self.sequentialOutput.requires_grad)
        # m = torch.linalg.lstsq(x_clone, self.sequentialOutput).solution.detach()
        # o = F.linear(x, torch.transpose(m, 0, 1))
        return self.gradDiverge(x, self.sequentialOutput.clone().detach(), self.saver)

    def backwardHidden(self):
        # print('in MNet, ', self.saver.grad)
        self.sequentialOutput.backward(gradient = self.saver.grad.clone().detach())
        
    def get_parameters(self):
        ps = []
        for l in self.layers:
            ps.append(l.weight)
            ps.append(l.bias)
        return ps


class FastUpdateNet(torch.nn.Module):
    def __init__(self, teacherNet = None, total_image_pixel = 784):
        super(FastUpdateNet, self).__init__()
        if teacherNet:
            self.fc1 = teacherNet.fc1
            self.mNet = MNet(teacherNet.distillable)
            self.fc3 = teacherNet.fc3
        else:
            self.fc1 = nn.Linear(total_image_pixel, 392)
            self.mNet = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1'))
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        # print(x.device)
        o_1 = torch.reshape(x, (x.shape[0], 28*28)).to('cuda:1')
        # print(o_1.device)
        o_2 = F.relu(self.fc1(o_1))
        # print(o_2.device)
        o_3 = self.mNet(o_2)
        # print(o_3.device)
        o_4 = self.fc3(o_3)
        # print(o_4.device)
        return F.log_softmax(o_4)

    def get_parameters(self):
        ps = []
        mnetPs = self.mNet.get_parameters()
        ps.append(self.fc1.weight)
        ps.append(self.fc1.bias)
        ps.append(self.fc3.weight)
        ps.append(self.fc3.bias)
        for p in mnetPs:
            ps.append(p)
        return ps
    
class FastUpdateNetLarge(torch.nn.Module):
    def __init__(self, use_M= True, teacherNet = False, total_image_pixel = 784):
        super(FastUpdateNetLarge, self).__init__()
        if teacherNet:
            self.fc1 = teacherNet.fc1
            self.mNet = MNet(teacherNet.distillable)
            self.fc3 = teacherNet.fc3
        else:
            self.fc1 = nn.Linear(total_image_pixel, 392)
            if use_M:
                self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1'))
                self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1'))
            else:
                self.mNet1 = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1')
                self.mNet2 = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1')
            
            self.fc2 = nn.Linear(49,392)
            # self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU()))

            
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet1(o_2)
        o_imd = F.relu(self.fc2(o_3))
        # print('mnet2')
        o_4 = self.mNet2(o_imd)

        o_5 = self.fc3(o_4)
        return F.log_softmax(o_5)

    def get_parameters(self):
        ps = []
        mnet1Ps = self.mNet1.get_parameters()
        mnet2Ps = self.mNet2.get_parameters()
        ps.append(self.fc1.weight)
        ps.append(self.fc1.bias)
        ps.append(self.fc2.weight)
        ps.append(self.fc2.bias)
        ps.append(self.fc3.weight)
        ps.append(self.fc3.bias)
        for p in mnet1Ps:
            ps.append(p)
        for p in mnet2Ps:
            ps.append(p)
        return ps
    
class FastUpdateNetLarge_Better(torch.nn.Module):
    def __init__(self, use_M= True, total_image_pixel = 784):
        super(FastUpdateNetLarge_Better, self).__init__()
   
        self.fc1 = nn.Linear(total_image_pixel, 700)
        if use_M:
            self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(700, 620), nn.ReLU(), nn.Linear(620, 540), nn.ReLU(), nn.Linear(540, 460), nn.ReLU()).to('cuda:1'))
            self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(380, 300), nn.ReLU(), nn.Linear(300, 220), nn.ReLU(), nn.Linear(220, 140), nn.ReLU()).to('cuda:1'))
        else:
            self.mNet1 = torch.nn.Sequential(nn.Linear(700, 620), nn.ReLU(), nn.Linear(620, 540), nn.ReLU(), nn.Linear(540, 460), nn.ReLU()).to('cuda:1')
            self.mNet2 = torch.nn.Sequential(nn.Linear(380, 300), nn.ReLU(), nn.Linear(300, 220), nn.ReLU(), nn.Linear(220, 140), nn.ReLU()).to('cuda:1')
        
        self.fc2 = nn.Linear(460,380)
            # self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU()))

            
        self.fc3 = nn.Linear(140, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet1(o_2)
        o_imd = F.relu(self.fc2(o_3))
        # print('mnet2')
        o_4 = self.mNet2(o_imd)

        o_5 = self.fc3(o_4)
        return F.log_softmax(o_5)
    
class FastUpdateNetLarge_Best(torch.nn.Module):
    def __init__(self, use_M= True, teacherNet = False, total_image_pixel = 784):
        super(FastUpdateNetLarge_Best, self).__init__()

        self.fc1 = nn.Linear(total_image_pixel, 392)
        if use_M:
            # self.mNet1 = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 392), nn.ReLU(), nn.Linear(392, 784), nn.ReLU()).to('cuda:1')
            self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 392), nn.ReLU(), nn.Linear(392, 784), nn.ReLU()).to('cuda:1'))
            self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1'))
        else:
            self.mNet1 = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 392), nn.ReLU(), nn.Linear(392, 784), nn.ReLU()).to('cuda:1')
            self.mNet2 = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1')
        
        self.fc2 = nn.Linear(784,392)
        # self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU()))

        
        self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet1(o_2)
        o_imd = F.relu(self.fc2(o_3))
        # print('mnet2')
        o_4 = self.mNet2(o_imd + o_2) # NOTE the residual connection

        o_5 = self.fc3(o_4)
        return F.log_softmax(o_5)



class Linear_AE(torch.nn.Module):
    def __init__(self, teacherNet = None, total_image_pixel = 784, encoding_dim=10):
        super(Linear_AE, self).__init__()
        
        self.fc1 = nn.Linear(total_image_pixel, 392)
        self.conv1 = nn.Conv2d(1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten(start_dim=1)
        mnet_in = 1568# 392

        self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(mnet_in, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()).to('cuda:1'))
        self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(49, 98), nn.ReLU(), nn.Linear(98, 196), nn.ReLU(), nn.Linear(196, 392), nn.ReLU()).to('cuda:1'))
        self.fc3 = nn.Linear(49, 10)

        self.fc5 = nn.Linear(10,49)
        self.fc8 = nn.Linear(392, total_image_pixel)
        # try conv here too?
        self.convf = nn.ConvTranspose2d(1, 1, 3, stride=2, 
            padding=1, output_padding=1)

    def forward(self, x):

        # x = torch.reshape(x, (x.shape[0], 28*28)).to('cuda:1')

        # encoder
        o_1 = self.flatten(F.relu(self.conv1(x)))
        # print(o_1.shape)
        o_4 = self.mNet1(o_1)
        o_5 = self.fc3(o_4)

        self.encoded_x = o_5 # in case we want to do any sort of clustering analysis

        # decoder (reverse encoder layers)
        o_6 = F.relu(self.fc5(o_5))
        o_9 = self.mNet2(o_6)
        o_10 = self.fc8(o_9)

        y = torch.reshape(o_10, (o_10.shape[0], 28, 28))
        y = y.squeeze(1)
        return o_10, y

    def get_parameters(self):
        ps = []
        mnet1Ps = self.mNet1.get_parameters()
        mnet2Ps = self.mNet2.get_parameters()
        ps.append(self.conv1.weight)
        ps.append(self.conv1.bias)
        ps.append(self.fc1.weight)
        ps.append(self.fc1.bias)
        ps.append(self.fc3.weight)
        ps.append(self.fc3.bias)
        ps.append(self.fc5.weight)
        ps.append(self.fc5.bias)
        ps.append(self.fc8.weight)
        ps.append(self.fc3.bias)
        for p in mnet1Ps:
            ps.append(p)
        for p in mnet2Ps:
            ps.append(p)
        return ps