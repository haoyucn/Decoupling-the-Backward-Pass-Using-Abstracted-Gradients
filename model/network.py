
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

class GradSaver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sequentialOutput, saver):
        ctx.save_for_backward(x, saver, sequentialOutput)
        return sequentialOutput.clone().detach()

    @staticmethod
    def backward(ctx, gradients):
        x, saver, sequentialOutput, = ctx.saved_tensors
        target_len = 392
        x1 = x[:target_len]
        x2 = x[target_len:]
        # print('x2.shape[0]', x2.shape[0], 'target_len', target_len)
        x2 = F.pad(x2, (0,0,0, abs(x2.shape[0] - target_len)), "constant", 0)
        m1 = torch.linalg.lstsq(x1, sequentialOutput[:target_len]).solution
        m1_grad = torch.matmul(gradients[:target_len], torch.transpose(m1, 0, 1))
        s2 = sequentialOutput[target_len:]
        s2 = F.pad(s2, (0,0,0, abs(s2.shape[0] - target_len)), "constant", 0)
        # m2 = F.pad(m2, (0,0,0, abs(m2.shape[0] - target_len)), "constant", 0)
        
        m2 = torch.linalg.lstsq(x2, s2).solution
        # m2 = m2[:x[target_len:].shape[0]]
        g2 = gradients[target_len:]
        g2 = F.pad(g2, (0,0,0, abs(g2.shape[0] - target_len)), "constant", 0)
        m2_grad = torch.matmul(g2, torch.transpose(m2, 0, 1))
        # m2_grad_zero = m2_grad[x[target_len:].shape[0]:]
        m2_grad = m2_grad[:x[target_len:].shape[0]]
        saver.grad = gradients.clone()
        # print('m2_grad_zero sum', m2_grad_zero.abs().sum())
        
        # print('m1', m1.shape)
        # print('m2', m2.shape)
        

        # print('g2', gradients[target_len:].shape)
        # print('m1_grad', m1_grad.shape)
        # print('m2_grad', m2_grad.shape)

        return torch.cat((m1_grad,m2_grad), 0), None, None


        # m = torch.linalg.lstsq(x, sequentialOutput).solution
        # saver.grad = gradients.clone()
        # outGrad = torch.matmul(gradients, torch.transpose(m, 0, 1))
        # # print('outGrad', outGrad.shape)
        # return outGrad, None, None



class TeacherNet(nn.Module):
    def __init__(self, total_image_pixel = 784):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(total_image_pixel, 392)
        self.distillable = torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU())
        self.fc3 = nn.Linear(49, 10)

        self.distill = False

        self.distilledLayer = nn.Linear(392, 49)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 28*28))
        
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
        self.saver = torch.ones((64,49), dtype = self.layers[1].weight.dtype, requires_grad=True) # check dims of this (batch, output of M) -- TODO
        self.gradDiverge = GradSaver.apply

    def getLayersOutput(self, x):
        self.layersOutput = []

        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        self.input_x = x_clone

        lo = x_clone
        for l in self.layers:
            lo = l(lo)
            lo = F.relu(lo)
        return lo

    def forward(self, x):
        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        self.sequentialOutput = self.getLayersOutput(x_clone)
        # self.saver = sequentialOutput.clone().detach()
        if self.saver.shape != self.sequentialOutput.shape:
            self.saver = torch.ones(self.sequentialOutput.shape)
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
            self.mNet = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet(o_2)
        o_4 = self.fc3(o_3)
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
    def __init__(self, teacherNet = None, total_image_pixel = 784):
        super(FastUpdateNetLarge, self).__init__()
        if teacherNet:
            self.fc1 = teacherNet.fc1
            self.mNet = MNet(teacherNet.distillable)
            self.fc3 = teacherNet.fc3
        else:
            self.fc1 = nn.Linear(total_image_pixel, 392)
            self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc2 = nn.Linear(49,392)
            # self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU(), nn.Linear(49, 49), nn.ReLU()))

            self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet1(o_2)
        o_imd = F.relu(self.fc2(o_3))
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


class IncomingGradDiverger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sequentialOutput, incomingGradSaver, mSaver):
        ctx.save_for_backward(x, incomingGradSaver, sequentialOutput, mSaver)
        ctx.mark_non_differentiable(sequentialOutput)
        ctx.mark_non_differentiable(incomingGradSaver)
        ctx.mark_non_differentiable(mSaver)
        return sequentialOutput.clone().detach()

    @staticmethod
    def backward(ctx, gradients):
        x, incomingGradSaver, sequentialOutput, mSaver = ctx.saved_tensors
        m = torch.linalg.lstsq(x, sequentialOutput).solution
        incomingGradSaver.grad = gradients.clone()
        mSaver.m = m.clone()
        z = torch.matmul(gradients, torch.transpose(m, 0, 1))
        return z, None, None, None

class OutgoingGradDiverger(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, saver):
        ctx.save_for_backward(x, saver)
        return x

    @staticmethod
    def backward(ctx, gradients):
        x, saver = ctx.saved_tensors
        saver.grad = gradients.clone()
        return gradients, None

class IncomingGradDiverger_c(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sequentialOutput, incomingGradSaver, mSaver, cNet):
        ctx.x = x
        ctx.sequentialOutput = sequentialOutput
        ctx.incomingGradSaver = incomingGradSaver
        ctx.mSaver = mSaver
        ctx.cNet = cNet
        return sequentialOutput.clone().detach()

    @staticmethod
    def backward(ctx, gradients):
        x = ctx.x
        sequentialOutput = ctx.sequentialOutput
        incomingGradSaver = ctx.incomingGradSaver
        mSaver = ctx.mSaver
        cNet = ctx.cNet
        incomingGrad_permutation = cNet(gradients)
        incomingGrad_permutated = gradients * incomingGrad_permutation
        m = torch.linalg.lstsq(x, sequentialOutput).solution
        incomingGradSaver.grad = gradients.clone()
        mSaver.m = m.clone()
        return torch.matmul(incomingGrad_permutated, torch.transpose(m, 0, 1)), None, None, None, None


class CNet(torch.nn.Module):
    def __init__(self, sequentialLayers):
        super(CNet, self).__init__()
        self.layers = []
        for l in sequentialLayers:
            if not isinstance(l, nn.ReLU):
                # l.requires_grad = True
                self.layers.append(l)
        # self.layers.requires_grad = False
        self.input_x = None
        self.layersOutput = []
        self.incomingGradSaver = torch.ones((64,49), dtype = self.layers[1].weight.dtype, requires_grad=True) # check dims of this (batch, output of M) -- TODO
        self.incomingGradDiverger = IncomingGradDiverger.apply
        self.incomingGradDiverger_c = IncomingGradDiverger_c.apply
        self.outgoingGradSaver = torch.ones((64,49), dtype = self.layers[1].weight.dtype, requires_grad=True)
        self.outgoingGradDiverger = OutgoingGradDiverger.apply
        
        self.layerNorm = torch.nn.LayerNorm(49)
        self.c = torch.nn.Linear(49, 49)

        self.cOptimizer = torch.optim.Adam([self.c.weight, self.c.bias, self.layerNorm.weight, self.layerNorm.bias], lr=0.01)

        self.mSaver =  torch.ones((self.layers[-1].weight.shape[1], self.layers[0].weight.shape[0]), requires_grad = True, dtype = self.layers[1].weight.dtype)
        self.mse = torch.nn.MSELoss()
        self.useC = False


    def getLayersOutput(self, x):
        self.layersOutput = []
        self.input_x = x

        lo = x
        for l in self.layers:
            lo = l(lo)
            lo = F.relu(lo)
        return lo

    def forward(self, x):
        x_clone = x.clone().detach()
        x_clone.requires_grad = True
        if self.outgoingGradSaver.shape != x.shape:
            self.outgoingGradSaver = x.clone().detach()
        x_clone = self.outgoingGradDiverger(x_clone, self.outgoingGradSaver)
        self.sequentialOutput = self.getLayersOutput(x_clone)
        if self.incomingGradSaver.shape != self.sequentialOutput.shape:
            self.incomingGradSaver = torch.ones(self.sequentialOutput.shape)
        if self.useC:
            return self.incomingGradDiverger_c(x, self.sequentialOutput.clone().detach(), self.incomingGradSaver, self.mSaver, self.c)

        return self.incomingGradDiverger(x, self.sequentialOutput.clone().detach(), self.incomingGradSaver, self.mSaver)

    def backwardHidden(self):
        # print('in MNet, ', self.saver.grad)
        self.sequentialOutput.backward(gradient = self.incomingGradSaver.grad.clone().detach())
    
    def train_c(self):
        incomingGrad = self.incomingGradSaver.grad
        outgoingGrad = self.outgoingGradSaver.grad
        m = self.mSaver.m
        lm = self.layerNorm(incomingGrad)
        incomingGrad_permutation = self.c(lm)
        incomingGrad_permutated = incomingGrad * incomingGrad_permutation
        
        outgoingGrad_permutated = torch.matmul(incomingGrad_permutated, torch.transpose(m, 0, 1))
        loss = self.mse(outgoingGrad_permutated, outgoingGrad)
        loss.backward()
        self.cOptimizer.step()

    def get_parameters(self):
        ps = []
        for l in self.layers:
            ps.append(l.weight)
            ps.append(l.bias)
        return ps

class FastUpdateNet_CNet(torch.nn.Module):
    def __init__(self, teacherNet = None, total_image_pixel = 784):
        super(FastUpdateNet_CNet, self).__init__()
        if teacherNet:
            self.fc1 = teacherNet.fc1
            self.cNet = MNet(teacherNet.distillable)
            self.fc3 = teacherNet.fc3
        else:
            self.fc1 = nn.Linear(total_image_pixel, 392)
            self.cNet = CNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.cNet(o_2)
        o_4 = self.fc3(o_3)
        return F.log_softmax(o_4)

    def get_parameters(self):
        ps = []
        mnetPs = self.cNet.get_parameters()
        ps.append(self.fc1.weight)
        ps.append(self.fc1.bias)
        ps.append(self.fc3.weight)
        ps.append(self.fc3.bias)
        for p in mnetPs:
            ps.append(p)
        return ps


class FastUpdateNetLarge_CNet(torch.nn.Module):
    def __init__(self, teacherNet = None, total_image_pixel = 784):
        super(CNetWithM, self).__init__()
        if teacherNet:
            self.fc1 = teacherNet.fc1
            self.mNet = MNet(teacherNet.distillable)
            self.fc3 = teacherNet.fc3
        else:
            self.fc1 = nn.Linear(total_image_pixel, 392)
            self.mNet1 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc2 = nn.Linear(49,392)
            self.mNet2 = MNet(torch.nn.Sequential(nn.Linear(392, 196), nn.ReLU(), nn.Linear(196, 98), nn.ReLU(), nn.Linear(98, 49), nn.ReLU()))
            self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        o_1 = torch.reshape(x, (x.shape[0], 28*28))
        
        o_2 = F.relu(self.fc1(o_1))
        o_3 = self.mNet1(o_2)
        o_imd = F.relu(self.fc2(o_3))
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