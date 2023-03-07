
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

class PrimitiveLayerWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layer.requires_grad = False
    
    def forward(self, x):
        x = x.detach()
        x.requires_grad = False
        output = self.layer(x)
        output.detach()
        return output.detach()



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
                l.requires_grad = False
                self.layers.append(l)
        # self.layers.requires_grad = False
        self.input_x = None
        self.layersOutput = []

    def getLayersOutput(self, x):
        self.layersOutput = []

        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        self.input_x = x_clone

        lo = x_clone
        for l in self.layers:
            lo = l(lo)
            self.layersOutput.append(lo)
            lo = F.relu(lo)

        lo = lo.detach()
        lo.requires_grad = False
        return lo

    def weightUpdate(self, correctness, lr = 0.001):
        for i in range(len(self.layersOutput)):
            output = self.layersOutput[i]
            layer = self.layers[i]
            maxIndexs = torch.argmax(output, dim = -1)
            weightChange = layer.weight.abs() * (1 - layer.weight.abs()) * (-1)
            for j in range(len(maxIndexs)):
                if not correctness[j]:
                    index = maxIndexs[j]
                    deltaWeight = weightChange.clone()
                    deltaWeight[int(index)] = deltaWeight[int(index)] * (-1)
                    layer.weight = torch.nn.Parameter(layer.weight + deltaWeight * lr)

    def forward(self, x):
        x_clone = x.clone().detach()
        x_clone.requires_grad = False
        sequentialOutput = self.getLayersOutput(x_clone)
        x_clone_inv = torch.linalg.pinv(x_clone)
        m = F.linear(x_clone_inv, torch.transpose(sequentialOutput, 0, 1)).detach()
        return F.linear(x, torch.transpose(m, 0, 1))
        

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
