
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

