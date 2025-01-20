import torchvision
from torch import nn
import torch.functional as F

vgg16 = torchvision.models.vgg16(pretrained=False)
class LAnet(nn.Module):
    def __init__(self,numclasses=1,ipt_size = 56):
        super(LAnet,self).__init__()
        self.input_w = ipt_size
        self.input_h = ipt_size
        self.vgg_p3 = vgg16.features[:17]
        self.extra = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear((self.input_w*self.input_h*256)//64,512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.2),
            nn.Linear(512,128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.1),
            nn.Linear(128,numclasses),
            nn.Tanh(),
        )


    def forward(self,x):
        out = self.vgg_p3(x)
        out = self.extra(out)
        return out
