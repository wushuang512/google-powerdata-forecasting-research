"""Task-specific dense and temporal mixer forecasters, not official replicas."""
import torch
from torch import nn
class ResidualDense(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width))
    def forward(self,x):return x+self.net(x)
class MixerBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.time=nn.Sequential(nn.Linear(168,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,168))
        self.channel=ResidualDense(width)
    def forward(self,x):
        x=x+self.time(self.norm(x).transpose(1,2)).transpose(1,2)
        return self.channel(x)
class DirectNetwork(nn.Module):
    def __init__(self,family,width,shared):
        super().__init__();self.family=family;self.shared=shared
        self.site=nn.Embedding(57,4);self.cell=nn.Embedding(10,4)
        if family=='dense':
            self.encode=nn.Sequential(nn.Linear(168*6+12,width),nn.GELU(),nn.Dropout(.1),ResidualDense(width),ResidualDense(width))
            self.out=nn.Linear(width,96)
        else:
            self.embed=nn.Linear(6,width)
            self.blocks=nn.Sequential(MixerBlock(width),MixerBlock(width))
            self.encode=nn.Sequential(nn.Linear(24*width+width+12,width*2),nn.GELU(),nn.Dropout(.1),nn.Linear(width*2,width),nn.GELU())
            self.out=nn.Linear(width,96)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,calendar,site,cell,last,std):
        if not self.shared:
            x=x.clone();x[:,:,4:]=0
        aux=torch.cat([calendar,self.site(site),self.cell(cell)],-1)
        if self.family=='dense':z=self.encode(torch.cat([x.flatten(1),aux],-1))
        else:
            hidden=self.blocks(self.embed(x))
            z=self.encode(torch.cat([hidden[:,-24:].flatten(1),hidden.mean(1),aux],-1))
        # Full learned daily curve around the causal last-day profile. No small
        # residual cap or teacher/last-day anchoring penalty is imposed.
        return last+std*self.out(z)
