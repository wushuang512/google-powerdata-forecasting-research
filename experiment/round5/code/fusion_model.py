import torch
from torch import nn
import torch.nn.functional as F

class FusionMoE(nn.Module):
    """Compact task-specific implementation; W is production-power proxy, not measured compute."""
    def __init__(self,width=32,cross=True,freq=True,sparse=True,scales=(16,32,96)):
        super().__init__();self.width=width;self.cross=cross;self.freq=freq;self.sparse=sparse;self.scales=scales
        self.pembed=nn.ModuleList([nn.Linear(s*4,width) for s in scales]);self.wembed=nn.ModuleList([nn.Linear(s*4,width) for s in scales])
        self.penc=nn.ModuleList([nn.TransformerEncoderLayer(width,4,width*2,.15,batch_first=True,norm_first=True) for _ in scales])
        self.wenc=nn.ModuleList([nn.TransformerEncoderLayer(width,4,width*2,.15,batch_first=True,norm_first=True) for _ in scales])
        self.att=nn.ModuleList([nn.MultiheadAttention(width,4,dropout=.1,batch_first=True) for _ in scales])
        self.fft_p=nn.Linear(2,width);self.fft_w=nn.Linear(2,width);self.cell=nn.Embedding(10,4)
        self.gate=nn.Linear(width*3+4+3,3);self.experts=nn.ModuleList([nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.15),nn.Linear(width*2,width)) for _ in scales])
        self.head=nn.Sequential(nn.Linear(width+4+4+2,width),nn.GELU(),nn.Dropout(.15),nn.Linear(width,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def _parts(self,x,m):
        count=m.sum(1,keepdim=True).clamp_min(1);mu=(x*m).sum(1,keepdim=True)/count
        sd=(((x-mu).square()*m).sum(1,keepdim=True)/count).sqrt().clamp_min(.01)
        z=(x-mu)/sd;z=z*m
        def ma(k):return F.avg_pool1d(F.pad(z[:,None],(k//2,k//2),mode='replicate'),k,stride=1)[:,0]
        slow=ma(25);medium=ma(5);features=torch.stack([z,slow,medium-slow,m],-1)
        return features,mu,sd,z
    def forward(self,x,calendar,cell):
        p,pmu,psd,pz=self._parts(x[:,:,0],x[:,:,2]);w,wmu,wsd,wz=self._parts(x[:,:,1],x[:,:,3]);availability=x[:,:,3].mean(1);embedded=[]
        for j,s in enumerate(self.scales):
            b,l,c=p.shape;pt=p[:,-(l//s)*s:].reshape(b,-1,s*4);wt=w[:,-(l//s)*s:].reshape(b,-1,s*4)
            pe=self.pembed[j](pt);we=self.wembed[j](wt)
            pos=torch.arange(pe.shape[1],device=x.device).float();dim=torch.arange(self.width,device=x.device).float();encoding=torch.sin(pos[:,None]/(10000**(dim[None]/self.width)))
            pe=self.penc[j](pe+encoding);we=self.wenc[j](we+encoding)
            if self.cross:pe=pe+self.att[j](pe,we,we,need_weights=False)[0]*availability[:,None,None]
            pooled=pe.mean(1)
            if self.freq:
                fp=torch.fft.rfft(pz.float(),norm='forward')[:,1:17];fw=torch.fft.rfft(wz.float(),norm='forward')[:,1:17]
                pooled=pooled+self.fft_p(torch.stack([fp.real,fp.imag],-1)).mean(1)
                if self.cross:pooled=pooled+self.fft_w(torch.stack([fw.real,fw.imag],-1)).mean(1)*availability[:,None]
            embedded.append(pooled)
        ce=self.cell(cell.long());ginput=torch.cat([*embedded,ce,availability[:,None],pmu,psd],-1);prob=self.gate(ginput).softmax(-1)
        if self.sparse:
            top=prob.topk(2,-1).indices;active=torch.zeros_like(prob).scatter_(1,top,1);weights=prob*active;weights=weights/weights.sum(-1,keepdim=True)
        else:active=torch.ones_like(prob);weights=prob
        latent=torch.zeros_like(embedded[0]);loads=[]
        for j,expert in enumerate(self.experts):
            ix=active[:,j].bool();part=torch.zeros_like(latent)
            if ix.any():part[ix]=expert(embedded[j][ix])*weights[ix,j,None]
            latent=latent+part;loads.append(ix.float().mean())
        load_balance=3*(prob.mean(0)*torch.stack(loads)).sum()/active.sum(1).mean()
        lastp=(x[:,-96:,0]-pmu)/psd;lastw=(x[:,-96:,1]-wmu)/wsd
        h=self.head(torch.cat([latent[:,None].expand(-1,96,-1),ce[:,None].expand(-1,96,-1),calendar,torch.stack([lastp,lastw],-1)],-1))
        centerp=x[:,-96:,0]+psd*1.5*torch.tanh(h[:,:,0]);centerw=x[:,-96:,1]+wsd*1.5*torch.tanh(h[:,:,3])
        pq=torch.stack([centerp-psd*F.softplus(h[:,:,1]-2),centerp,centerp+psd*F.softplus(h[:,:,2]-2)],-1).clamp(0,1)
        wq=torch.stack([centerw-wsd*F.softplus(h[:,:,4]-2),centerw,centerw+wsd*F.softplus(h[:,:,5]-2)],-1).clamp_min(0)
        wq=torch.minimum(wq,pq)
        return pq,wq,load_balance,prob
