import torch

import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Module ProxyAnchor/NCA loss.
        """
        super(Criterion, self).__init__()

        self.opt = opt
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim

        self.pars = {
            'pos_alpha': opt.loss_oproxy_pos_alpha,
            'pos_delta': opt.loss_oproxy_pos_delta,
            'neg_alpha': opt.loss_oproxy_neg_alpha,
            'neg_delta': opt.loss_oproxy_neg_delta
        }

        self.mode = opt.loss_oproxy_mode # ProxyAnchor
        self.name = 'proxynca' if self.mode == 'nca' else 'proxyanchor' 
        
        
        self.proxy_dim = opt.proxy_dim
        self.class_idxs = torch.arange(self.num_proxies)
        self.proxies = torch.randn(self.num_proxies, self.embed_dim) / 8
        self.proxies = torch.nn.Parameter(self.proxies)
        
        self.new_epoch = True
        self.save_mix = False
        self.saved_batch_mix = None
        
        self.optim_dict_list = [{
            'params': self.proxies,
            'lr': opt.lr * opt.loss_oproxy_lrmulti
        }]


    def forward(self, batch, labels, **kwargs):
        
        proxies = self.proxies
        
        #batch = torch.nn.functional.normalize(batch, dim=-1) # BS x dim
        
        if self.opt.mix_extension:
            proxies = self.proxies[labels]
            batch_mix = self.mixup(batch, proxies, mode = self.opt.mix_mode,
                                s = self.opt.shift_scale)
            batch = torch.cat((batch, batch), dim=0)
            batch_mix = torch.cat((batch_mix, proxies), dim=0)
            labels = torch.cat((labels, labels), dim=0)
            batch = F.normalize(batch, dim=-1)
            #batch = F.normalize(batch_mix, dim=-1)
            proxies = F.normalize(proxies, dim=-1)
        else:
            batch = F.normalize(batch, dim=-1) # BS x dim
            proxies = F.normalize(proxies, dim=-1)
        
        
        self.labels = labels.unsqueeze(1) # BS x 1
        self.u_labels = self.labels.view(-1) # BS  flat labels
        self.same_labels = (self.labels.T == self.labels.view(-1, 1)).to(
            batch.device).T # BS x BS  positive mask: if one label is same to the other in this batch
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
            torch.float).to(batch.device).T # BS x NClass  negative mask: invert of one-hot labels
        
        return self.compute_proxyloss(batch, proxies)

    def compute_proxyloss(self, batch, proxies):
        proxies = torch.nn.functional.normalize(self.proxies, dim=-1) # NClass x dim
        
        pos_sims = batch.mm(proxies[self.u_labels].T) # BS x BS, sims between samples and labeled proxies
        sims = batch.mm(proxies.T) # BS x NClass, sims between samples and [all Nclass proxies]
        w_pos_sims = -self.pars['pos_alpha'] * (pos_sims - self.pars['pos_delta'])
        w_neg_sims = self.pars['neg_alpha'] * (sims - self.pars['neg_delta'])
        
        pos_s = self.masked_logsumexp(w_pos_sims,
                                      mask=self.same_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        neg_s = self.masked_logsumexp(w_neg_sims,
                                      mask=self.diff_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        return pos_s.mean() + neg_s.mean()
    
    
    def unit_batch_normalization(self, batch, proxy):
        
        var, mean = torch.var_mean(torch.cat((batch, proxy)))
        
        self.ubn_batch = (batch - mean)/torch.sqrt(var)
        self.ubn_proxy = (proxy - mean)/torch.sqrt(var)
        
        return self.ubn_batch, self.ubn_proxy
    
    def masked_logsumexp(self, sims, dim=0, mask=None):
        # select features by mask
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/\
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1 # select between nca and anchor loss
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims
        
    
    def mixup(self, t1, t2, mode='random', s=1.0):
        '''
        t1, t2: BS x dim
        mix up alpha: 1.0
        shift: -1<s<1
        s<0 shift to t2
        s>0 shift to t1 
        '''
        bs = t1.shape[0]
        
        assert (s > -1.0 and s <= 1.0)
        if mode == 'random':
            #m = self.beta.sample()  # torch distribution has a wired bug with faiss
            m = np.random.beta(self.opt.mix_alpha, self.opt.mix_beta)
            #return m * s * t1 + (1.0 - m * s) * t2
            
            if s < 0:
                s = s + 1
                return m * s * t1 + (1.0 - m * s) * t2
            else:
                return (1.0 - m * s) * t1 + m * s * t2
        
       
        elif mode == 'individual':
        # learnable mixture parameter for each attributes
            
            f = torch.tile(self.mix_w, (bs, 1))
            return f * t1 + (1.0 - f) * t2
            
        elif mode == 'linear':
        # learnable linear parameter to project t1 and t2
            
            t = torch.cat((t1, t2), dim=1)
            return t.mm(self.mix_w)
       
        else:
            raise NotImplementedError('mixture method is not implemented.')
            
        return 0    



