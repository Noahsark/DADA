import torch

import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        
        super(Criterion, self).__init__()

        self.opt = opt
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim
        
        self.name = 'DADA' # advproxy
        
        self.mle = nn.NLLLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.relu = nn.ReLU()
        self.beta = torch.distributions.beta.Beta(torch.tensor([self.opt.mix_alpha]), 
                                                  torch.tensor([self.opt.mix_alpha]))
        
        # save prediction correlations
        self.s_predictions = np.zeros((opt.n_classes, opt.n_classes))
        self.t_predictions = np.zeros((opt.n_classes, opt.n_classes))
        self.proxy_sims = np.zeros((opt.n_classes, opt.n_classes))
        
        self.proxy_sims_list = {}
        self.source_pred_list = {}
        self.target_pred_list = {}
        for i in range(opt.n_classes):
            self.source_pred_list[i] = []
            self.target_pred_list[i] = []
            self.proxy_sims_list[i] = []
        
        self.pars = {
            'pos_alpha': opt.loss_oproxy_pos_alpha,
            'pos_delta': opt.loss_oproxy_pos_delta,
            'neg_alpha': opt.loss_oproxy_neg_alpha,
            'neg_delta': opt.loss_oproxy_neg_delta
        }
        
        self.class_idxs = torch.arange(self.num_proxies)
        
        # control flags
        self.new_epoch = True
        self.save_mix = False
        self.saved_batch_mix = None
        
        if opt.proxy_init_mode == 'norm':
            self.proxies = torch.randn(self.num_proxies, self.embed_dim) / 8
            self.proxies = Parameter(self.proxies)
        else:
            raise NotImplementedError('mode {} is not implemented!'.format(opt.proxy_init_mode))
        
        self.proxy_optim_dict_list = [{
            'params': self.proxies,
            'lr': opt.loss_adv_proxy_lr
        }]
        
        # prepare for adv
        self.uni_discriminator_a = Uni_Predictor(opt)       
        self.dis_discriminator_a = Dis_Predictor(opt)
        
        self.dis_lr = opt.loss_adv_classifier_lr
        
        self.dis_optim_dict_list = [{'params': self.uni_discriminator_a.parameters(), 
                                    'lr': self.dis_lr, 'weight_decay': opt.dis_decay}]
        
        self.dis_optim_dict_list += [{'params': self.dis_discriminator_a.parameters(), 
                                    'lr': self.dis_lr, 'weight_decay': opt.dis_decay}]
        
        if self.opt.mix_mode == 'individual':
            self.mix_w = self._par((1, self.opt.embed_dim))
            self.proxy_optim_dict_list += [{'params': self.mix_w, 'lr': self.dis_lr}]
        
        if self.opt.mix_mode == 'linear':
            self.mix_w = self._par((self.opt.embed_dim * 2, self.opt.embed_dim))
            self.dis_optim_dict_list += [{'params': self.mix_w, 'lr': self.dis_lr}]
        
        if self.opt.pos_class_mix:
            assert(self.opt.samples_per_class >= 2)
    
    def save_average(self):
        for i in range(self.opt.n_classes):
            self.s_predictions[i] = np.mean(self.source_pred_list[i], axis=0)
            self.t_predictions[i] = np.mean(self.target_pred_list[i], axis=0)
            self.proxy_sims[i] = np.mean(self.proxy_sims_list[i], axis=0)
            

    def _par(self, shape, mode='xavier_uniform'):
        if mode == 'xavier_uniform':
            return Parameter(init.xavier_uniform_(torch.empty(shape), gain=1.414))
        elif mode == 'xavier_norm':
            return Parameter(init.xavier_norm_(torch.empty(shape), gain=1.414))
        elif mode == 'norm':
            return Parameter(init.normal_(torch.empty(shape)))
        elif mode == 'zeros':
            return Parameter(torch.zeros(shape))
        else:
            raise NotImplementedError('mode {} is not implemented!'.format(mode))

    
    def _prepare_pos_mix(self, batch, labels, **kwargs):
        '''
            mix (a, p)
            return: concated batch,  concated proxies, concated label
        '''
        idx_a, idx_p = get_pos_pairs(None, labels)
        a_labels = labels[idx_a]    # a_labels = p_labels
        a_batch = batch[idx_a]
        p_batch = batch[idx_p]
        a_proxies = self.proxies[labels[idx_a]]
        p_proxies = self.proxies[labels[idx_p]]
        
        c_batch = torch.cat((batch, self._class_mixup(a_batch, p_batch)))
        c_proxies = torch.cat((self.proxies[labels], self._class_mixup(a_proxies, p_proxies)))
        c_labels = torch.cat((labels, a_labels))
        
        return c_batch, c_proxies, c_labels
        
    
    def forward(self, batch, labels, **kwargs):
        
        if self.opt.pos_class_mix:
            batch, self.ex_proxies, labels = self._prepare_pos_mix(batch, labels)
        
        # don't change original proxy loss
        # because mixup won't change any proxy
        self.proxy_loss = self.compute_proxyloss(batch, labels, **kwargs)
        self.adv_loss = self.adv_forward(batch, labels, **kwargs)
        
        return self.opt.oproxy_ratio * self.proxy_loss + self.opt.dis_gamma * self.adv_loss
        
    
    def adv_forward(self, batch, labels, **kwargs):

        self.u_dis_loss = (1.0 - self.opt.d_dis_ratio) * \
                    self.compute_unit_discriminator(batch, labels, **kwargs)
        self.d_dis_loss = self.opt.d_dis_ratio * self.compute_dis_discriminator(
                                                batch, labels, save=True, **kwargs
                                                )[2] #s_loss + dis_loss
        
        return self.d_dis_loss - self.u_dis_loss
        
        
    
    def discriminator_forward(self, batch, labels, **kwargs):
        
        if self.opt.pos_class_mix:
            batch, self.ex_proxies, labels = self._prepare_pos_mix(batch, labels)
        
        self.u_loss = (1.0 - self.opt.d_dis_ratio) * \
                   self.compute_unit_discriminator(batch, labels, **kwargs)
        
        self.d_loss = self.opt.d_dis_ratio * self.compute_dis_discriminator(
                                                batch, labels, **kwargs
                                            )[0] # s_loss - dis_loss
                                        
        return self.u_loss + self.d_loss
        
        
    def compute_dis_discriminator(self, batch, labels, save=False, **kwargs):
        '''
        batch: BS x dim
        labels: BS
        batch_proxies: BS x dim
        '''
        
        # gradient pass has a bug if pass a copy
        if not self.opt.pos_class_mix:
            proxies = self.proxies[labels]
        else:
            proxies = self.ex_proxies
        
        batch_mix = self.mixup(batch, proxies, mode = self.opt.mix_mode,
                                s = self.opt.shift_scale)
        
        #extend batch
        if self.opt.mix_extension:
            batch = torch.cat((batch, batch), dim=0)
            batch_mix = torch.cat((batch_mix, proxies), dim=0)
            labels = torch.cat((labels, labels), dim=0)
        
        if self.opt.normal:
            batch = F.normalize(batch, dim=-1)
            batch_mix = F.normalize(batch_mix, dim=-1)
            proxies = F.normalize(proxies, dim=-1)
        
        s_output_a = self.dis_discriminator_a(batch)
        t_output_a = self.dis_discriminator_a(batch_mix)
   
        s_loss = 2.0 * self.ce(s_output_a, labels.to(self.opt.device))                         
        dis_loss = self.opt.d_discrepancy_ratio * self.discrepancy(s_output_a, t_output_a, 
                                                                    self.opt.dis_mode)
        
        if save and self.opt.save_results:
            bs = batch.shape[0]
            proxies_norm = F.normalize(self.proxies, dim=-1)
            s_output_a = torch.softmax(s_output_a, dim=-1)
            t_output_a = torch.softmax(t_output_a, dim=-1)
            sims = batch.mm(proxies_norm.T) # BS x NClass
            for i in range(bs):      
                self.source_pred_list[labels[i].item()].append(s_output_a[i].detach().cpu().numpy())
                self.target_pred_list[labels[i].item()].append(t_output_a[i].detach().cpu().numpy())
                self.proxy_sims_list[labels[i].item()].append(sims[i].detach().cpu().numpy())
                
        return s_loss - dis_loss, dis_loss, s_loss + dis_loss
    
    
    def compute_unit_discriminator(self, batch, labels, **kwargs):
        '''
        batch: BS x dim
        labels: BS
        batch_proxies: BS x dim
        '''
        BS = batch.shape[0]
        T0 = torch.zeros([BS]).long().to(self.opt.device)
        T1 = torch.ones([BS]).long().to(self.opt.device)
        T2 = (torch.ones([BS]) + 1.0).long().to(self.opt.device)
        
        if not self.opt.pos_class_mix:
            proxies = self.proxies[labels]
        else:
            proxies = self.ex_proxies
        
        batch_mix = self.mixup(batch, proxies, mode = self.opt.mix_mode,
                               s = self.opt.shift_scale)
        
        if self.save_mix:
            if self.saved_batch_mix == None or self.new_epoch:
                self.saved_batch_mix = batch_mix.detach()
                self.new_epoch = False
            else:
                self.saved_batch_mix = torch.cat((self.saved_batch_mix, batch_mix.detach()), dim=0)
        
        if self.opt.mix_extension:
            batch = torch.cat((batch, batch), dim=0)
            batch_mix = torch.cat((batch_mix, proxies), dim=0)
            T0 = torch.cat((T0, T0), dim=0)
            T1 = torch.cat((T1, T1), dim=0)
            T2 = torch.cat((T2, T2), dim=0)
            
        if self.opt.normal:
            batch = F.normalize(batch, dim=-1)
            batch_mix = F.normalize(batch_mix, dim=-1)
            proxies = F.normalize(proxies, dim=-1)
        
        input_ss = self.uni_discriminator_a(batch)
        input_st = self.uni_discriminator_a(batch_mix)
        

        # 3 domain + 3 loss
        input_tt = self.uni_discriminator_a(proxies)
        return (self.ce(input_ss, T0) + self.ce(input_st, T1) + 
            self.ce(input_tt, T2))
        
    
    def discrepancy(self, out1, out2, dis_mode='l1'):
        
        if dis_mode == 'l1':
            return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))
        if dis_mode == 'nwd':
            return (torch.norm(F.softmax(out1, dim=1), 'nuc') - \
                             torch.norm(F.softmax(out2, dim=1), 'nuc')) / out1.shape[0]
        else:
            raise NotImplementedError("discrepancy {} not exist!".format(dis_mode))
    
    def _class_mixup(self, t1, t2):
        
        m = np.random.beta(2.0, 2.0)
        return m * t1 + (1.0 - m) * t2
    
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
        #pdb.set_trace()
        if mode == 'random':
            # torch distribution has a wired bug with faiss
            m = np.random.beta(self.opt.mix_alpha, self.opt.mix_beta)
            
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

    
    def compute_proxyloss(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS)
        """
        batch = F.normalize(batch, dim=-1) # BS x dim
        self.labels = labels.unsqueeze(1) # BS x 1
        self.u_labels = self.labels.view(-1) # BS  flat labels
        self.same_labels = (self.labels.T == self.labels.view(-1, 1)).to(
            batch.device).T # BS x BS  positive mask: if one label is same to the other in this batch
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
            torch.float).to(batch.device).T # BS x NClass  negative mask: invert of one-hot labels
            
        proxies_norm = F.normalize(self.proxies, dim=-1)
        if not self.opt.pos_class_mix:
            proxies = proxies_norm[self.u_labels] # BS x dim
        else:
            proxies = F.normalize(self.ex_proxies, dim=-1) # ex does not create new proxy
        
        pos_sims = batch.mm(proxies.T) # BS x BS, sims between samples and labeled proxies
        
        sims = batch.mm(proxies_norm.T) # BS x NClass, sims between samples and [all Nclass proxies]
        
        w_pos_sims = -self.pars['pos_alpha'] * (pos_sims - self.pars['pos_delta'])
        w_neg_sims = self.pars['neg_alpha'] * (sims - self.pars['neg_delta'])
        
        pos_s = self.masked_logsumexp(w_pos_sims, mask=self.same_labels.type(torch.bool), dim=self.opt.proxy_dim)
        neg_s = self.masked_logsumexp(w_neg_sims, mask=self.diff_labels.type(torch.bool), dim=self.opt.proxy_dim)
        return pos_s.mean() + neg_s.mean()
    
    
    def masked_logsumexp(self, sims, dim=0, mask=None):
        # select features by mask
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        # dim = 0: ProxyAnchor
        # dim = 1: ProxyNCA
        
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1 # select between nca and nca anchor loss
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims
    
    
    
# unit class predictor
class Uni_Predictor(nn.Module):
    def __init__(self, opt):
        super(Uni_Predictor, self).__init__()
        self.opt = opt
        if self.opt.active_func == 'relu':
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(p=self.opt.dropout)
        self.fc1 = nn.Linear(opt.embed_dim, self.opt.fd_fc1_dim)
        self.bn1_fc = nn.BatchNorm1d(self.opt.fd_fc1_dim)
        self.fc2 = nn.Linear(self.opt.fd_fc1_dim, 3)
        self.bn2_fc = nn.BatchNorm1d(3)
        
    
    def forward(self, x):
        x = self.fc1(x)
        
        x = self.bn1_fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.opt.fd2_bn:
            x = self.bn2_fc(x)
        return x
    
# discrepancy class predictor
class Dis_Predictor(nn.Module):
    def __init__(self, opt):
        super(Dis_Predictor, self).__init__()
        self.opt = opt
        
        if self.opt.active_func == 'relu':
            self.relu = nn.ReLU()
        else:
            self.relu = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(p=self.opt.dropout)
        self.fc1 = nn.Linear(opt.embed_dim, self.opt.fc_fc1_dim)
        self.bn1_fc = nn.BatchNorm1d(self.opt.fc_fc1_dim) 
        self.fc2 = nn.Linear(self.opt.fc_fc1_dim, self.opt.fc_fc2_dim)
        self.bn2_fc = nn.BatchNorm1d(self.opt.fc_fc2_dim) 
        self.fc3 = nn.Linear(self.opt.fc_fc2_dim, self.opt.n_classes)
        self.bn3_fc = nn.BatchNorm1d(self.opt.n_classes) 
        
    def forward(self, x):
        x = self.fc1(x)

        x = self.bn1_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.bn2_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        if self.opt.fc2_bn:
            x = self.bn3_fc(x)
        
        return x


# mixup utilities
# adapted from https://github.com/KevinMusgrave/pytorch-metric-learning

def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs

def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx
       
       
def convert_to_pairs(indices_tuple, labels, ref_labels=None):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, ref_labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n


def get_pos_pairs(indices_tuple, labels):
    '''
        convert_to_pos_pairs_with_unique_labels
    '''
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    _, unique_idx = np.unique(labels[a].cpu().numpy(), return_index=True)
    return a[unique_idx], p[unique_idx]


