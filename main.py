import argparse
import os
import time
import warnings
import numpy as np
import termcolor
from tqdm import tqdm
import parameters as par
import yaml
import utilities.misc as misc
from metrics import query_gallery_metrics as recall
from evaluation import save_numpy

import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler as dsamplers
import datasets as dataset_library
import criteria as criteria
import metrics as metrics
import evaluation as eval
from utilities import misc
from utilities import logger

warnings.filterwarnings("ignore")

### ---------------------------------------------------------------
### INPUT ARGUMENTS
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.advproxy_parameters(parser)
opt = parser.parse_args()

# update setting from config
with open(opt.config) as file:
    if file is not None:
        _config = yaml.load(file, Loader=yaml.FullLoader)
        for key, value in _config.items():
            setattr(opt, key, value)    

full_training_start_time = time.time()

opt.source_path += '/' + opt.dataset
opt.save_path += '/' + opt.dataset

# Assert that the construction of the batch makes sense, i.e. the division into
# class-subclusters.
assert_text = 'Batchsize needs to fit number of samples per class for distance '
assert_text += 'sampling and margin/triplet loss!'
assert not opt.batch_size % opt.samples_per_class, assert_text

opt.pretrained = not opt.not_pretrained
opt.evaluate_on_gpu = not opt.evaluate_on_cpu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
misc.set_seed(opt.seed)

### ---------------------------------------------------------------
### Embedding Network model (resnet, inception).
opt.device = torch.device('cuda')
model = archs.select(opt.arch, opt)

if hasattr(model, 'optim_dict_list') and len(model.optim_dict_list):
    to_optim = model.optim_dict_list
else:
    to_optim = [{
        'params': model.parameters(),
        'lr': opt.lr,
        'weight_decay': opt.decay
    }]
   

_ = model.to(opt.device)

### Datasetse & Dataloaders.
datasets = dataset_library.select(opt.dataset, opt, opt.source_path)

dataloaders = {}
if not opt.dataset == 'inshop':
    dataloaders['evaluation'] = torch.utils.data.DataLoader(
                datasets['evaluation'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
    dataloaders['testing'] = torch.utils.data.DataLoader(
                datasets['testing'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
else:
    dataloaders['query'] = torch.utils.data.DataLoader(
                datasets['query'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
    dataloaders['gallery'] = torch.utils.data.DataLoader(
                datasets['gallery'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)

train_data_sampler = dsamplers.select(opt.data_sampler, opt,
                                      datasets['training'].image_dict,
                                      datasets['training'].image_list)
if train_data_sampler.requires_storage:
    train_data_sampler.create_storage(dataloaders['evaluation'], model,
                                      opt.device)

dataloaders['training'] = torch.utils.data.DataLoader(
    datasets['training'],
    num_workers=opt.kernels,
    batch_sampler=train_data_sampler)

opt.n_classes = len(dataloaders['training'].dataset.avail_classes)
if not opt.dataset == 'inshop':
    opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)
    metric_evaluation_keys = ['testing', 'evaluation']
    
else:
    opt.n_test_classes = len(dataloaders['gallery'].dataset.avail_classes)

### Create logging setup.
sub_loggers = ['Train', 'Test', 'Model Grad']

LOG = logger.LOGGER(opt,
                    sub_loggers=sub_loggers,
                    start_new=True)

adv_trainer = True if opt.loss == 'dadaproxy' else False;

### Criterion.
criterion, to_optim = criteria.select(opt.loss, opt, to_optim)
_ = criterion.to(opt.device)

### Optimizer.
if opt.optim == 'adam':
    optimizer = torch.optim.Adam(to_optim)
elif opt.optim == 'adamw':
    optimizer = torch.optim.AdamW(to_optim)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))

if opt.scheduler == 'multi':
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=opt.lr_reduce_multi_steps,
                                                 gamma=opt.lr_reduce_rate)
elif opt.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_reduce_step,
                                            gamma=opt.lr_reduce_rate)
elif opt.scheduler == 'linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                            start_factor=opt.lr_linear_start,
                                            end_factor=opt.lr_linear_end,
                                            total_iters=opt.lr_linear_length)
# adversarial optimizer for dadaproxy
if adv_trainer:
        dis_optimizer = torch.optim.Adam(criterion.dis_optim_dict_list)
        proxy_optimizer = torch.optim.Adam(criterion.proxy_optim_dict_list)
        
        if opt.scheduler == 'multi':
            
            dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(dis_optimizer,
                                                 milestones=opt.lr_reduce_multi_steps,
                                                 gamma=opt.lr_reduce_rate)
            proxy_scheduler = torch.optim.lr_scheduler.MultiStepLR(proxy_optimizer,
                                                 milestones=opt.lr_reduce_multi_steps,
                                                 gamma=opt.lr_reduce_rate)                                     
        elif opt.scheduler == 'step':
        
            dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optimizer, 
                                            step_size=opt.lr_reduce_step,
                                            gamma=opt.lr_reduce_rate)
            proxy_scheduler = torch.optim.lr_scheduler.StepLR(proxy_optimizer, 
                                            step_size=opt.lr_reduce_step,
                                            gamma=opt.lr_reduce_rate)
        elif opt.scheduler == 'linear':
            dis_scheduler = torch.optim.lr_scheduler.LinearLR(dis_optimizer,
                                            start_factor=opt.lr_linear_start,
                                            end_factor=opt.lr_linear_end,
                                            total_iters=opt.lr_linear_length)
            proxy_scheduler = torch.optim.lr_scheduler.LinearLR(proxy_optimizer,
                                            start_factor=opt.lr_linear_start,
                                            end_factor=opt.lr_linear_end,
                                            total_iters=opt.lr_linear_length)

### Metric Computer.
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)


### ---------------------------------------------------------------
### Summary.
data_text = 'Dataset:\t {}'.format(opt.dataset)
setup_text = 'Objective:\t {}'.format(opt.loss)
arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch,
                                                   misc.gimme_params(model))
summary = data_text + '\n' + setup_text + '\n' + '\n' + arch_text
print(summary)

### ---------------------------------------------------------------
#### Training routing.

iter_count = 0
loss_args = {
    'batch': None,
    'labels': None,
    'batch_features': None,
    'f_embed': None
}

opt.epoch = 0
epochs = range(opt.epoch, opt.n_epochs)
scaler = torch.cuda.amp.GradScaler()

torch.cuda.empty_cache()

max_r1 = 0.0

for epoch in epochs:
    opt.epoch = epoch
    
    warmup = (epoch < opt.warmup)
    d_warmup = (epoch < opt.d_warmup)
    
    if warmup or d_warmup:
        print(termcolor.colored(' ========================= WARMUP ========================= ', 
        'yellow', attrs=['bold']))
    else:
        print('\n' + termcolor.colored('===============================================', 'red', 
        attrs=['bold']) + '\n')
    
    # Set seeds for each epoch - this ensures reproducibility after resumption.
    misc.set_seed(opt.seed + epoch)
    print('G LR: {}'.format(' | '.join(
            '{}'.format(x['lr']) for x in optimizer.param_groups)))
    
    if adv_trainer:
        print('D LR: {}'.format(dis_optimizer.param_groups[0]['lr']))
        print('Proxy LR: {}'.format(proxy_optimizer.param_groups[0]['lr']))
    
    epoch_start_time = time.time()
    
    # Train one epoch
    data_iterator = tqdm(dataloaders['training'],
                bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt}, '
                           '{elapsed}/{remaining}{postfix}]',
                ncols=96, ascii=True, desc='[Train Epoch %d]: ' % epoch)
                
    _ = model.train()
    
    criterion.new_epoch = True
    
    # for each minibatch
    for i, out in enumerate(data_iterator):
            
        loss_collect = []    
        class_labels, input_dict, sample_indices = out

        # load data to generator
        input = input_dict['image'].to(opt.device)
        
        model_args = {
                'x': input.to(opt.device),
                'warmup': warmup    # freeze model param in warmupu phase
        }

        
        # ============  Begin: adv discriminator trainer  ===============
        if adv_trainer:
            avg_loss = []
            with torch.no_grad():
                
                out_dict = model(**model_args)
                
                loss_args['batch'] = out_dict['embeds']
                loss_args['labels'] = class_labels   
                             
            criterion.save_mix = False
            for _ in range(opt.d_step):
                dis_optimizer.zero_grad()
                dloss = criterion.discriminator_forward(**loss_args)
                avg_loss.append(dloss.item())
                scaler.scale(dloss).backward()
                scaler.step(dis_optimizer)
                scaler.update()
                    
            loss_collect.append(np.mean(avg_loss))                
                
        # ===============  adv discriminator trainer end ================
        
        if not d_warmup:
            # optimize proxy for descrpency
            criterion.save_mix = False
            for j in range(opt.g_step):
                if j == 0:
                    criterion.save_mix = True
                
                avg_loss = []
                optimizer.zero_grad()
                if adv_trainer:
                    proxy_optimizer.zero_grad()
                out_dict = model(**model_args)
                
                loss_args['batch'] = out_dict['embeds']
                loss_args['batch_t'] = out_dict['embeds']
                loss_args['labels'] = class_labels
                loss = criterion(**loss_args)
                
                avg_loss.append(loss.item())
                scaler.scale(loss).backward()
                if adv_trainer:
                    scaler.step(proxy_optimizer)
                scaler.step(optimizer)
                scaler.update()
                
            
            loss_collect.append(np.mean(avg_loss))
        
        if d_warmup:
            loss_collect.append(0)
        
        iter_count += 1
        
        if adv_trainer:
            data_iterator.set_postfix_str('DisL:{0:.4f}, DML:{1:.4f}'.format(
                loss_collect[0], loss_collect[1]))

        else:
            data_iterator.set_postfix_str('Loss: {0:.4f}'.format(
                loss_collect[0]))
    
    
    result_metrics = {'loss': loss_collect[0]}
    
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log(
        'time', np.round(time.time() - epoch_start_time, 4))
    
    if opt.scheduler != 'none' and (not warmup) and (not d_warmup):
        scheduler.step()
        if adv_trainer:
            dis_scheduler.step()
            proxy_scheduler.step()
    
    
    
    # Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    aux_store = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'dataloaders': dataloaders,
        'datasets': datasets,
        'train_data_sampler': train_data_sampler
    }
    
    # regular save
    if opt.save_results and opt.save_step > 0 and epoch % opt.save_step == 0:
        #criterion
        save_numpy(model, opt, 
                LOG.prop.save_path + '/np_saved_epoch_{}.npz'.format(epoch), criterion)
        
    
    
    # clean memory
    torch.cuda.empty_cache()
    
    
    if not opt.dataset == 'inshop':
        # run test metrics
        if not opt.no_test_metrics:
            print('\n' + termcolor.colored(
                'Computing Testing Metrics...', 'green', attrs=['bold']))
            eval.evaluate(opt.dataset,
                      LOG,
                      metric_computer, [dataloaders['testing']],
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Test',
                      aux_store=aux_store,
                      criterion=criterion)
    
        # run train metrics
        if not opt.no_train_metrics:
            print('\n' + termcolor.colored(
                'Computing Training Metrics...', 'green', attrs=['bold']))
            eval.evaluate(opt.dataset,
                      LOG,
                      metric_computer, [dataloaders['evaluation']],
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Train',
                      aux_store=aux_store)
    else:
        print('\n' + termcolor.colored(
                'Computing Testing Metrics...', 'green', attrs=['bold']))
        eval.evaluate(opt.dataset,
                      LOG,
                      recall, dataloaders,
                      model,
                      opt,
                      opt.evaltypes,
                      opt.device,
                      log_key='Test',
                      aux_store=aux_store)
        
    LOG.update(all=True)
    print('\nTotal Epoch Runtime: {0:4.2f}s'.format(time.time() -
                                                    epoch_start_time))
    print('\n' + termcolor.colored('===============================================', 'red', 
                attrs=['bold']) + '\n')

"""======================================================="""

