#-*-coding:utf-8-*-

import numpy as np
import termcolor
from tqdm import tqdm

import parameters as par
import utilities.misc as misc
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasets as dataset_library
import criteria as criteria
import metrics as metrics
from metrics import inshop_recall

def test_parameters(parser):
    parser.add_argument('--dataset', default='cub200', type=str,
                        help='Dataset to use: cub200, cars196, online_products, inshop')
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'e_recall@10', 'f1', 'mAP_R'], 
                        type=str, help='Metrics to evaluate performance by.')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--source_path', default='../dml_data',   type=str,
                        help='Path to training data.')
    parser.add_argument('--batch_size', default=90 , type=int,
                        help='Mini-Batchsize to use. default=90')
    parser.add_argument('--evaluate_on_cpu', action='store_true',
                        help='Flag. If set, computed evaluation metrics on CPU instead of GPU.')
    parser.add_argument('--kernels', default=6, type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--evaltypes', nargs='+', default=['embeds'], type=str)
    parser.add_argument('--normal', action='store_true')
    parser.add_argument('--test_path', default='../Results/cub200/checkpoint_1.pth.tar')
    parser.add_argument('--infor_save_path', default='')
    
    return parser


def evaluation():
    parser = argparse.ArgumentParser()
    parser = test_parameters(parser)
    opt = parser.parse_args()
    
    opt.source_path += '/' + opt.dataset
    opt.evaluate_on_gpu = not opt.evaluate_on_cpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
    
    opt.not_pretrained = True
    opt.augmentation = 'base'
    
    opt.device = torch.device('cuda')
    opt.arch = torch.load(opt.test_path)['opt'].arch
    opt.embed_dim = torch.load(opt.test_path)['opt'].embed_dim
    model = archs.select(opt.arch, opt)
    _ = model.to(opt.device)
    datasets = dataset_library.select(opt.dataset, opt, opt.source_path)
    
    dataloaders = {}
    if not opt.dataset == 'inshop':
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
    
    model.load_state_dict(torch.load(opt.test_path)['state_dict'])
    
    _ = model.eval()
    
    print('\n' + termcolor.colored(
            'Computing Testing Metrics...', 'green', attrs=['bold']))
    
    
    ### Metric Computer.
    metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)
    
    if not opt.dataset == 'inshop':
        opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)
    else:
        opt.n_test_classes = len(dataloaders['gallery'].dataset.avail_classes)
        
    if not opt.dataset == 'inshop':
        computed_metrics, extra_infos = metric_computer.compute_standard(
            opt, model, dataloaders['testing'], opt.evaltypes, opt.device, mode='Val')
        # print eval
        numeric_metrics = {}
        histogr_metrics = {}
        for main_key in computed_metrics.keys():
            for name, value in computed_metrics[main_key].items():
                if isinstance(value, np.ndarray):
                    if main_key not in histogr_metrics:
                        histogr_metrics[main_key] = {}
                    histogr_metrics[main_key][name] = value
                else:
                    if main_key not in numeric_metrics:
                        numeric_metrics[main_key] = {}
                    numeric_metrics[main_key][name] = value     
        
        if len(opt.infor_save_path) > 0:
            savepath = opt.infor_save_path + '/extra_info_{}.npy'.format(opt.dataset)
            np.save(savepath, extra_infos['embeds'])
            print ('save extra_infos to {} \n'.format(savepath))
        
    else:
        # for inshop
        recall, keys = inshop_recall.evaluate_cos_Inshop(model, dataloaders['query'], dataloaders['gallery'])
        numeric_metrics = {}
        numeric_metrics['embed'] = {}
        for i in np.arange(len(keys)):
            key = keys[i]
            if key not in numeric_metrics:
                numeric_metrics['embed'][key] = {}
            numeric_metrics['embed'][key] = recall[i]
            
    full_result_str = ''
    for evaltype in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
        for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format('\n' if i>0 else '',metricname, metricval)    
        
        full_result_str += '\n'
        
    print(full_result_str)
    

if __name__ == '__main__':
    evaluation()
