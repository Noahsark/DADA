import argparse
import os


def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--dataset', default='cub200', type=str,
                        help='Dataset to use: cub200, cars196, online_products')
    
    parser.add_argument('--no_train_metrics', action='store_false',
                        help='Flag. If set, no training metrics are computed and logged.')
    parser.add_argument('--no_test_metrics', action='store_true',
                        help='Flag. If set, no test metrics are computed and logged.')
    
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'e_recall@8', 'nmi', 'f1', 'mAP_R'], type=str,
                        help='Metrics to evaluate performance by.')
    
    ### setup utilization                     
    parser.add_argument('--evaltypes', nargs='+', default=['embeds'], type=str)
    
    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'], type=str,
                        help='Improvement in these metrics will trigger checkpointing.')
    parser.add_argument('--store_improvements', action='store_true',
                        help='If set, will store checkpoints whenever the storage metric improves.')
    parser.add_argument('--save_step', default = 0, type=int, help='save model for x epochs')
    parser.add_argument('--save_results', action='store_true', help='')
    
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        help='gpus')
    parser.add_argument('--save_name', default='test_cub200',   type=str,
                        help='Appendix to save folder name if any special information is to be included.')
    parser.add_argument('--source_path', default='../dml_data',   type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default='../Results', type=str,
                        help='Where to save models and logs.')
    parser.add_argument('--config', default='./scripts/config.yaml', type=str)
    
    ### General Optimization Parameters
    parser.add_argument('--lr',  default=0.0001, type=float,
                        help='Learning Rate for network parameters.')
    parser.add_argument('--n_epochs', default=50, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--kernels', default=4, type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--batch_size', default=90 , type=int,
                        help='Mini-Batchsize to use. default=90')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='Type of learning rate scheduling. Currently: step & multi, linear.')
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='Learning rate reduction after --lr_reduce_step epochs.')
    parser.add_argument('--lr_reduce_step', default=5, type=int,
                        help='step before reducing learning rate.')
    parser.add_argument('--lr_reduce_multi_steps', default=[20, 45, 70], nargs='+', type=int, 
                        help='milestones (epoch) before reducing learning rate.')
    parser.add_argument('--lr_linear_start', default=1.0, type=float)
    parser.add_argument('--lr_linear_end', default=0.01, type=float)
    parser.add_argument('--lr_linear_length', default=20, type=int)
    parser.add_argument('--decay', default=0.0004, type=float,
                        help='Weight decay for optimizer.')
    parser.add_argument('--augmentation', default='base', type=str,
                        help='image augmentation: base, big, adv, v2, auto')
    parser.add_argument('--warmup', default=1, type=int,
                        help='warmup stage: freeze model and train the last layer and others only.')
    parser.add_argument('--d_warmup', default=1, type=int, help='warmup stage for discriminator')
    parser.add_argument('--evaluate_on_cpu', action='store_true',
                        help='Flag. If set, computed evaluation metrics on CPU instead of GPU.')
    parser.add_argument('--internal_split', default=1, type=float,
                        help='Split parameter used for meta-learning extensions.')
    parser.add_argument('--optim', default='adam', type=str,
                        help='Optimizer to use.')
    parser.add_argument('--loss', default='dadaproxy', type=str,
                        help='Trainin objective to use. See folder <criteria> for available methods.')
    
    ### General Model Architecture Parameters
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Embedding dimensionality of the network. Note: dim=512, 128 or 64 is used in most papers.')
    parser.add_argument('--arch', default='resnet50_frozen_layernorm_double',  type=str,
                        help='Underlying network architecture. \
                        Frozen denotes that exisiting pretrained batchnorm layers are frozen, \
                        and normalize denotes normalization of the output embedding.')
                        
    parser.add_argument('--not_pretrained', action='store_true',
                        help='Flag. If set, does not initialize the backbone network with ImageNet pretraining.')
     
    return parser
    

def advproxy_parameters(parser):
    
    ### Learning Parameters of DADA
    parser.add_argument('--proxy_dim', default=0, type=int,
                        help='select between 0: proxy_anchor 1: proxy_nac')
    parser.add_argument('--loss_adv_classifier_lr', default=0.0005, type=float,
                        help='Learning rate multiplier for discriminators.')
    parser.add_argument('--loss_adv_proxy_lr', default=0.04, type=float,
                        help='Learning rate multiplier for adv proxies')
    parser.add_argument('--dis_gamma', default=1.0, type=float,
                        help='weight for all dis_loss')
    parser.add_argument('--d_dis_ratio', default=0.01, type=float) # ratio between u and d
    parser.add_argument('--d_discrepancy_ratio',default=0.5, type=float)
    parser.add_argument('--oproxy_ratio', default=0.0075, type=float) # 
    parser.add_argument('--dis_init_mode', default='xavier_uniform', type=str,
                        help='mode to init the parameters of discriminator')
    parser.add_argument('--proxy_init_mode', default='norm', type=str,
                        help='mode to init the proxies')
    
    ### Discriminator Model Architecture Parameters
    parser.add_argument('--active_func', default='relu')
    parser.add_argument('--fd_fc1_dim', default=512, type=int)
    parser.add_argument('--fc_fc1_dim', default=512, type=int)
    parser.add_argument('--fc_fc2_dim', default=128, type=int)
    parser.add_argument('--fd2_bn', action='store_true', help='use batchnorm in fc1') 
    parser.add_argument('--fc2_bn', action='store_true', help='use batchnorm in fc2') 
    parser.add_argument('--dis_decay', default = '0.0001', type=float)
    parser.add_argument('--dropout', default=0.5)
    
    ### Adversarial Learning Parameters
    parser.add_argument('--g_step', default=1, type=int, help='step of generator')
    parser.add_argument('--d_step', default=3, type=int, help='step of discriminator')
    parser.add_argument('--dis_mode', default='nwd', type=str, help='mode of categratory discriminator')
    parser.add_argument('--normal', action='store_false')
    
    ### mixup setting
    parser.add_argument('--mix_mode', default='random', type=str, 
                        help='random, linear, individual')
    parser.add_argument('--shift_scale', default=1.0, type=float)                    
    parser.add_argument('--pos_class_mix', action='store_true', help='synthetic mixup')
    parser.add_argument('--mix_extension', action='store_true')
    parser.add_argument('--mix_alpha', default=0.5, type=float, help='alpha for beta dist')
    parser.add_argument('--mix_beta', default=0.5, type=float, help='beta for beta dist')
    
    return parser




def loss_specific_parameters(parser):
    ### Settings of ProxyAnchor.
    parser.add_argument('--loss_oproxy_mode', default='anchor', type=str,
                        help='Proxy-method: anchor = ProxyAnchor, nca = ProxyNCA.')
    parser.add_argument('--loss_oproxy_lrmulti', default=200, type=float,
                        help='Learning rate multiplier for proxies.')
    parser.add_argument('--loss_oproxy_pos_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for positive sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_neg_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for negative sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_pos_delta', default=0.1, type=float,
                        help='Threshold for positive sample-proxy similarities')
    parser.add_argument('--loss_oproxy_neg_delta', default=-0.1, type=float,
                        help='Threshold for negative sample-proxy similarities')
    
    return parser


def batch_creation_parameters(parser):
    ### Parameters for batch sampling methods.
    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='Batch-creation method. Default <class_random> ensures that for each class, at least --samples_per_class samples per class are available in each minibatch.')
    parser.add_argument('--data_ssl_set', action='store_true',
                        help='Obsolete. Only relevant for SSL-based extensions.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    return parser
