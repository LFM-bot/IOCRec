import argparse
import torch.cuda
from easydict import EasyDict
from src.utils.utils import HyperParamDict

EXP_HYPER_LIST = {'Data': {'dataset': None, 'data_aug': None, 'seq_filter_len': None,
                           'if_filter_target': None, 'max_len': None},
                  'Pretraining': {'do_pretraining': None, 'pretraining_task': None, 'pretraining_epoch': None,
                                  'pretraining_batch': None, 'pretraining_lr': None, 'pretraining_l2': None},
                  'Training': {'epoch_num': None, 'train_batch': None,
                               'learning_rate': None, 'l2': None, 'patience': None,
                               'device': None, 'num_worker': None, 'seed': None},
                  'Evaluation': {'split_type': None, 'split_mode': None, 'eval_mode': None, 'metric': None, 'k': None,
                                 'valid_metric': None, 'eval_batch': None},
                  'Save': {'log_save': None, 'save': None, 'model_saved': None}}


def experiment_hyper_load(exp_config):
    hyper_types = EXP_HYPER_LIST.keys()
    for hyper_dict in EXP_HYPER_LIST.values():
        for hyper in hyper_dict.keys():
            hyper_dict[hyper] = getattr(exp_config, hyper)
    return list(hyper_types), EXP_HYPER_LIST


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_default_config():
    parser = HyperParamDict()
    # Model
    parser.add_argument('--model', default='URCL4SRec')
    # Data
    parser.add_argument('--dataset', default='toys', type=str,
                        choices=['home', 'grocery', 'grocery', 'yelp_s3', 'toys'])
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--seq_filter_len', default=0, type=int, help='filter seq less than 3')
    parser.add_argument('--if_filter_target', action='store_true',
                        help='if filter target appearing in previous sequence')
    parser.add_argument('--separator', default=' ', type=str, help='separator to split item sequence')
    parser.add_argument('--graph_type', default='None', type=str, help='do not use graph',
                        choices=['None', 'BIPARTITE', 'TRANSITION'])
    parser.add_argument('--max_len', default=50, type=int, help='max sequence length')
    parser.add_argument('--kg_data_type', default='pretrain', type=str, choices=['pretrain', 'jointly_train', 'other'])
    # Pretraining
    parser.add_argument('--do_pretraining', default=False, action='store_true')
    parser.add_argument('--pretraining_task', default='MISP', type=str, choices=['MISP', 'MIM', 'PID'],
                        help='pretraining task:' \
                             'MISP: Mask Item Prediction and Mask Segment Prediction' \
                             'MIM: Mutual Information Maximization' \
                             'PID: Pseudo Item Discrimination'
                        )
    parser.add_argument('--pretraining_epoch', default=10, type=int)
    parser.add_argument('--pretraining_batch', default=512, type=int)
    parser.add_argument('--pretraining_lr', default=1e-3, type=float)
    parser.add_argument('--pretraining_l2', default=0., type=float, help='l2 normalization')
    # Training
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--seed', default=1034, type=int, help="random seed, only -1 means don't set random seed")
    parser.add_argument('--train_batch', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0., type=float, help='l2 normalization')
    parser.add_argument('--patience', default=5, type=int, help='early stop patience')
    parser.add_argument('--device', default=get_device(), choices=['cuda:0', 'cpu'],
                        help='training on gpu or cpu, default gpu')
    parser.add_argument('--num_worker', default=0, type=int,
                        help='num_workers for dataloader, best: 6')
    parser.add_argument('--mark', default='', type=str,
                        help='mark of this run which will be added to the name of the log')

    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS', type=str,
                        help='LS: Leave-one-out splitting.'
                             'LS_R@0.2: use LS and a ratio 0.x of test data for validate if use valid_and_test.'
                             'PS: Pre-Splitting, prepare xx.train and xx.eval, also xx.test if use valid_and_test')
    parser.add_argument('--eval_mode', default='full', help='[uni100, uni200, full]')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--k', default=[5, 10], help='top k for each metric')
    parser.add_argument('--valid_metric', default='hit@10', help='specifies which indicator to apply early stop')
    parser.add_argument('--eval_batch', default=256, type=int)

    # save
    parser.add_argument('--log_save', default='log', type=str, help='log saving path')
    parser.add_argument('--save', default='save', type=str, help='model saving path')
    parser.add_argument('--model_saved', default=None, type=str)

    return parser


def config_override(model_config, cmd_config):
    default_config = get_default_config()
    command_args = set([arg for arg in vars(cmd_config)])
    # overwrite model config by cmd config
    for arg in vars(model_config):
        if arg in command_args:
            setattr(model_config, arg, getattr(cmd_config, arg))

    # overwrite default config by cmd config
    for arg in vars(default_config):
        if arg in command_args:
            setattr(default_config, arg, getattr(cmd_config, arg))

    # overwrite default config by model config
    for arg in vars(model_config):
        setattr(default_config, arg, getattr(model_config, arg))

    return default_config
