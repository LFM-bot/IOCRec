import argparse
from src.train.trainer import load_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='IOCRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    # Contrast Learning Hyper Params
    parser.add_argument('--aug_types', default=['crop', 'mask', 'reorder'], help='augmentation types')
    parser.add_argument('--crop_ratio', default=0.2, type=float,
                        help='Crop augmentation: proportion of cropped subsequence in origin sequence')
    parser.add_argument('--mask_ratio', default=0.7, type=float,
                        help='Mask augmentation: proportion of masked items in origin sequence')
    parser.add_argument('--reorder_ratio', default=0.2, type=float,
                        help='Reorder augmentation: proportion of reordered subsequence in origin sequence')
    parser.add_argument('--all_hidden', action='store_false', help='all hidden states for cl')
    parser.add_argument('--tao', default=1., type=float, help='temperature for softmax')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='weight for contrast learning loss, only work when jointly training')
    parser.add_argument('--k_intention', default=4, type=int, help='number of disentangled intention')
    # Transformer
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--ffn_hidden', default=512, type=int, help='hidden dim for feed forward network')
    parser.add_argument('--num_blocks', default=3, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of head for multi-head attention')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='dropout rate for attention')
    parser.add_argument('--layer_norm_eps', default=1e-12, type=float, help='transformer layer norm eps')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')
    # Data
    parser.add_argument('--dataset', default='toys', type=str)
    # Training
    parser.add_argument('--epoch_num', default=150, type=int)
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--train_batch', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0, type=float, help='l2 normalization')
    parser.add_argument('--patience', default=10, help='early stop patience')
    parser.add_argument('--seed', default=-1, help='random seed, -1 means no fixed seed')
    parser.add_argument('--mark', default='', help='log suffix mark')
    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS', type=str, help='[LS (leave-one-out), LS_R@0.x, PS (pre-split)]')
    parser.add_argument('--eval_mode', default='full', help='[uni100, pop100, full]')
    parser.add_argument('--k', default=[5, 10, 20, 50], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@10', help='specifies which indicator to apply early stop')

    config = parser.parse_args()

    trainer = load_trainer(config)
    trainer.start_training()


