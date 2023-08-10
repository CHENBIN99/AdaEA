import argparse

parser = argparse.ArgumentParser(description='benchmark of cifar10')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_root', type=str, default='../checkpoint/',
                    help='the direction to save the dataset')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=32,
                    help='the batch size when training')
parser.add_argument('--image_size', type=int, default=224,
                    help='image size of the dataloader')
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='whether use gpu')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu_id')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--save_img', action='store_true')
parser.add_argument('--num_worker', type=int, default=4)

parser.add_argument('--attack_method', type=str)
parser.add_argument('--fusion_method', type=str, default='add')
parser.add_argument('--no_norm', action='store_true',
                    help='do not use normalization')
parser.add_argument('--use_adv_model', action='store_true')

# attack parameters
parser.add_argument('--eps', type=float, default=16/255)
parser.add_argument('--alpha', type=float, default=1.6/255)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--momentum', type=float, default=0.9,
                    help='default momentum value')
parser.add_argument('--resize_rate', type=float, default=0.9,
                    help='resize rate')
parser.add_argument('--diversity_prob', type=float, default=0.5,
                    help='diversity_prob')
parser.add_argument('--max_value', type=float, default=1.0)
parser.add_argument('--min_value', type=float, default=0.0)

# CrossAttack
parser.add_argument('--use_cos', action='store_true')
parser.add_argument('--use_uncertainty', action='store_true')
parser.add_argument('--threshold', type=float, default=-0.3)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--no_agm', action='store_true')
parser.add_argument('--no_drf', action='store_true')

# SVRE
parser.add_argument('--m', type=int, default=4)

# BASES
parser.add_argument('--victim', type=str)

# feature
parser.add_argument('--log_grad', action='store_true')
parser.add_argument('--log_cam', action='store_true')

# Save Path
parser.add_argument('--metrix_path', type=str, default='metrix_result')

# model combination
parser.add_argument('--model_combination', type=int, default=1)

args = parser.parse_args()
