import torch
from tqdm import tqdm
from utils.get_attack import get_attack
from utils.get_dataset import get_dataset
from utils.get_models import get_models
from utils.tools import same_seeds, get_project_path
import argparse


def get_args():
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
    parser.add_argument('--num_worker', type=int, default=4)

    parser.add_argument('--attack_method', type=str)
    parser.add_argument('--fusion_method', type=str, default='add')
    parser.add_argument('--no_norm', action='store_true',
                        help='do not use normalization')
    parser.add_argument('--use_adv_model', action='store_true')

    # attack parameters
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='default momentum value')
    parser.add_argument('--resize_rate', type=float, default=0.9,
                        help='resize rate')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='diversity_prob')
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--min_value', type=float, default=0.0)

    # AdaEA
    parser.add_argument('--threshold', type=float, default=-0.3)
    parser.add_argument('--beta', type=float, default=10)

    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    # dataset
    dataloader = get_dataset(args)
    # models
    models, metrix = get_models(args, device=device)
    ens_model = ['resnet18', 'inc_v3', 'vit_t', 'deit_t']
    print(f'ens model: {ens_model}')

    for idx, (data, label) in enumerate(tqdm(dataloader)):
        n = label.size(0)
        data, label = data.to(device), label.to(device)
        attack_method = get_attack(args, ens_models=[models[i] for i in ens_model], device=device, models=models)
        adv_exp = attack_method(data, label)

        for model_name, model in models.items():
            with torch.no_grad():
                r_clean = model(data)
                r_adv = model(adv_exp)
            # clean
            pred_clean = r_clean.max(1)[1]
            correct_clean = (pred_clean == label).sum().item()
            # adv
            pred_adv = r_adv.max(1)[1]
            correct_adv = (pred_adv == label).sum().item()
            metrix[model_name].update(correct_clean, correct_adv, n)

        if idx == 10:
            break

    # show result
    print('-' * 73)
    print('|\tModel name\t|\tNat. Acc. (%)\t|\tAdv. Acc. (%)\t|\tASR. (%)\t|')
    for model_name, _ in models.items():
        print(f"|\t{model_name.ljust(10, ' ')}\t"
              f"|\t{str(round(metrix[model_name].clean_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].adv_acc * 100, 2)).ljust(13, ' ')}\t"
              f"|\t{str(round(metrix[model_name].attack_rate * 100, 2)).ljust(8, ' ')}\t|")
    print('-' * 73)


if __name__ == '__main__':
    args = get_args()
    same_seeds(args.seed)
    root_path = get_project_path()
    setattr(args, 'root_path', root_path)
    main(args)
