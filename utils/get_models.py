import os
import timm
import yaml
import torch
from torchvision import transforms
from utils.AverageMeter import AccuracyMeter

# checkpoint yaml file
yaml_path = '../configs/checkpoint.yaml'


def get_models(args, device):
    metrix = {}
    with open(os.path.join(args.root_path, 'utils', yaml_path), 'r', encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    print('🌟\tBuilding models...')
    models = {}
    if args.dataset == 'cifar10':
        save_root_path = r"checkpoint_file/cifar10"
        for key, value in yaml_data.items():
            models[key] = timm.create_model(value['full_name'],
                                            checkpoint_path=os.path.join(args.root_path, save_root_path,
                                                                         yaml_data[key]['ckp_path']),
                                            num_classes=10).to(device).eval()
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'cifar100':
        save_root_path = r"checkpoint_file/cifar100"
        for key, value in yaml_data.items():
            models[key] = timm.create_model(value['full_name'],
                                            checkpoint_path=os.path.join(args.root_path, save_root_path,
                                                                         yaml_data[key]['ckp_path']),
                                            num_classes=100).to(device).eval()
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')
    elif args.dataset == 'imagenet':
        # save_root_path = r"checkpoint/tinyimagenet"
        for key, value in yaml_data.items():
            model = timm.create_model(value['full_name'], pretrained=True, num_classes=1000).to(device)
            model.eval()
            if 'inc' in key or 'vit' in key or 'bit' in key:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
            else:
                models[key] = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  model)
            metrix[key] = AccuracyMeter()
            print(f'⭐\tload {key} successfully')

    return models, metrix

