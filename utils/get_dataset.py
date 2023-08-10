import os
import torch
from torchvision import datasets, transforms

def get_dataset(args):
    if args.dataset == 'cifar10':
        setattr(args, 'num_classes', 10)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size, antialias=True),
        ])
        test_set = datasets.CIFAR10(root=os.path.join(args.root_path, 'data/cifar10/'), train=False, download=True,
                                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=args.num_worker)
    elif args.dataset == 'cifar100':
        setattr(args, 'num_classes', 100)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size, antialias=True),
        ])
        test_set = datasets.CIFAR100(root=os.path.join(args.root_path, 'data/cifar10/'), train=False, download=True,
                                     transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=args.num_worker)
    elif args.dataset == 'imagenet':
        setattr(args, 'num_classes', 1000)
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
        ])
        test_set = datasets.ImageFolder(root=os.path.join(args.root_path, './data', 'ImageNet', 'val'),
                                        transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=args.num_worker)
    else:
        raise NotImplemented

    return test_loader
