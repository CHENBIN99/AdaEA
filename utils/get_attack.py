from attack import AdaEA_FGSM, AdaEA_IFGSM, AdaEA_MIFGSM, AdaEA_DIFGSM, AdaEA_TIFGSM


def get_attack(args, ens_models, device, models=None):
    # AdaEA
    if args.attack_method == 'AdaEA_FGSM':
        attack_method = AdaEA_FGSM.AdaEA_FGSM(
            ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
            beta=args.beta, device=device)
    elif args.attack_method == 'AdaEA_IFGSM':
        attack_method = AdaEA_IFGSM.AdaEA_IFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    elif args.attack_method == 'AdaEA_MIFGSM':
        attack_method = AdaEA_MIFGSM.AdaEA_MIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum)
    elif args.attack_method == 'AdaEA_DIFGSM':
        attack_method = AdaEA_DIFGSM.AdaEA_DIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
            momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
    elif args.attack_method == 'AdaEA_TIFGSM':
        attack_method = AdaEA_TIFGSM.AdaEA_TIFGSM(
            ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
            min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
    else:
        raise NotImplemented

    return attack_method
