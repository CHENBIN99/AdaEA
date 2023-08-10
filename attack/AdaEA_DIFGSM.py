"""
AdaEA base on DI-FGSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attack.AdaEA_Base import AdaEA_Base


class AdaEA_DIFGSM(AdaEA_Base):
    def __init__(self, models, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10, momentum=0.9,
                 resize_rate=0.9, diversity_prob=0.5, no_agm=False, no_drf=False):
        super().__init__(models=models, eps=eps, max_value=max_value, min_value=min_value, threshold=threshold,
                         device=device, beta=beta, no_agm=no_agm, no_drf=no_drf)
        self.alpha = alpha
        self.iters = iters
        self.momentum = momentum
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data, label = data.clone().detach().to(self.device), label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        grad_mom = torch.zeros_like(data, device=self.device)

        for i in range(self.iters):
            adv_data.requires_grad = True

            outputs = [self.models[idx](self.input_diversity(adv_data)) for idx in range(len(self.models))]
            losses = [loss_func(outputs[idx], label) for idx in range(len(self.models))]
            grads = [torch.autograd.grad(losses[idx], adv_data, retain_graph=True, create_graph=False)[0]
                     for idx in range(len(self.models))]

            # AGM
            if not self.no_agm:
                if i == 0:
                    alpha = self.agm(ori_data=data, cur_adv=adv_data, grad=grads, label=label)
            else:
                alpha = torch.tensor([1 / self.num_models] * self.num_models, dtype=torch.float, device=self.device)

            # DRF
            if not self.no_drf:
                cos_res = self.drf(grads, data_size=(B, C, H, W))
                cos_res[cos_res >= self.threshold] = 1.
                cos_res[cos_res < self.threshold] = 0.
            else:
                cos_res = torch.ones([B, 1, H, W])

            output = torch.stack(outputs, dim=0) * alpha.view(self.num_models, 1, 1)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]
            grad = grad * cos_res

            # Momentum
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + self.momentum * grad_mom
            grad_mom = grad

            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data

