"""
AdaEA base on TI-FGSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attack.AdaEA_Base import AdaEA_Base

import numpy as np
from scipy import stats as st


class AdaEA_TIFGSM(AdaEA_Base):
    def __init__(self, models, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10, no_agm=False,
                 no_drf=False, kernel_name='gaussian', len_kernel=15, nsig=3, resize_rate=0.9, diversity_prob=0.5,
                 decay = 0.0):
        super().__init__(models=models, eps=eps, alpha=alpha, max_value=max_value, min_value=min_value, threshold=threshold,
                         device=device, beta=beta, no_agm=no_agm, no_drf=no_drf)
        self.iters = iters
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data, label = data.clone().detach().to(self.device), label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(data).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        for i in range(self.iters):
            adv_data.requires_grad = True

            outputs = [self.models[idx](self.input_diversity(adv_data)) for idx in range(len(self.models))]
            losses = [loss_func(outputs[idx], label) for idx in range(len(self.models))]
            grads = [torch.autograd.grad(losses[idx], adv_data, retain_graph=True, create_graph=False)[0]
                     for idx in range(len(self.models))]

            # AGM
            alpha = self.agm(ori_data=data, cur_adv=adv_data, grad=grads, label=label)

            # DRF
            cos_res = self.drf(grads, data_size=(B, C, H, W))
            cos_res[cos_res >= self.threshold] = 1.
            cos_res[cos_res < self.threshold] = 0.

            output = torch.stack(outputs, dim=0) * alpha.view(self.num_models, 1, 1)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]
            grad = grad * cos_res

            # TI-FGSM
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2,
                                        (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize,
                            size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(),
                                size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(
            low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
        ), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

