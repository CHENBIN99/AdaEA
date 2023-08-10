"""
AdaEA base on MI-FGSM
"""
import torch
import torch.nn as nn
from attack.AdaEA_Base import AdaEA_Base


class AdaEA_MIFGSM(AdaEA_Base):
    def __init__(self, models, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), momentum=0.9, no_agm=False,
                 no_drf=False):
        super().__init__(models=models, eps=eps, alpha=alpha, max_value=max_value, min_value=min_value, threshold=threshold,
                         device=device, beta=beta, no_agm=no_agm,
                         no_drf=no_drf)
        self.iters = iters
        self.momentum = momentum

    def attack(self, data, label, idx=-1):
        B, C, H, W = data.size()
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        grad_mom = torch.zeros_like(data, device=self.device)

        for i in range(self.iters):
            adv_data.requires_grad = True

            outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
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

            # momentum
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + self.momentum * grad_mom
            grad_mom = grad

            # add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data
