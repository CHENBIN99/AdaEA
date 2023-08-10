"""
Base of the AdaEA
"""
from abc import abstractmethod

import torch
import torch.nn.functional as F


class AdaEA_Base:
    def __init__(self, models, eps=8/255, alpha=2/255, max_value=1., min_value=0., threshold=0., beta=10, no_agm=False,
                 no_drf=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        assert isinstance(models, list) and len(models) >= 2, 'Error'
        self.device = device
        self.models = models
        self.num_models = len(self.models)
        for model in models:
            model.eval()

        # attack parameter
        self.eps = eps
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.beta = beta
        self.alpha = alpha

        # op
        self.no_agm = no_agm
        self.no_drf = no_drf

    def get_adv_example(self, ori_data, adv_data, grad, attack_step=None):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        if attack_step is None:
            adv_example = adv_data.detach() + grad.sign() * self.alpha
        else:
            adv_example = adv_data.detach() + grad.sign() * attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    def agm(self, ori_data, cur_adv, grad, label):
        """
        Adaptive gradient modulation
        :param ori_data: natural images
        :param cur_adv: adv examples in last iteration
        :param grad: gradient in this iteration
        :param label: ground truth
        :return: coefficient of each model
        """
        loss_func = torch.nn.CrossEntropyLoss()

        # generate adversarial example
        adv_exp = [self.get_adv_example(ori_data=ori_data, adv_data=cur_adv, grad=grad[idx])
                   for idx in range(self.num_models)]
        loss_self = [loss_func(self.models[idx](adv_exp[idx]), label) for idx in range(self.num_models)]
        w = torch.zeros(size=(self.num_models,), device=self.device)

        for j in range(self.num_models):
            for i in range(self.num_models):
                if i == j:
                    continue
                w[j] += loss_func(self.models[i](adv_exp[j]), label) / loss_self[i] * self.beta
        w = torch.softmax(w, dim=0)

        return w

    def drf(self, grads, data_size):
        """
        disparity-reduced filter
        :param grads: gradients of each model
        :param data_size: size of input images
        :return: reduce map
        """
        reduce_map = torch.zeros(size=(self.num_models, self.num_models, data_size[0], data_size[-2], data_size[-1]),
                                 dtype=torch.float, device=self.device)
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        reduce_map_result = torch.zeros(size=(self.num_models, data_size[0], data_size[-2], data_size[-1]),
                                        dtype=torch.float, device=self.device)
        for i in range(self.num_models):
            for j in range(self.num_models):
                if i >= j:
                    continue
                reduce_map[i][j] = sim_func(F.normalize(grads[i], dim=1), F.normalize(grads[j], dim=1))
            if i < j:
                one_reduce_map = (reduce_map[i, :].sum(dim=0) + reduce_map[:, i].sum(dim=0)) / (self.num_models - 1)
                reduce_map_result[i] = one_reduce_map

        return reduce_map_result.mean(dim=0).view(data_size[0], 1, data_size[-2], data_size[-1])

    @abstractmethod
    def attack(self,
               data: torch.Tensor,
               label: torch.Tensor,
               idx: int = -1) -> torch.Tensor:
        ...

    def __call__(self, data, label, idx=-1):
        return self.attack(data, label, idx)





