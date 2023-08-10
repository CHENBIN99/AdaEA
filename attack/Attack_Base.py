"""
Attack Base
"""
import torch


class Attack_Base:
    def __init__(self, model, eps=8/255, max_value=1., min_value=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        # assert isinstance(models, list) and len(models) >= 2, 'Error'
        self.device = device
        self.model = model
        self.model.eval()

        # attack parameter
        self.eps = eps
        self.max_value = max_value
        self.min_value = min_value
        self.attack_step = 0

    def get_adv_example(self, ori_data, adv_data, grad):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        adv_example = adv_data.detach() + grad.sign() * self.attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    def attack(self, data, label, idx=-1):
        pass

    def __call__(self, data, label, idx=-1):
        self.data_size = data.size()
        self.label = label
        self.alpha_log = []
        self.distance = []

        return self.attack(data, label, idx)





