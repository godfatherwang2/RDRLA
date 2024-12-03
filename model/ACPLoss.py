from torch.nn import Parameter
import math
import torch
import torch.nn as nn

def l2_norm(input, axis = 1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output
class ACPLoss(nn.Module):
    def __init__(self, in_features, out_features,d_rate=0,m = 0.65, s = 64.,weight = 0.003):
        super(ACPLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = weight
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)
        self.drop_layer = nn.Dropout(d_rate)

    def forward(self,embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        with torch.no_grad():
            center = self.kernel.t().clone()[label]
        c_norm = l2_norm(center, axis=1)
        cos_theta_center = torch.sum(embbedings * c_norm, dim=1)
        cos_theta_center = cos_theta_center.clamp(-1, 1)
        center_loss = torch.pow(torch.acos(cos_theta_center), 1.5).mean(dim=-1)
        output = self.drop_layer(output)
        return output, origin_cos * self.s,self.s*self.weight*center_loss