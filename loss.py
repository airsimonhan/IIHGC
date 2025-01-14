import torch.nn.functional as F
import torch
from torch import einsum
class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, sorted_risk, censors):

        riskmax = F.normalize(sorted_risk, p=2, dim=0)

        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))

        return loss
