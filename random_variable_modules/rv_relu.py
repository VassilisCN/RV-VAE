import torch, math
import numpy as np
import pyro
from torch._C import device


class RandomVariableReLU(torch.nn.Module):
    def __init__(self):
        super(RandomVariableReLU, self).__init__()
        self.normdis = torch.distributions.Normal(torch.tensor([0.0], device='cuda'), torch.tensor([1.0], device='cuda')) # carefull with device

    def forward(self, input):
        """
        Input: (N,2,∗) where * means, any number of additional dimensions
        Output: (N,2,∗)
        """
        epsilon = 1e10 # epsilon for negative variances later
        means = input[:, 0, ...] 
        varis = input[:, 1, ...]
        
        a = -means/(varis**0.5)
        a = a.double() # temporary double precision for more accurate estimations
        
        phi_a = self.normdis.log_prob(a).exp()
        Phi_a = self.normdis.cdf(a)
        # E[max(X,c)] = E[X∣X>c]Pr[X>c]+E[c∣X≤c]Pr[X≤c] 
        # E[X∣X>c] = Lower tail truncated normal distribution mean
        # Pr[X>c] = 1 - Φ(a)
        # E[c∣X≤c] = c
        # a = (c - μ) / σ
        # can add abs() on varis for the pytorch "bug"
        est_means_clean = (1-Phi_a)*means + (varis**0.5)*phi_a 
        # Var[max(X,c)] = Var[X|X>c]Pr[X>c]+E[X|X>c]^2(1-Pr[X>c])Pr[X>c]
        # Var[X|X>c] = Lower tail truncated normal distribution variance
        # Sometimes this gives variances < 0 but its numeric error that in next layers is fixed by adding an other epsilon
        est_varis_raw = (1-Phi_a) * (varis + (means**2)*Phi_a) + (varis**0.5)*phi_a*(2*means*Phi_a - means - (varis**0.5)*phi_a)

        est_varis = torch.where(est_varis_raw<=0, epsilon, est_varis_raw).float()
        # if est_varis.isnan().any():
        #     print("HAS var NaNs!!")
        #     print(means[est_varis.isnan()], varis[est_varis.isnan()], est_varis[est_varis.isnan()])
        # if est_means_clean.isnan().any():
        #     print("HAS mean NaNs!!")
        #     print(means[est_means_clean.isnan()], varis[est_means_clean.isnan()], est_means_clean[est_means_clean.isnan()])
        means = torch.unsqueeze(est_means_clean.float(), 1) # At dim = 1 we have our dist_params
        variances = torch.unsqueeze(est_varis, 1)
        output = torch.cat((means, variances), 1)
        return output


if __name__ == "__main__":
    device = 'cuda:0'
    i = torch.ones(1,2,1,1,1, device=device)
    i[:,0,...] = 4.4292#3.4028234663852886e+28
    i[:,1,...] = torch.tensor([2.7096]) # 
    i.requires_grad = True

    # torch.cuda.synchronize()
    # start_max_memory = torch.cuda.max_memory_allocated() / 1024**2
    # start_memory = torch.cuda.memory_allocated() / 1024**2
    
    r = RandomVariableReLU().to(device)
    y = r(i)
    o = sum(y.view(-1))
    o.backward()
    print(y)
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     for _ in range(1000):
    #         y = r(i)
    #         o = sum(y.view(-1))
    #         o.backward()
    # print(prof)
    # Measure allocated memory after the call
    # torch.cuda.synchronize()
    # end_max_memory = torch.cuda.max_memory_allocated() / 1024**2
    # end_memory = torch.cuda.memory_allocated() / 1024**2

    # print(end_max_memory-start_max_memory, end_memory-start_memory)
    # print(y,i.grad)