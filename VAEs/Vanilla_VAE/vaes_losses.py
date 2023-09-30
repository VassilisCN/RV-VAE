import torch, torchvision, cv2
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from vaes_models import *
import numpy as np

def VanillaLoss(x, x_hat, mean, log_var, celeba_trainset, rv):
    kld_weight = x.shape[0] / len(celeba_trainset) # Account for the minibatch samples from the dataset
    if rv:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        means = x_hat[:,0,...]
        varss = x_hat[:,1,...]
        var_loss = 50.*F.mse_loss(varss, torch.zeros(varss.shape, device=x_hat.device))
        recons_loss = F.mse_loss(means, x) + var_loss
        return recons_loss + kld_weight * kld_loss, kld_loss # smaller learning rate for the variances
    else:
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
        recons_loss = F.mse_loss(x_hat, x)
        return recons_loss + kld_weight * kld_loss, kld_loss

import math
def log_density_gaussian(x, mu, logvar):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

def BetaTCVAELoss(recons, input, mu, log_var, z, train_set, model, rv=False):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """

    weight = 1 #kwargs['M_N']  # Account for the minibatch samples from the dataset
    if rv:
        recons = recons[:, 0, ...]
        varss = recons[:, 1, ...]
        var_loss = 50*torch.nn.MSELoss()(varss, torch.zeros(varss.shape, device=recons.device))
        recons_loss =F.mse_loss(recons, input, reduction='sum') + var_loss
        z = z[:, 0, ...] # is this right???
    else:
        recons_loss =F.mse_loss(recons, input, reduction='sum')

    log_q_zx = log_density_gaussian(z, mu, log_var).sum(dim = 1)

    zeros = torch.zeros_like(z)
    log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim = 1)

    batch_size, latent_dim = z.shape
    mat_log_q_z = log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                            mu.view(1, batch_size, latent_dim),
                                            log_var.view(1, batch_size, latent_dim))

    # Reference
    # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    dataset_size = len(train_set)#(1 / kwargs['M_N']) * batch_size # dataset size
    strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(input.device)
    importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    importance_weights.view(-1)[1::batch_size] = strat_weight
    importance_weights[batch_size - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

    log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

    mi_loss  = (log_q_zx - log_q_z).mean()
    tc_loss = (log_q_z - log_prod_q_z).mean()
    kld_loss = (log_prod_q_z - log_p_z).mean()
    if rv:
        kld_loss *= 1e-3

    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    model.num_iter += 1
    anneal_rate = min(0 + 1 * model.num_iter / model.anneal_steps, 1)
    

    loss = recons_loss/batch_size + \
            model.alpha * mi_loss + \
            weight * (model.beta * tc_loss +
                        anneal_rate * model.gamma * kld_loss)
    
    return loss, recons_loss,kld_loss

def WAELoss(input, recons, z, rv, reg_weight = 100, kernel_type = 'imq', latent_var = 2.):

    batch_size = input.size(0)
    bias_corr = batch_size *  (batch_size - 1)
    reg_weight = reg_weight / bias_corr

    recons_loss =F.mse_loss(recons, input)

    mmd_loss = compute_mmd(z, reg_weight, kernel_type, latent_var)

    loss = recons_loss + mmd_loss
    return loss

def compute_kernel(x1, x2, kernel_type, latent_var):
    # Convert the tensors into row and column vectors
    D = x1.size(1)
    N = x1.size(0)

    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor

    """
    Usually the below lines are not required, especially in our case,
    but this is useful when x1 and x2 have different sizes
    along the 0th dimension.
    """
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)

    if kernel_type == 'rbf':
        result = compute_rbf(x1, x2, latent_var)
    elif kernel_type == 'imq':
        result = compute_inv_mult_quad(x1, x2, latent_var)
    else:
        raise ValueError('Undefined kernel type.')

    return result


def compute_rbf(x1, x2, z_var, eps = 1e-7):
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    sigma = 2. * z_dim * z_var

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result

def compute_inv_mult_quad(x1, x2, z_var, eps = 1e-7):
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by
            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * z_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result

def compute_mmd(z, reg_weight, kernel_type, latent_var):
    # Sample from prior (Gaussian) distribution
    prior_z = torch.randn_like(z)

    prior_z__kernel = compute_kernel(prior_z, prior_z, kernel_type, latent_var)
    z__kernel = compute_kernel(z, z, kernel_type, latent_var)
    priorz_z__kernel = compute_kernel(prior_z, z, kernel_type, latent_var)

    mmd = reg_weight * prior_z__kernel.mean() + \
            reg_weight * z__kernel.mean() - \
            2 * reg_weight * priorz_z__kernel.mean()
    return mmd
