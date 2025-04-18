import torch
import torch.nn.functional as F
import numpy as np

def gaussian_elbo(x1,x2,z,sigma,mu,logvar):
    
    #
    # Problem 5b: Compute the evidence lower bound for the Gaussian VAE.
    #             Use the closed-form expression for the KL divergence from Problem 1.
    #
    
    reconstruction = 1/2 * (1 / sigma**2) * torch.sum((x1-x2)**2, dim=[1,2,3]).mean()

    divergence = 1/2 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1).mean()

    return reconstruction, divergence

def mc_gaussian_elbo(x1,x2,z,sigma,mu,logvar):

    #
    # Problem 5c: Compute the evidence lower bound for the Gaussian VAE.
    #             Use a (1-point) monte-carlo estimate of the KL divergence.
    #
    reconstruction = 1/2 * (1 / sigma**2) * torch.sum((x1-x2)**2, dim=[1,2,3]).mean()

    var = torch.exp(logvar)
    interm1 = -1/2 * torch.sum(np.log(2*np.pi) + logvar+((z-mu)**2)/var, dim=1)

    interm2 = -1/2*torch.sum(np.log(2*np.pi)+z**2, dim=1)

    divergence = (interm1-interm2).mean()

    return reconstruction, divergence

def cross_entropy(x1,x2):
    return F.binary_cross_entropy_with_logits(x1, x2, reduction='sum')/x1.shape[0]

def discrete_output_elbo(x1,x2,z,logqzx):

    #
    # Problem 6b: Compute the evidence lower bound for a VAE with binary outputs.
    #             Use a (1-point) monte carlo estimate of the KL divergence.
    #

    bce = F.binary_cross_entropy_with_logits(x1, x2, reduction='none')
    reconstruction = bce.view(bce.size(0), -1).sum(dim=1).mean()

    interm1 = -1/2*torch.sum(z**2+np.log(2*np.pi),dim=1)

    divergence = (-logqzx-interm1).mean()

    return reconstruction, divergence
