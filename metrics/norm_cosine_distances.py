import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

def norm_cosine_distances(output1, output2):
    cos = nn.CosineSimilarity(dim=1)
    logits = cos(output1, output2)

    return 1 - logits

def norm_pearson_distances(output1, output2):
    cos = nn.CosineSimilarity(dim=1)
    pearson = cos(output1 - output1.mean(dim=1, keepdim=True),
                  output2 - output2.mean(dim=1, keepdim=True))
    return 1 - pearson


def pair_dot_product(output1, output2, tau=0.07, norm=True):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """
    if norm:
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)

    logits = torch.matmul(output1, output2.t()) / tau

    return logits


def norm_mahalanobis(output1, output2, tau=0.07, norm=True):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """
    if norm:
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)

    data = torch.cat((output1, output2), 0)
    cov = torch.cov(data.t())
    diff = output1 - output2

    left= torch.matmul(diff, cov)
    mahal = torch.matmul(left, diff.t())

    return mahal.diag()


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data 
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def pair_norm_cosine_regs(output1, output2, tau=1.0):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """

    a = output1.shape[0]
    b = output2.shape[0]

    output1 = output1.unsqueeze(1).expand(a, b, -1)
    output2 = output2.unsqueeze(0).expand(a, b, -1)

    cos = nn.CosineSimilarity(dim=2)
    logits = cos(output1, output2) / tau

    logits = torch.exp(logits)

    return logits


def pair_norm_cosine_distances(output1, output2, tau=1.0, mode='test', exp=False):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """

    a = output1.shape[0]
    b = output2.shape[0]

    output1 = output1.unsqueeze(1).expand(a, b, -1)
    output2 = output2.unsqueeze(0).expand(a, b, -1)

    cos = nn.CosineSimilarity(dim=2)
    logits = cos(output1, output2) / tau

    if mode is 'test':
        if exp:
            return np.exp(1) - torch.exp(logits)
        else:
            return 1 - logits

    else:
        return logits


def pair_norm_pearson_distances(output1, output2, tau=1.0, mode='test'):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (a * m Tensor)
        output2 (b * m Tensor)
    Returns:
        pair eculidean distances (a * b Tensor)
    """

    a = output1.shape[0]
    b = output2.shape[0]

    output1 = output1.unsqueeze(1).expand(a, b, -1)
    output2 = output2.unsqueeze(0).expand(a, b, -1)

    cos = nn.CosineSimilarity(dim=2)
    logits = cos(output1 - output1.mean(dim=2, keepdim=True),
                 output2 - output2.mean(dim=2, keepdim=True))

    if mode is 'test':
        return 1 - logits / tau
    else:
        return logits / tau




def pair_norm_cosine_distances_dim3(output1, output2):
    """Estimate the eculidean distances between output1 and output2

    Args:
        output1 (batch * a * m Tensor)
        output2 (batch * b * m Tensor)
    Returns:
        pair eculidean distances (batch * a * b Tensor)
    """
    batch1, a, _ = output1.size()
    batch2, b, _ = output2.size()
    assert batch1 == batch2
    output1 = output1.unsqueeze(2).expand(batch1, a, b, -1)
    output2 = output2.unsqueeze(1).expand(batch2, a, b, -1)

    cos = nn.CosineSimilarity(dim=3)
    logits = cos(output1, output2)
    return 1 - logits
