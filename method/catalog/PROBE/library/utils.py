import datetime
from typing import List, Optional

import numpy as np
import math
import torch
import torch.optim as optim
import torch.distributions.normal as normal_distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
from torch.autograd import Variable

import logging


DECISION_THRESHOLD = 0.5

""" 
Code is largely taken and modified from https://github.com/MartinPawelczyk/ProbabilisticallyRobustRecourse/
"""

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.tensor(1)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_jacobian(inputs, output, num_classes=1):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad
    grad = gradient(output, inputs)
    return grad


def compute_invalidation_rate_closed(torch_model, x, sigma2):
    # Compute input into CDF
    prob = torch_model(x)
    logit_x = torch.log(prob[0][1] / prob[0][0]).to(x.device) # logit of the positive class probability
    Sigma2 = sigma2 * torch.eye(x.shape[0]).to(x.device) # covariance matrix of the noise
    jacobian_x = compute_jacobian(x, logit_x, num_classes=1).reshape(-1)
    denom = torch.sqrt(sigma2) * torch.norm(jacobian_x, 2)
    arg = logit_x / denom
    
    # Evaluate Gaussian cdf
    normal = normal_distribution.Normal(loc=0.0, scale=1.0)
    normal_cdf = normal.cdf(arg)
    
    # Get invalidation rate
    ir = 1 - normal_cdf
    
    return ir


# def perturb_sample(x, n_samples, sigma2):
#     # stack copies of this sample, i.e. n rows of x.
#     X = x.repeat(n_samples, 1)
#     # sample normal distributed values
#     Sigma = torch.eye(x.shape[1]) * sigma2
#     eps = MultivariateNormal(
#         loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
#     ).sample((n_samples,))
    
#     return X + eps


def reparametrization_trick(mu, sigma2, n_samples):
    #var = torch.eye(mu.shape[1]) * sigma2
    std = torch.sqrt(sigma2).to(mu.device)
    epsilon = MultivariateNormal(loc=torch.zeros(mu.shape[0]), covariance_matrix=torch.eye(mu.shape[0]))
    epsilon = epsilon.sample((n_samples,)).to(mu.device)  # standard Gaussian random noise
    ones = torch.ones_like(epsilon).to(mu.device)
    random_samples = mu.reshape(-1) * ones + std * epsilon
    
    return random_samples


def compute_invalidation_rate(torch_model, random_samples):
    yhat = torch_model(random_samples)[:, 1]
    hat = (yhat > 0.5).float()
    ir = 1 - torch.mean(hat, 0)
    return ir


def probe_recourse(
    model: torch.nn.Module,
    x: np.ndarray,
    cat_feature_indices: List[int],
    binary_cat_features: bool = True,
    feature_costs: Optional[List[float]] = None,
    lr: float = 0.07,
    lambda_param: float = 5,
    y_target: List[int] = [0.45, 0.55],
    n_iter: int = 500,
    t_max_min: float = 0.15,
    norm: int = 1,
    clamp: bool = False,
    loss_type: str = "MSE",
    invalidation_target: float = 0.45,
    inval_target_eps: float = 0.005,
    noise_variance: float = 0.01
) -> np.ndarray:
    """
    Generate counterfactual explanation for a given input sample using PROBE method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)

    noise_variance = torch.tensor(noise_variance, device=device)

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y_target = torch.tensor(y_target, dtype=torch.float32, device=device)
    lamb = torch.tensor(lambda_param, dtype=torch.float32, device=device)

    x_new = Variable(x.clone(), requires_grad=True)

    # NOTE: x and all its clones are in the shape (1, num_features)
    # This is based on the original design, ill try to preseve it best I can.

    # x_new_enc is a copy of x_new with reconstructed encoding constraints of x_new
    # such that categorical data is either 0 or 1
    # go through the list of categorical features given to us from the
    # data module and use the list of encoded feature names to reconstruct the encoding constraints for the categorical features in x_new
    x_new_enc = x_new.clone()

    # print(f"This is the shape of x_new {x_new}")
    # print(f"these are the cat feature indices {cat_feature_indices}")

    for index in cat_feature_indices:
        x_new_enc[0][index] = torch.round(x_new_enc[0][index])

    optimizer = optim.Adam([x_new], lr=lr, amsgrad=True)

    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
        f_x_new = model(x_new)[0][1]
    else:
        loss_fn = torch.nn.BCELoss()
        f_x_new = model(x_new)[0][1]

    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=t_max_min)

    costs = []
    ces = []

    random_samples = reparametrization_trick(x_new, noise_variance, n_samples=1000)
    invalidation_rate = compute_invalidation_rate(model, random_samples)

    while (f_x_new <= DECISION_THRESHOLD) or (invalidation_rate > invalidation_target + inval_target_eps):

        for _ in range(n_iter):

            optimizer.zero_grad()

            f_x_new_binary = model(x_new)[0]

            cost = torch.dist(x_new, x, norm)

            invalidation_rate_c = compute_invalidation_rate_closed(model, x_new, noise_variance)

            loss_invalidation = invalidation_rate_c - invalidation_target

            loss_invalidation[loss_invalidation < 0] = 0

            loss = 3 * loss_invalidation + loss_fn(f_x_new_binary, y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            random_samples = reparametrization_trick(x_new, noise_variance, n_samples=10000)
            invalidation_rate = compute_invalidation_rate(model, random_samples)

            if clamp:
                x_new.clone().clamp_(0, 1)

            x_new_enc = x_new.clone()

            for index in cat_feature_indices:
                x_new_enc[0][index] = torch.round(x_new_enc[0][index])

            f_x_new = model(x_new)[0][1]
        
        if (f_x_new > DECISION_THRESHOLD) and (invalidation_rate < invalidation_target + inval_target_eps):
            
            costs.append(cost)
            ces.append(x_new)

            break
    
        lamb -= 0.10

        if datetime.datetime.now() - t0 > t_max:
            logging.info("Timeout")
            break
    
    if not ces:
        logging.info("No Counterfactual Explanation Found at that Target Rate - Try Different Target")
    else:
        logging.info("Counterfactual Explanation Found")
        costs = torch.tensor(costs)
        min_idx = int(torch.argmin(costs).numpy())
        x_new_enc = ces[min_idx]
            
    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)