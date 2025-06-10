import torch
import torch.nn.functional as F
from random import sample


def PLL_all(W, y_true, nb_neigh=0, T=1, hints_logit=None):
    """
    Compute the total PLL loss over all variables and all batch samples

    Input: the predicted cost tensor W
           the true sequence y_true (tensor)
           the number of neighbours to mask (Gangster PLL parameter), int
           unary costs hints_logit (tensor, optional)

    Output: the PLL loss
    """

    nb_var = W.shape[1]
    grid_size = W.shape[3]
    bs = W.shape[0]

    y_indices = (y_true - 1).unsqueeze(-1).expand(bs, nb_var, grid_size).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, grid_size, grid_size)
    L_cost = Wr[
        torch.arange(bs)[:, None, None, None],
        torch.arange(nb_var)[None, :, None, None],
        torch.arange(nb_var)[None, None, :, None],
        torch.arange(grid_size)[None, None, None, :],
        y_indices,
    ]

    if nb_neigh > 0:  # number of neighbours to ignore
        samp = [sample([i for i in range(nb_var)], nb_neigh) for j in range(nb_var)]
        samp = torch.tensor(samp).reshape(nb_var, -1)

        neigh = torch.ones_like(L_cost)
        neigh[:, torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh

    costs_per_value = torch.sum(L_cost, dim=2)
    _, pred = torch.min(costs_per_value, -1)

    if hints_logit is not None:
        costs_per_value -= hints_logit
    lsm = F.log_softmax(-costs_per_value / T, dim=2)
    val_lsm = lsm[torch.arange(bs)[:, None], torch.arange(nb_var)[None, :], y_true - 1]

    return torch.sum(val_lsm)


def pred_all_var(W, y):
    """
     Predict the value for each variables given all other variables.
     Input: the cost matrix W
            the true sequence y_true
    Output: the predicted values and their associated cost
    """

    nb_var = W.shape[1]
    grid_size = int(nb_var**0.5)
    bs = W.shape[0]

    y_indices = (y - 1).unsqueeze(-1).expand(bs, nb_var, grid_size).unsqueeze(1)
    Wr = W.reshape(bs, nb_var, nb_var, grid_size, grid_size)
    L_cost = Wr[
        torch.arange(bs)[:, None, None, None],
        torch.arange(nb_var)[None, :, None, None],
        torch.arange(nb_var)[None, None, :, None],
        torch.arange(grid_size)[None, None, None, :],
        y_indices,
    ]

    costs_per_value = torch.sum(L_cost, dim=2)
    _, idx = torch.min(costs_per_value, dim=2)

    # PLL
    lsm = F.log_softmax(-costs_per_value, dim=2)
    PLLs = lsm[torch.arange(bs)[:, None], torch.arange(nb_var)[None, :], y - 1]
    return torch.stack([(idx + 1), PLLs])


def val_metrics(W, targets):
    """return the number of correctly predicted values and the value of -PLL"""

    best_acc = 0
    for target in targets:
        y_pred, PLL = pred_all_var(W, target)
        target_acc = torch.sum((target - y_pred) == 0)
        target_nPLL = -torch.sum(PLL)

        if target_acc > best_acc:
            best_acc = target_acc
            nPLL = target_nPLL

    return (best_acc, nPLL)
