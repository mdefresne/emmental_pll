from os import getenv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

def guess_device():
    if (getenv("TORCH_DEVICE")):
       print("Using",getenv("TORCH_DEVICE"),"from TORCH_DEVICE")
       return getenv("TORCH_DEVICE")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS enabled")
        return "mps"
    if torch.cuda.is_available():
        print("CUDA enabled")
        return "cuda:0"
    else:
        print("Using CPU")
        return "cpu"


def list_from_dataframe(df):
    return df.apply(lambda X: [int(x) for x in X])


def get_loader(df, batch_size=1):
    """
    Return a dataloader objet from a pandas Dataframe with column named
    'x' (for the data) and 'y' (for the target)
    """

    X = list_from_dataframe(df["x"].reset_index(drop=True))
    Y = list_from_dataframe(df["y"].reset_index(drop=True))
    dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))

    return DataLoader(dataset, batch_size)


def to_array(sudoku_in_string):
    return np.array([int(x) for x in sudoku_in_string])


# Inspired from https://github.com/martius-lab/blackbox-backprop
def maybe_parallelize(function, arg_list):
    """
    Parallelizes execution is ray is enabled
    :param function: callable
    :param arg_list: list of function arguments (one for each execution)
    :return:
    """
    # if 'ray' in sys.modules and ray.is_initialized():
    # results = ray.get([function.remote(*arg) for arg in arg_list])
    # return(results)

    results = [function(*arg) for arg in arg_list]
    return results


def get_batch_input(data):
    """
    To shape the raw data into normalized batch input (bs, 81, 2)
    """
    nb_var = data.shape[1]
    grid_size = int(nb_var**0.5)
    r = torch.linspace(0.0, 1.0, steps=grid_size)
    f = torch.cartesian_prod(r, r).unsqueeze(0)

    return f.expand(data.shape[0], nb_var, f.shape[-1])


def Hamming(y_true, y_hat):
    """Input: lists"""

    Hamming = []
    for i in range(len(y_hat)):
        Hamming.append(y_hat[i] != y_true[i])

    return np.array(Hamming)


def plot_hist(W, NN_input, return_list=False):
    """Plot histograms showing the minimum of the diagonal for each matrix with constraint,
    and the maximum of all matrix without constraint.
    Input: predicted matrix W
           boolean indicating whether to return the list of values
    Output: list of values used to plot the histograms"""

    R, nR, Rndiag = [], [], []
    nb_var = NN_input.shape[1]  # shape bs, grid_size**2, ft_size
    grid_size = int(nb_var**0.5)
    num_square = int(grid_size**0.5)

    for i in range(nb_var):
        line_i, col_i = i // 9, i % 9
        sq_i = num_square * (line_i // num_square) + col_i // num_square
        for j in range(i + 1, nb_var):
            line_j, col_j = j // 9, j % 9
            sq_j = num_square * (line_j // num_square) + col_j // num_square

            # If constraint between cell i and cell j
            if (line_i == line_j) or (col_i == col_j) or (sq_i == sq_j):
                # min of the diagonal of the constraint table
                diag = torch.diagonal(W[0, i, j].view(grid_size, grid_size), 0)
                m = torch.min(diag).item()

                # min out of diagonal
                R.append(m)
                M = torch.max(
                    W[0, i, j].view(grid_size, grid_size)
                    - torch.diag(diag)
                    - 100 * torch.eye(grid_size)
                ).item()
                Rndiag.append(M)

            else:
                # max of the constraint table
                nR.append(torch.max(W[0, i, j]).item())

    plt.figure(figsize=(16, 4))
    plt.subplot(131)
    plt.hist(R)
    plt.title("Diagonal minimun if constraint")
    plt.xlabel("Learned cost")
    plt.subplot(132)
    plt.hist(Rndiag)
    plt.title("Martix maximum out of diagonal if constraint")
    plt.xlabel("Learned cost")
    plt.subplot(133)
    plt.hist(nR)
    plt.title("Martix maximum if no constraint")
    plt.xlabel("Learned cost")

    if return_list:
        return (R, nR, Rndiag)
