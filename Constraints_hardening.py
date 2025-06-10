# -*- coding: utf-8 -*-

import pandas as pd
import torch
import argparse

from Scripts import *


def count_cst(W, top):
    """Return the number of correct and incorrect constraints for Sudoku."""

    nb_var = W.shape[1]  # shape bs, grid_size**2, ft_size
    grid_size = W.shape[-1]
    num_square = int(grid_size**0.5)

    n_learnt_cst, n_incorrect_cst = 0, 0
    for i in range(nb_var):
        line_i, col_i = i // 9, i % 9
        sq_i = num_square * (line_i // num_square) + col_i // num_square
        for j in range(i + 1, nb_var):
            line_j, col_j = j // 9, j % 9
            sq_j = num_square * (line_j // num_square) + col_j // num_square

            # If constraint between cell i and cell j
            if (line_i == line_j) or (col_i == col_j) or (sq_i == sq_j):
                # min of the diagonal of the constraint table
                diag = torch.diagonal(W[i, j].view(grid_size, grid_size), 0)
                m = torch.min(diag).item()
                if m == top:
                    n_learnt_cst += 1

                # min out of diagonal
                M = torch.max(
                    W[i, j].view(grid_size, grid_size)
                    - torch.diag(diag)
                    - 100 * torch.eye(grid_size)
                ).item()
                if M > 0:
                    n_incorrect_cst += 1

            else:
                # max of the constraint table
                M = torch.max(W[i, j]).item()
                if M > 0:
                    n_incorrect_cst += 1

    return n_learnt_cst, n_incorrect_cst


def main():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--path_to_data",
        type=str,
        default="Data_raw/sudoku-hard/",
        help="path for loading training data",
    )
    argparser.add_argument(
        "--hidden_size", type=int, default=64, help="width of hidden layers"
    )
    argparser.add_argument(
        "--nblocks", type=int, default=2, help="number of blocks of 2 layers in ResNet"
    )
    argparser.add_argument(
        "--train_size", type=int, default=100, help="number of training samples"
    )
    argparser.add_argument(
        "--filename", type=str, default="PLL", help="filename were results are saved."
    )

    args = argparser.parse_args()

    device = torch.device(guess_device())
    grid_size = 9
    model = Net(grid_size=grid_size, hidden_size=args.hidden_size, nblocks=args.nblocks).to(device)

    checkpoint = torch.load("Results/tb2/" + args.filename, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_set = pd.read_csv(args.path_to_data + "train.csv", names=["x", "y"])
    train_loader = get_loader(train_set[: args.train_size], batch_size=1)

    ### Cost function hardening ###
    grid = get_batch_input(torch.ones(1, grid_size**2)).to(device)
    W = model(grid, device).squeeze()
    _, sorted_idx = torch.sort(W.flatten(), descending=True)
    top = 9999999
    logic_W = torch.zeros_like(W).to(device)

    to_harden = True
    i = 0
    while to_harden and i < len(sorted_idx):
        current_idx = sorted_idx[i]
        idx_i, idx_j, val_i, val_j = torch.unravel_index(current_idx, W.shape)

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.squeeze().to(device)
            if (target[idx_i] == val_i) and (target[idx_j] == val_j):
                to_harden = False
                break

        if to_harden:
            logic_W[idx_i, idx_j, val_i, val_j] = top
        i += 1

    ### Verifying the learnt constraints
    n_learnt_cst, n_incorrect_cst = count_cst(logic_W.detach().cpu(), top)
    print(
        f"{n_learnt_cst} constraints learnt ({n_learnt_cst / 810 * 100}%) and {n_incorrect_cst} incorrect constraints."
    )


if __name__ == "__main__":
    main()
