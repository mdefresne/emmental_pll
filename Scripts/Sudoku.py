# -*- coding: utf-8 -*-
"""
Sudoku Class
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


class Sudoku:
    def __init__(self, sudoku_in_line):
        "Sudoku in line is a 1D tensor containing the cells"

        try:
            self.sudoku_in_line = torch.LongTensor(sudoku_in_line)
        except Exception:
            self.sudoku_in_line = sudoku_in_line
        self.grid_size = int(len(sudoku_in_line) ** 0.5)
        self.grid = self.sudoku_in_line.view(self.grid_size, self.grid_size)
        # usually grid_size = 9

    def plot_sudoku(self):
        """
        To plot a sudoku.

        """

        fig = plt.figure(figsize=(6, 6), constrained_layout=False)
        box_size = int(self.grid_size**0.5)  # size of the boxes in the sudoku
        outer_grid = fig.add_gridspec(box_size, box_size, wspace=0, hspace=0)
        anno_opts = dict(
            xy=(0.5, 0.5), xycoords="axes fraction", va="center", ha="center"
        )  # from plt doc

        for a in range(box_size):
            for b in range(box_size):
                ax = fig.add_subplot(outer_grid[a, b])
                # removing axis:
                ax.set_xticks([])
                ax.set_yticks([])

                # gridspec inside gridspec
                inner_grid = outer_grid[a, b].subgridspec(
                    box_size, box_size, wspace=0, hspace=0
                )
                for a2 in range(box_size):
                    for b2 in range(box_size):
                        ax2 = fig.add_subplot(inner_grid[a2, b2])
                        # figure in the cell
                        cell = int(self.grid[box_size * a + a2, box_size * b + b2])
                        ax2.annotate(str(cell), **anno_opts)
                        ax2.set_xticks([])
                        ax2.set_yticks([])
        for ax in fig.get_axes():
            ax.spines["top"].set_visible(ax.is_first_row())
            ax.spines["bottom"].set_visible(ax.is_last_row())
            ax.spines["left"].set_visible(ax.is_first_col())
            ax.spines["right"].set_visible(ax.is_last_col())

        plt.show()

    def get_box(self, num_box):
        """
        Return the wanted box of the sudoku
        Input: the number of the box (int between 1 and 9), count line by line
        Output: the box as a numpy array
        """

        reference = list(np.arange(1, 10))
        if num_box not in reference:
            print("The box id is incorrect.")

            return []

        else:
            box_size = int(self.grid_size**0.5)
            num_line = ((num_box - 1) // box_size) * box_size
            num_col = ((num_box - 1) % box_size) * box_size

            return self.grid[
                num_line : num_line + box_size, num_col : num_col + box_size
            ]

    def check_sudoku(self):
        reference = list(np.arange(1, 10))
        correct = True

        for i in range(self.grid_size):
            # checking lines
            if sorted(self.grid[i]) != reference:
                correct = False
            # checking columns
            if sorted(self.grid[:, i]) != reference:
                correct = False
            # checking boxes
            box = self.get_box(i + 1)
            if sorted(box.flatten()) != reference:
                correct = False

        return correct

    def get_NN_input(self):
        """
        Returns the input used by the neural network [row_j, col_j]
        """

        r = torch.linspace(0.0, 1.0, steps=self.grid_size)
        return torch.cartesian_prod(r, r)

    def edges(self):
        """
        Returns a matrix indicating the edges (between cell in same rom, column or box)
        """
        grid_size = self.grid_size
        M_edge = np.zeros((grid_size**2, grid_size**2))
        # same lign: indice// 9 is the same
        # same comlumn: indice%9 is the same
        for i in range(grid_size**2):
            num_lign_i = i // grid_size
            num_col_i = i % grid_size
            for j in range(grid_size**2):
                num_lign_j = j // grid_size
                num_col_j = j % grid_size
                if num_lign_i == num_lign_j:  # same line
                    M_edge[i, j] = 1
                if num_col_i == num_col_j:  # same column
                    M_edge[i, j] = 1
                # same square
                if (
                    int(num_lign_i // grid_size**0.5)
                    == int(num_lign_j // grid_size**0.5)
                ) & (
                    int(num_col_i // grid_size**0.5) == int(num_col_j // grid_size**0.5)
                ):
                    M_edge[i, j] = 1

        return M_edge
