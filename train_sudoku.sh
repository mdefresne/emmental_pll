#!/bin/bash

echo
echo "Symbolic Sudoku, training on 100 RRN dataset grids"
echo
uv run Main_PLL.py --filename Sudoku --seed 3 --train_size 100

echo
echo "Constraint hardening (to be run after symbolic Sudoku training)"
echo
uv run Constraints_hardening.py --filename Sudoku

echo
echo "Symbolic Sudoku, training on the many-solution dataset"
echo
uv run Main_PLL.py --train_size 1000 --One_of_Many True --path_to_data Data_raw/one_of_many/ --filename 1oM

echo
echo "Symbolic Sudoku, training on a 10 grids (augmented to 100 times) dataset of the Hierarchical reasoning model Extreme Sudoku dataset"
echo "Tested on 1000 grids, be patient"
echo
uv run Main_PLL.py --train_size 10 --HRM 99 --test_size 1000 --path_to_data Data_raw/sudoku-extreme-10-aug-99/ 
