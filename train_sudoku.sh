#!/bin/bash

#symbolic Sudoku
python3 Main_PLL.py --filename Sudoku --seed 1 --train_size 100

# Constraint hardening (to be run after symbolic Sudoku training)
python3 Constraints_hardening.py --filename Sudoku

#Symbolic Sudoku, many-solution dataset
python3 Main_PLL.py --train_size 1000 --One_of_Many True --path_to_data Data_raw/one_of_many/ --filename 1oM --seed 1



