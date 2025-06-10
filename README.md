# Scalable Coupling of Deep Learning with Logic Reasoning

The paper, together with the supplementary information, is available here : https://arxiv.org/abs/2305.07617

## Learning how to solve Sudoku.

This requires a Linux/MacOS machine (x86 architecture) with Python3, PyTorch (torch and torchvision) and pytoulbar2 installed, roughly 1GB of free disk space, ideally a CUDA GPU with few GBs of VRAM (available as cuda:0 in PyTorch) and few GBs of VRAM. If the code is instead executed on CPU (much slower), 32GB of RAM at least will be required.

We assume that `wget` and `unzip` are available on the system. If not, ask your system engineer for installation.

If needed install PyTorch and pytoulbar2 with

```
pip3 install torch
pip3 install torchvision
pip3 install pytoulbar2
pip3 install pandas
pip3 install matplotlib
```

## Folders

There are 4 folders: `Data_raw`, `Results.done`, `Results`, `Scripts`.

* `Data_raw` is used to store/dowload data sets
* `Results.done` contains traces of executions performed for the paper
* `Results` will contain traces of your executions
* `Scripts` contains all Python and shell scripts

## Install data sets

Data sets will be downloaded to the `Data_raw` folder. For reference:

* the RRN Sudoku data set will be downloaded from [this link](https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip) (14MB, 34MB uncompressed)  
* the One of many solution data set from [this page](https://sites.google.com/view/yatinnandwani/1oml) (47MB, all sudoku_* files)
* the SATNet Sudoku dataset from [this link](https://powei.tw/sudoku.zip) (142MB, 662MB uncompressed)

To download all data, type:

```
cd emmental-pll/Data_raw
bash ../download_data.sh
```

This can take some time, depending on your network connection, and it will eat up to 900MB of your disk space.

## Symbolic Sudoku

Training and testing is done by `Main_PLL.py`. Training and validation accuracies, as well as test results are written in `Results/` folder.

Options include:

* `--k` for the number of holes of the E-PLL
* `--train_size` to set the number of training grids
* `--One_of_Many` (Boolean) to train on the many-solution data set
* `--seed` to fix the initialization of the neural net weights
* `--epoch_max` to set the maximum number of training epochs
* `--path_to_data` to indicate the relative path to the data
* `--filename` to change the name of the file where results are written.

The script `train_sudoku.sh` gives the options used in the paper experiment (seeds were changed from 0 to 9).

```
bash train_sudoku.sh
```

## Visual Sudoku

Training on the visual Sudoku data set (9,000 grids training set) and testing on hard Sudoku (100 grids of each difficulty):

```
python3 Visual_sudoku_grounding.py
```

Since all parameters are fixed based on the experiments on the symbolic Sudoku.

## MinCut and MaxCut
To solve MaxCut or MinCut and plot the regret curve in `Results/`:

```
python3 Main_Cut.py --problem maxcut
```
Use option `--problem mincut` to switch to MinCut.

## Futoshki

```
python3 Main_Futoshiki.py --save_path "Results/model_futoshiki.pk" --game_type Futoshiki --k 10
```

