# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision

from Scripts import *


def fill_missing_hints(W, target, hints, top=999999, resolution=2, backtrack=500000):
    bs = W.shape[0]
    cost_fn_size = W.shape[1]
    grid_size = int(cost_fn_size**0.5)
    thresh = nn.Threshold(10 ** (-resolution), 0)  # MUCH faster but kills all negative W

    hints = hints[0].detach().cpu().numpy()
    target = target[0].detach().cpu().numpy()
    W = thresh(W)
    W = W.view(bs, cost_fn_size, cost_fn_size, -1).detach().cpu().numpy()
    bad = 0
    solutions = []
    for b in range(bs):
        Problem = make_CFN(W[b], top=top, resolution=resolution)

        for i in range(cost_fn_size):
            costs = hints[i].copy()
            if target[i] > 0:
                extra_costs = 2 * top * np.ones(grid_size)
                extra_costs[target[i] - 1] = 0
                costs += extra_costs
            Problem.AddFunction([i], costs)

        try:
            sol, _, _ = Problem.Solve()
            ok = Sudoku(sol).check_sudoku()
            if not ok: 
                #print(".",end="")
                bad += 1
        except Exception:
            print("No solution found")
            sol = np.random.randint(0, 10, len(target))
            sol[target != 0] = target[target != 0]
        solutions.append(sol)

    return (np.array(solutions),bad)


def load_hard_set(test_set):
    hard_test_set = pd.DataFrame(columns=["x", "y"])
    for x in test_set["x"]:
        if np.sum(np.array([int(c) for c in x]) != 0) == 17:
            hard_test_set = pd.concat((hard_test_set, test_set[test_set["x"] == x]))

    return get_loader(hard_test_set, batch_size=1)


def get_loader(df, batch_size=1, n_hints=None, max_hints=None, size=None):
    """
    Return a dataloader objet from a pandas Dataframe with column named
    'x' (for the data) and 'y' (for the target)
    """

    X = list_from_dataframe(df["x"].reset_index(drop=True))
    Y = list_from_dataframe(df["y"].reset_index(drop=True))
    if (n_hints is None) and (max_hints is None):
        dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))
    else:
        L_x, L_y = [], []
        for i, x in enumerate(X):
            x = np.array(x)
            n = np.sum(x != 0)
            if n_hints is not None:
                cond = n == n_hints
            if max_hints is not None:
                cond = n <= max_hints
            if size is not None:
                cond = cond and (len(L_x) < size)
            if cond:
                L_x.append(torch.LongTensor(x))
                L_y.append(torch.LongTensor(np.array(Y[i])))
        X = torch.stack(L_x)
        Y = torch.stack(L_y)
        dataset = TensorDataset(X, Y)

    return DataLoader(dataset, batch_size)


# Loading data: 3x3 sudoku
path_to_data = "Data_raw/sudoku-hard/"
train_set = pd.read_csv(path_to_data + "train.csv", names=["x", "y"])
valid_set = pd.read_csv(path_to_data + "valid.csv", names=["x", "y"])
test_set = pd.read_csv(path_to_data + "test.csv", names=["x", "y"])

train_set = train_set[:9000]
sudoku_train_loader = get_loader(train_set, batch_size=1)
sudoku_test_loader = get_loader(test_set[:100], batch_size=1)
sudoku_val_loader = get_loader(valid_set[:64], batch_size=1)

hard_train_loader = load_hard_set(train_set[:50000])
hard_test_loader = load_hard_set(test_set)

### MNIST
mnist_train_set = torchvision.datasets.MNIST(
    "Data_raw",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

mnist_test_set = torchvision.datasets.MNIST(
    "Data_raw",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
mnist_test_loader = torch.utils.data.DataLoader(
    mnist_test_set, batch_size=1, shuffle=True
)

### Table of the indices of each digit by type
img_table = -np.ones((10, 10000, 1))
for idx, (img, label) in enumerate(mnist_train_set):
    img_table[label, np.argmin(img_table[label] != -1)] = idx
# img_table[label, idx] contains the idx-th data with label label

label_len = np.argmin(img_table != -1, axis=1).flatten()  # number of ex of each digit

#### Seed for reproducibility ####
random.seed(arg.seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed) 
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

### Training
device = torch.device(guess_device())

grid_size = 9
nb_var = grid_size**2
hidden_size = 64 # 128 
nblocks = 2  # 5
grounding = 2  # with permute
model = Visual_Net(
    grid_size, hidden_size=hidden_size, nblocks=nblocks, grounding=grounding
)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.to(device)
permutation = model.permute if grounding == 2 else None

### Model init ###
lr = 0.001
weight_decay = 0 #1e-8 
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
reg_term = 1 / 10000  # * 2
nb_neigh = 10  # for e-PLL (k)

L_mnist, L_solved = [], []
if True:
    L_mnist.append(
        test_mnist(model.LeNet, mnist_test_loader, device, permutation, topk=3)
    )
    L_solved.append(
        test_visual(
            model,
            sudoku_val_loader,
            img_table,
            label_len,
            mnist_train_set,
            device,
            resolution=1,
            quick=True,
            grounding=grounding,
        )[0]
    )

### Training ###
N_hints = [17, 20, 25, 30, 35] + [80] * 100
for epoch in range(20):
    bad_imput = 0
    model.train()
    print("Epoch ", epoch)
    if epoch == 6 or epoch == 8:
        optimizer.param_groups[0]["lr"] = lr / 10
    train_loader = get_loader(train_set, batch_size=1, max_hints=N_hints[epoch])
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        W, hints_logit = model(
            data, batch_idx, device, img_table, label_len, mnist_train_set
        )

        if model.grounding == 1:  # hide hints
            target[torch.where(data != 0)] = grid_size + 1
            hints_logit = None
        elif model.grounding > 1:
            target[torch.where(data != 0)] = 0
            if grounding == 2:
                hints_logit = model.permute(hints_logit)
            (target,bad) = fill_missing_hints(W, target, hints_logit, resolution=1)
            target = torch.tensor(target)
            bad_imput += bad

        y_true = target.type(torch.LongTensor).to(device)
        PLL = -PLL_all(W, y_true, nb_neigh=nb_neigh, hints_logit=hints_logit)
        L1 = torch.linalg.vector_norm(W, ord=1)

        loss = PLL + reg_term * L1
        loss.backward()
        optimizer.step()
    #print("Bad imputations:",bad_imput)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "Results/tb2/" + "grounding2" + str(epoch),
    )

    L_mnist.append(
        test_mnist(model.LeNet, mnist_test_loader, device, permutation, topk=3)
    )
    L_solved.append(
        test_visual(
            model,
            sudoku_val_loader,
            img_table,
            label_len,
            mnist_train_set,
            device,
            resolution=1,
            quick=True,
            grounding=grounding,
        )[0]
    )

### Test ###
### Table of the indices of each digit by type
img_table_test = -np.ones((10, 1500, 1))
for idx, (img, label) in enumerate(mnist_test_set):
    img_table_test[label, np.argmin(img_table_test[label] != -1)] = idx
label_len_test = np.argmin(img_table_test != -1, axis=1).flatten()

print("Test on 100 grids of each difficulty")
L_correct, L_corrected = [], []
for n_hints in range(34, 17 - 1, -1):
    test_loader = get_loader(test_set, batch_size=1, n_hints=n_hints, size=100)
    correct, corrected = test_visual(
        model,
        test_loader,
        img_table_test,
        label_len_test,
        mnist_test_set,
        device,
        resolution=1,
        quick=False,
        grounding=grounding,
    )
    L_correct.append(correct)
    L_corrected.append(corrected)
print("Average test accuracy: ", np.mean(L_correct) * 100, 'of which corrected ', np.mean(L_corrected)*100)
