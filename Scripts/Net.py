import torch
import torch.nn as nn
import torch.nn.functional as F

from Scripts.utils import get_batch_input


def weights_init(m):
    """
    For initializing weights of linear layers (bias are put to 0).
    """

    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    """
    Residual block of 2 hidden layer for resMLP
    Init: size of the input (the output layer as the same dimension for the sum)
          size of the 1st hidden layer
    """

    def __init__(self, input_size, hidden_size):
        super(ResBlock, self).__init__()

        self.MLP = nn.Sequential(
            nn.BatchNorm1d(num_features=input_size),
            nn.ReLU(),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        x_out = self.MLP(x)
        x = x_out + x

        return x


class ResMLP(nn.Module):
    """
    ResMLP with nblocks residual blocks of 2 hidden layers.
    Init: size of the output output size (int)
          size of the input (int)
          size of the hidden layers
    """

    def __init__(self, output_size, input_size, hidden_size, nblocks=2):
        super(ResMLP, self).__init__()

        self.ResNet = torch.nn.Sequential()
        self.ResNet.add_module("In_layer", nn.Linear(input_size, hidden_size))
        self.ResNet.add_module("relu_1", torch.nn.ReLU())
        for k in range(nblocks):
            self.ResNet.add_module(
                "ResBlock" + str(k), ResBlock(hidden_size, hidden_size)
            )
        self.ResNet.add_module("Final_BN", nn.BatchNorm1d(num_features=hidden_size))
        self.ResNet.add_module("relu_n", torch.nn.ReLU())
        self.ResNet.add_module("Out_layer", nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.ResNet(x)

        return x


class Net(nn.Module):
    """
    Network composed of embedding + MLP
    Init: grid_size (int)
          hiddensize (int): number of neurons in hidden layer (suggestion: 128 or 256)
          resNet (bool). If False (default), use a regular MLP. Else, use a ResMLP
          nblocks (int): number of residual blocks. Default is 2
    """

    def __init__(self, grid_size, hidden_size=128, nblocks=2, feature_size=2):
        super(Net, self).__init__()

        self.feature_size = feature_size
        embedding_dim = 1
        input_size = 2 * embedding_dim * self.feature_size
        self.grid_size = grid_size

        self.MLP = ResMLP(self.grid_size**2, input_size, hidden_size, nblocks)
        self.MLP.apply(weights_init)

    def forward(self, x, device):
        bs, nb_var, nb_ft = x.shape

        t = torch.triu_indices(nb_var, nb_var, 1)
        r = x[:, t]
        rr = torch.swapaxes(r, 1, 2).reshape(-1, 2 * nb_ft)
        pred = self.MLP(rr)

        pred = pred.reshape(bs, -1, self.grid_size, self.grid_size)
        r, c = t
        out = torch.zeros(
            bs, nb_var, nb_var, self.grid_size, self.grid_size, device=device
        )
        out[:, r, c] = pred
        pred = torch.swapaxes(pred, 2, 3)
        out[:, c, r] = pred

        return out


class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        if x.shape[-1] == 32:
            x = x[:, :, 2:-2, 2:-2]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 81, 10
        # return F.softmax(x, dim=1)[:,:9].contiguous()
        return x[:, :9].contiguous()


class Visual_Net(nn.Module):
    def __init__(self, grid_size, hidden_size, nblocks, grounding=0):
        """Grounding: either none (0), version 1 or version 2"""
        super(Visual_Net, self).__init__()

        self.n_class = grid_size
        # self.LeNet = LeNet5(self.n_class)
        self.LeNet = DigitConv()
        self.grounding = grounding
        if self.grounding == 1:
            self.ResNet = Net(
                grid_size + 1,
                hidden_size=hidden_size,
                nblocks=nblocks,
                feature_size=grid_size + 2,
            )
        else:
            self.ResNet = Net(
                grid_size, hidden_size=hidden_size, nblocks=nblocks, feature_size=2
            )
        if grounding == 2:
            self.permute = nn.Linear(grid_size, grid_size, bias=False)
            nn.init.normal_(self.permute.weight, mean=1, std=0)

    def forward(self, data, batch_idx, device, img_table, label_len, mnist_set):
        # recognition of hints
        sample = data[0]
        sample_idx = batch_idx
        hints_idx, X = [], []
        for idx, c in enumerate(sample):
            c = int(c.item())
            if c != 0:
                img_idx = hash(str(c) + str(idx) + str(sample_idx)) % label_len[c]
                # img_idx = int(hashlib.md5(
                #   (str(c)+str(idx)+str(sample_idx)
                #  ).encode()).hexdigest(), 16) % label_len[c]
                x, y = mnist_set[int(img_table[c, img_idx].squeeze())]  # y=c
                hints_idx.append(idx)
                X.append(x)

        bs = data.shape[0]
        nb_var = data.shape[1]
        hints_pba = torch.zeros((bs, nb_var, self.n_class), device=device)
        X = torch.stack(X).to(device)
        hints_pba[:, hints_idx] = self.LeNet(X)
        # contains the cost predicted by LeNet for cells with a hint, otherwise 0

        ### rule prediction
        NN_input = get_batch_input(data).to(device)
        if self.grounding == 1:  # give LeNet output as input to ResNet
            NN_input = torch.cat((NN_input, hints_pba), dim=2)

        W = self.ResNet(NN_input, device)

        return W, hints_pba
