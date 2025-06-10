# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader

from Scripts import *

def source_node_fixed(crd):
    """Identify source node and end node"""

    n_var = len(crd)
    d = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(n_var):
            d[i, j] = np.linalg.norm(crd[i] - crd[j], 2)

    source, well = np.unravel_index(np.argmax(d, axis=None), d.shape)
    return int(source), int(well)


def source_node(graph, i):
    """Identify source node and end node"""

    crd = np.array(graph["vertices"])
    n_var = len(crd)
    source = i % n_var
    well = np.argmax([np.linalg.norm(crd[source] - crd[j], 2) for j in range(n_var)])

    return source, well


def solve_cut(W, source, well, n_domain=2, resolution=2, top=9999999):
    # enforce source_node=0, end_node = 1 with unary
    n_var = W.shape[1]
    unary_cost = torch.zeros((n_var, 2))
    unary_cost[source] = torch.tensor(np.array([0, top]))
    unary_cost[well] = torch.tensor(np.array([top, 0]))

    Problem = make_CFN(
        W.reshape(n_var, n_var, -1).detach().cpu().numpy(),
        unary=unary_cost.detach().cpu().numpy(),
        resolution=resolution,
        top=top,
    )
    pred = Problem.Solve()

    return np.array(pred[0]), pred[1]


def solve_true_param(
    instance_capacities, graph, source, well, pb, n_domain=2, resolution=2, top=9999999
):
    n_var = len(graph["vertices"])
    W = torch.zeros((n_var, n_var, n_domain, n_domain))
    for i, edge in enumerate(graph["edges"]):
        node1, node2 = np.sort(edge)
        capacity = instance_capacities[i]
        if pb == "mincut":
            W[node1, node2] = capacity * (1 - torch.eye(n_domain))
        elif pb == "maxcut":
            W[node1, node2] = capacity * torch.eye(n_domain)

    return solve_cut(W, source, well, n_domain, resolution, top)


class GraphBridgeDataset(Dataset):
    """Dataset of graphs with bridge images at each edges."""

    def __init__(self, img_path, graph, pb, n_sample=100, augment=False):
        self.graph = graph
        self.n_sample = n_sample
        assert pb in ["mincut", "maxcut"]
        self.pb = pb

        self.bridge_img = []
        for name in os.listdir(img_path):
            img = read_image(img_path + name, ImageReadMode.GRAY)
            img = img.type(torch.Tensor)
            self.bridge_img.append(img)
        self.bridge_img = torch.stack(self.bridge_img)
        n_bridge = len(self.bridge_img)

        self.true_capacities = torch.tensor(np.array([1, 2, 5]))
        assert len(self.true_capacities) == n_bridge
        rng = np.random.default_rng(42)
        self.capacity_per_edge = rng.choice(
            np.arange(n_bridge), size=(self.n_sample, len(graph["edges"]))
        )

        ### Computing true solutions with solver
        self.all_targets = []
        self.source_well = []
        for idx in range(self.n_sample):
            source, well = source_node(self.graph, idx)
            self.source_well.append([source, well])
            # Edge to/from source and well have max capacity:
            for i, edge in enumerate(self.graph["edges"]):
                if source in edge or well in edge:
                    self.capacity_per_edge[idx, i] = torch.argmax(self.true_capacities)

            true_costs = self.true_capacities[self.capacity_per_edge[idx]]
            true_sol, cost = solve_true_param(
                true_costs, self.graph, source, well, self.pb
            )
            self.all_targets.append(true_sol)

            # Debiasing augmentation: create a second sample with all variables switched
            if augment:
                self.source_well.append([well, source])
                self.all_targets.append(1 - (np.array(true_sol) - 2))
        if augment:
            self.capacity_per_edge = np.repeat(self.capacity_per_edge, 2, axis=0)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        bridge_type = self.capacity_per_edge[idx]
        img = self.bridge_img[bridge_type]
        img /= 256

        true_costs = self.true_capacities[self.capacity_per_edge[idx]]
        target = self.all_targets[idx]

        return (img, self.source_well[idx]), (target, true_costs)


class DigitConv(nn.Module):
    """
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self, n_output=10):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, n_output)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BridgeNet(nn.Module):
    def __init__(self, n_output, pb="mincut", n_domain=2):
        super(BridgeNet, self).__init__()

        assert pb in ["mincut", "maxcut"]
        self.pb = pb
        self.CNN = DigitConv(n_output)
        self.n_domain = n_domain

    def forward(self, x, graph):
        n_var = len(graph["vertices"])
        W = torch.zeros((n_var, n_var, self.n_domain, self.n_domain),device=x.device)

        for i, edge in enumerate(graph["edges"]):
            node1, node2 = np.sort(edge)
            bridge_img = x[:, i]
            # cost_fn = self.CNN(bridge_img).reshape(bs, n_domain, n_domain)
            c = self.CNN(bridge_img)
            if self.pb == "mincut":
                cost_fn = c * (1 - torch.eye(self.n_domain,device=c.device))
            if self.pb == "maxcut":
                cost_fn = c * torch.eye(self.n_domain, device=c.device)
            W[node1, node2] = cost_fn

        return W.unsqueeze(0)


def compute_cost(sol, graph, true_costs):
    cost = 0
    for i, edge in enumerate(graph["edges"]):
        node1, node2 = np.sort(edge)
        if sol[node1] != sol[node2]:
            cost += true_costs.squeeze()[i]
    return cost.item()


def ind_regret(true_costs, true_sol, pred_sol, graph):
    true_costs = true_costs.squeeze()
    cost = compute_cost(true_sol, graph, true_costs)
    pred_cost = compute_cost(pred_sol, graph, true_costs)

    return abs(pred_cost - cost)


def regret(testloader, graph, model, device, resolution=2, top=9999999):
    reg = 0
    model.eval()
    with torch.no_grad():
        for _, (data, (target, true_costs)) in enumerate(testloader):
            img, (source, well) = data
            true_sol = target.squeeze().detach().cpu().numpy()
            W = model(img.to(device), graph)
            pred_sol, _ = solve_cut(
                W, source.item(), well.item(), resolution=resolution, top=top
            )
            reg += ind_regret(true_costs, true_sol, pred_sol, graph)

    return reg / len(testloader)


def main():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--problem", type=str, default="maxcut", help="mincut or maxcut"
    )
    args = argparser.parse_args()

    datafile = "Data_raw/truncated_icosahedron.json"
    with open(datafile, "r") as f:
        graph = json.load(f)

    pb = args.problem
    img_path = "Data_raw/images_ponts/grayscale/"
    dataset = GraphBridgeDataset(img_path, graph, pb, n_sample=50, augment=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    testset = GraphBridgeDataset(img_path, graph, pb, n_sample=50)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    n_domain = 2  # Boolean variables
    torch.manual_seed(0)
    top = 9999999
    resolution = 2
    np.random.seed(0)

    device = torch.device(guess_device())
    model = BridgeNet(n_output=1, pb=pb, n_domain=n_domain)
    model.to(device)

    lr = 0.001
    nb_neigh = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    L_reg = []
    for epoch in range(5):
        model.train()
        pll_epoch = 0
        for batch_idx, (data, (target, true_costs)) in enumerate(dataloader):
            optimizer.zero_grad()
            img, _ = data
            W = model(img.to(device), dataset.graph)
            y_true = target.type(torch.LongTensor).to(device)

            PLL = -PLL_all(W, y_true, nb_neigh=nb_neigh)
            PLL.backward()
            optimizer.step()
            pll_epoch += PLL.item()

            if batch_idx % 24 == 0 and batch_idx > 0:
                reg = regret(testloader, graph, model, device, resolution, top)
                print("Regret: ", reg)
                L_reg.append(reg)

    plt.plot(np.arange(1, len(L_reg) + 1), L_reg)
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.title(f"{pb} learning task")
    plt.savefig(f"Results/{pb}.pdf")


if __name__ == "__main__":
    main()
