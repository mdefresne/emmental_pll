# -*- coding: utf-8 -*-
import pandas as pd
import torch
import argparse
import pickle
import numpy as np

from Scripts import *


def train_PLL(args):
    #### Loading data ####
    if args.One_of_Many:
        file = open(args.path_to_data + "sudoku_9_train_e.pkl", "rb")
        many_data = pickle.load(file)
        file = open(args.path_to_data + "sudoku_9_dev_e.pkl", "rb")
        many_data_valid = pickle.load(file)
        file.close()
        grid_size = int(len(many_data[0]["query"]) ** 0.5)
        # list of dict {query, target_set}. Query is 1 array, traget_set is a list of arrays

    else:
        train_set = pd.read_csv(args.path_to_data + "train.csv", names=["x", "y"])
        valid_set = pd.read_csv(args.path_to_data + "valid.csv", names=["x", "y"])
        grid_size = int(len(train_set["x"][0]) ** 0.5)

        # Creating the dataset
        train_loader = get_loader(
            train_set[: args.train_size*(args.HRM+1)], batch_size=args.batch_size
        )
        val_loader = get_loader(valid_set[: args.valid_size], batch_size=1)

    #### MODEL ####
    torch.manual_seed(args.seed)

    device = torch.device(guess_device())

    model = Net(grid_size, hidden_size=args.hidden_size, nblocks=args.nblocks)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    file = open("Results/" + args.filename + ".txt", "a")
    file.write(
        "\n Training with the following parameters:"
        + "\n nblock: "
        + str(args.nblocks)
        + "\n hidden size: "
        + str(args.hidden_size)
        + "\n lr: "
        + str(args.lr)
        + "\n weight decay: "
        + str(args.weight_decay)
        + "\n batch size: "
        + str(args.batch_size)
        + "\n L1: "
        + str(args.reg_term)
        + "\n HRM aug: "
        + str(args.HRM)
        + "\n Train size: "
        + str(args.train_size)
        + "\n Seed: "
        + str(args.seed)
        + "\n E-PLL parameter: "
        + str(args.k)
        + "\n"
    )
    file.close()

    #### TRAINING ####
    for epoch in range(1, args.epoch_max):
        PLL_epoch, loss_epoch = 0, 0
        model.train()

        training_data = (
            many_data[: args.train_size] if args.One_of_Many else train_loader
        )

        for full_data in enumerate(training_data):
            if args.One_of_Many:
                _, dico = full_data
                data = torch.Tensor(dico["query"]).reshape(1, -1).to(device)
                targets = dico["target_set"]
                # sample 5 of the solutions (thus, incomplete info)
                idx = np.random.randint(0, min(5, len(targets)))
                target = (
                    torch.Tensor(targets[idx])
                    .reshape(1, -1)
                    .type(torch.LongTensor)
                    .to(device)
                )

            else:
                batch_idx, (data, target) = full_data

            optimizer.zero_grad()
            NN_input = get_batch_input(data).to(device)  # bs, grid_size**2, nb_feature
            y_true = target.type(torch.LongTensor).to(device)

            W = model(NN_input, device)
            L1 = torch.linalg.vector_norm(W, ord=1)  # L1 penalty on predicted cost

            PLL = -PLL_all(W, y_true, nb_neigh=args.k)
            PLL_epoch += torch.sum(PLL)

            loss = PLL + args.reg_term * L1
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        print(f"Training loss: {loss_epoch:.1f}")

        ### Validation ###
        with torch.no_grad():
            acc_tot, PLL_tot, L_tot = 0.0, 0.0, 0
            model.eval()
            valid_data = (
                many_data_valid[: args.valid_size] if args.One_of_Many else val_loader
            )

            for full_data in enumerate(valid_data):
                if args.One_of_Many:
                    _, dico = full_data
                    data = torch.Tensor(dico["query"]).reshape(1, -1).to(device)
                    targets = dico["target_set"]
                    targets = torch.Tensor(np.array(targets)).type(torch.LongTensor)

                else:
                    batch_idx, (data, targets) = full_data

                NN_input = get_batch_input(data).to(
                    device
                )  # bs, grid_size**2, nb_feature
                W = model(NN_input, device)

                acc, PLL = val_metrics(W, targets.to(device))
                acc_tot += acc
                L_tot += y_true.numel()
                PLL_tot += PLL.item()

        acc_tot = acc_tot / L_tot
        print(f"Epoch: {epoch}\t Accuracy: {acc_tot:.3f}\t -PLL: {PLL_tot:.1f}")

        file = open("Results/" + args.filename + ".txt", "a")
        file.write(
            "\n Epoch "
            + str(epoch)
            + " - training loss: "
            + str(loss_epoch)
            + "\n Validation: "
            + str(acc_tot.item())
            + ", "
            + str(PLL_tot)
        )
        file.close()

        if epoch % int(2000 / (args.train_size * (1 + args.HRM))) == 0:
            # print every 2 epoch with 1000 grids, 10 epoch for 200 grids
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "Results/tb2/" + args.filename,
            )  # + "_" + str(epoch))

            test_acc = test(
                model,
                valid_data,
                device,
                quick=True,
                resolution=1,
                filename=args.filename,
                One_of_Many=args.One_of_Many,
            )
            if test_acc[0] == 1:  # training ends
                return model
    return model


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
    # with 1oM: Data_raw/one_of_many/
    argparser.add_argument(
        "--hidden_size", type=int, default=64, help="width of hidden layers"
    )
    argparser.add_argument(
        "--nblocks", type=int, default=2, help="number of blocks of 2 layers in ResNet"
    )
    argparser.add_argument(
        "--epoch_max", type=int, default=301, help="maximum number of epochs"
    )
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay"
    )
    argparser.add_argument(
        "--reg_term",
        type=float,
        default=1 / 10000,
        help="L1 regularization on costs",
    )
    argparser.add_argument("--k", type=int, default=10, help="E-PLL parameter")
    argparser.add_argument(
        "--batch_size", type=int, default=1, help="training batch size"
    )
    argparser.add_argument(
        "--train_size", type=int, default=1000, help="number of training samples"
    )
    argparser.add_argument(
        "--valid_size", type=int, default=32, help="number of validation samples"
    )
    argparser.add_argument(
        "--test_size", type=int, default=256, help="number of test samples"
    )
    argparser.add_argument("--seed", type=int, default=0, help="manual seed")
    argparser.add_argument(
        "--filename", type=str, default="PLL", help="filename to save results"
    )
    argparser.add_argument(
        "--One_of_Many",
        type=bool,
        default=False,
        help="Whether to train on instances with many solutions",
    )
    argparser.add_argument(
        "--HRM",
        type=int,
        default=0,
        help="AUgmentation ratio of HRM extreme instances",
    )



    args = argparser.parse_args()

    ### TRAINING ###
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model = train_PLL(args)
        end.record()

        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)  # time in milliseconds

        print(f'Total training time: {round(total_time/1000,2)} seconds')
        file = open("Results/" + args.filename + ".txt", "a")
        file.write("\n Total training time: " + str(round(total_time/1000,2))+" seconds")
        file.close()

    else:
        model = train_PLL(args)

    ### TEST ###
    print("Test in progress (can take several minutes)")
    device = torch.device(guess_device())

    if args.One_of_Many:
        file = open(args.path_to_data + "sudoku_9_test_e_big_amb.pkl", "rb")
        many_data_test = pickle.load(file)
        file.close()
        test_acc = test_1oM(model, many_data_test, device, filename=args.filename)

    else:
        test_set = pd.read_csv(args.path_to_data + "test.csv", names=["x", "y"])
        hard_test_set = pd.DataFrame(columns=["x", "y"])
        if args.HRM:
            hard_test_set = test_set.head(args.test_size)
        else:
            for x in test_set["x"]:
                if np.sum(np.array([int(c) for c in x]) != 0) == 17:
                    hard_test_set = pd.concat((hard_test_set, test_set[test_set["x"] == x]))
        
        hard_test_loader = get_loader(hard_test_set, batch_size=1)
        test_acc = test(
            model,
            hard_test_loader,
            device,
            quick=False,
            resolution=1,
            filename=args.filename,
            One_of_Many=args.One_of_Many,
        )
        print("Test accuracy on hard instances: ", test_acc)


if __name__ == "__main__":
    main()
