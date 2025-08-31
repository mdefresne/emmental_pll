import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Scripts.utils import get_batch_input,maybe_parallelize
from Scripts.Solver import solver,make_CFN
from Scripts.PLL import val_metrics
from Scripts.Sudoku import Sudoku

def val_PLL(model, val_loader, optimizer, epoch, device, filename=""):
    acc_tot, PLL_tot, L_tot = 0.0, 0.0, 0
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        NN_input = get_batch_input(data).to(device)  # bs, nb_var, nb_feature
        W = model(NN_input, device)

        y_true = target.type(torch.LongTensor).to(device)

        acc, PLL = val_metrics(W, y_true)
        acc_tot += acc  # correct count
        L_tot += y_true.numel()
        PLL_tot += PLL.item()
        del W

    acc_tot = acc_tot / L_tot
    print(f"Epoch: {epoch}\t Accuracy: {acc_tot:.3f}\t negPLL: {PLL_tot:.1f}")

    file = open("Results/" + filename + ".txt", "a")
    file.write(
        "\n Epoch "
        + str(epoch)
        + "- Validation: "
        + str(PLL_tot)
        + ", "
        + str(acc_tot.item())
    )
    file.close()

    return (acc_tot, PLL_tot)


def solve(W, data, quick=True, resolution=1):
    """
    Solve the instance data based on the CFN W.
    """

    setUB0, margin = True, 0
    bs, grid_size = data.shape[0], int(data.shape[1] ** 0.5)
    L_input = [
        [
            W[b].detach().cpu().numpy(),
            data[b],
            None,
            False,
            grid_size,
            999999,
            resolution,
            margin,
            setUB0,
        ]
        for b in range(bs)
    ]

    L_sol = maybe_parallelize(solver, arg_list=L_input)

    if not quick:
        if len(L_sol[0]) < 3:  # no solution with cost 0 found -> run full tb2
            # not implemented for bs > 1
            setUB0 = False
            L_input = [
                [
                    W[b].detach().cpu().numpy(),
                    data[b],
                    None,
                    False,
                    grid_size,
                    999999,
                    resolution,
                    margin,
                    setUB0,
                ]
                for b in range(bs)
            ]
            L_sol = maybe_parallelize(solver, arg_list=L_input)

    return L_sol


def test(
    model, valid_data, device, quick=True, resolution=1, filename="", One_of_Many=False
):
    """
    Test if the training grids are properly filled.
    Inputs: - trained model model
            - test set test_loader
            - device
            - whether to to a quick test (quick = True) or a full test.
              If True, returns the % of correct grid for which tb2 finds a solution of cost 0
              (it is a lower bound of the true % of solved solution)
            - resolution of tb2
            - filename of the file where results are written

    Output: - the Hamming loss on the test set (% of correct boxes)
            - the % of grid properly filled
    """

    model.eval()
    test_loss, nb_solved = 0, 0
    thresh = nn.Threshold(10 ** (-resolution), 0)


    num_sample = len(valid_data) if One_of_Many else len(valid_data.dataset)

    with torch.no_grad():
        for full_data in tqdm(enumerate(valid_data),total=num_sample,smoothing=0):
            if One_of_Many:
                _, dico = full_data
                data = torch.Tensor(dico["query"]).reshape(1, -1).to(device)
                targets = dico["target_set"]

            else:
                batch_idx, (data, target) = full_data

            bs = data.shape[0]
            cost_fn_size = model.grid_size**2
            NN_input = get_batch_input(data).to(device)
            W = model(NN_input, device)
            W = W.view(bs, cost_fn_size, cost_fn_size, -1)

            L_sol = solve(thresh(W), data, quick=quick, resolution=resolution)

            if len(L_sol[0]) == 3:  # 1 solution found
                if One_of_Many:
                    pred = np.array(L_sol[0][0])
                    best_acc = 0
                    for target in targets:
                        target_acc = np.sum(pred == target) / cost_fn_size
                        if target_acc > best_acc:
                            best_acc = target_acc
                    test_loss += best_acc
                    if best_acc == 1:
                        nb_solved += 1

                else:
                    L_sudoku_solved = [
                        Sudoku(L_sol[b][0]).sudoku_in_line for b in range(bs)
                    ]

                    nb_correct_box = torch.sum(torch.stack(L_sudoku_solved) == target)
                    hamming = nb_correct_box / cost_fn_size
                    test_loss += hamming
                    if nb_correct_box == cost_fn_size:
                        nb_solved += 1

    if filename != "":
        file = open("Results/" + filename + ".txt", "a")
        file.write("\nTest accuracy " + str(test_loss / num_sample))
        file.write("% solved " + str(nb_solved / num_sample * 100) + "\n")
        file.close()
    print("Test accuracy", test_loss / num_sample)
    print("% of solved grids", nb_solved / num_sample * 100)
    return (test_loss / num_sample, nb_solved / num_sample * 100)


def test_1oM(model, many_data_test, device, resolution=1, filename=None):
    grid_size, nb_var = model.grid_size, model.grid_size**2
    L = []
    for dico in many_data_test[:256]:
        data = torch.Tensor(dico["query"]).reshape(1, -1).to(device)
        targets = dico["target_set"]

        NN_input = get_batch_input(data).to(device)
        W = model(NN_input, device)
        bs = W.shape[0]
        W = W.view(bs, nb_var, nb_var, -1)
        W = (W > 1) * 99999  # thresholding

        setUB = 10**-(resolution)
        all_solutions = True
        bs = data.shape[0]
        L_input = [
            [
                W[b].detach().cpu().numpy(),
                data[b],
                None,
                False,
                grid_size,
                999999,
                resolution,
                0,
                setUB,
                all_solutions,
            ]
            for b in range(bs)
        ]

        L_sol = maybe_parallelize(solver, arg_list=L_input)[0]

        correct_sol = 0
        for i in range(len(L_sol)):
            if np.any(np.all(np.array(L_sol[i][1]) == targets, axis=1)):
                correct_sol += 1
        L.append(correct_sol / len(targets))

    print(
        f"In average, {np.mean(L) * 100}% of the feasible solutions are predicted per test instance."
    )
    if filename:
        file = open("Results/" + filename + ".txt", "a")
        file.write(
            f"\nIn average, {np.mean(L) * 100}% of the feasible solutions are predicted per test instance."
        )
        file.close()
    return np.mean(L)


def test_mnist(network, test_loader, device, permutation=None, topk=1):
    network.eval()
    correct, nb_non0 = [0] * topk, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            if permutation is not None:
                output = permutation(output)
            output = nn.functional.softmax(output, dim=1)
            pred = output.data.sort(descending=True)[1] + 1
            # pred = output.data.max(1, keepdim=True)[1] + 1
            if target.item() != 0:
                for k in range(1, topk + 1):
                    correct[k - 1] += target in pred[:, :k]
                nb_non0 += 1

        print("MNIST test set Accuracy:")
        for k in range(1, topk + 1):
            print(
                "Top-{}: {}/{} ({:.1f}%)\n".format(
                    k, correct[k - 1], nb_non0, 100.0 * correct[k - 1] / nb_non0
                )
            )
    return 100.0 * correct[0] / nb_non0


def test_visual(
    model,
    sudoku_test_loader,
    img_table_test,
    label_len_test,
    mnist_test_set,
    device,
    resolution=1,
    quick=False,
    grounding=0,
):
    model.eval()
    grid_size = model.ResNet.grid_size - (grounding == 1)
    thresh = nn.Threshold(10 ** (-resolution), 0)
    backtrack = 1000000  # 00

    correct, corrected, accuracy = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(sudoku_test_loader):
            _, nb_var = data.shape[0], data.shape[1]
            cost_fn_size = grid_size**2
            W, hints_logit = model(
                data=data,
                batch_idx=batch_idx,
                device=device,
                img_table=img_table_test,
                label_len=label_len_test,
                mnist_set=mnist_test_set,
            )
            if grounding == 1:
                W = W[:, :, :, :grid_size, :grid_size]
            if grounding == 2:
                hints_logit = model.permute(hints_logit)
            W = thresh(W)
            W = W.view(cost_fn_size, cost_fn_size, -1).detach().cpu().numpy()

            y = target.squeeze().detach().cpu().numpy()
            d = data.squeeze().detach().cpu().numpy()
            hint = d != 0
            hints_logit = hints_logit.squeeze().detach().cpu().numpy()
            hints_logit -= np.max(hints_logit, axis=-1).reshape(-1, 1)

            # compute upper bound
            if not quick:
                oneH = torch.ones((nb_var, grid_size))
                oneH[torch.arange(nb_var), target[0].type(torch.LongTensor) - 1] = 0
                unary = oneH.squeeze().detach().cpu().numpy() * 9999 - hints_logit
                if grounding == 1:
                    unary = None
                Problem = make_CFN(
                    W, unary=unary, resolution=resolution, backtrack=backtrack
                )
                pred = Problem.Solve()
                UB = pred[1] + 10 ** (-resolution) * 10
            else:
                UB = 10 ** (-resolution) * 10

            if grounding == 1:
                unary = None
            Problem = make_CFN(
                W, unary=-hints_logit, resolution=resolution, backtrack=backtrack
            )
            Problem.SetUB(UB)
            pred = Problem.Solve()
            if pred is not None:
                pred = np.array(pred[0])

                acc_mnist = np.sum(
                    np.argmax(hints_logit, axis=1)[hint] + 1 == d[hint]
                ) / len(d[hint])
                acc = np.sum(y == pred) / len(y)
                accuracy += acc
                correct += acc == 1
                corrected += acc == 1 and (acc_mnist < 1)

    num_sample = len(sudoku_test_loader.dataset)
    print(
        f"Grid solved: {correct} ({100 * correct / num_sample}%), including {corrected} corrected grids."
    )

    return correct / num_sample, corrected / num_sample
