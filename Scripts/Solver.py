import numpy as np
import torch

from Scripts.CFN import CFN
from Scripts.Sudoku import Sudoku
from Scripts.utils import Hamming


def random_solver(W, sudoku_in_line):
    grid_size = int(W.shape[-1] ** 0.5)
    if np.random.randint(0, 20) == -1:
        c = np.random.randint(1, grid_size + 1)
        L = [c] * grid_size**2
        return [L]
    else:
        L = []
        for i in range(grid_size**2):
            if int(sudoku_in_line[i]) == 0:
                L.append(np.random.randint(1, grid_size + 1))
            else:
                L.append(int(sudoku_in_line[i]))
        return [L]


def make_CFN(W, unary=None, top=999999, resolution=1, backtrack=500000, allSolutions=0):
    Problem = CFN(
        top, resolution, vac=True, backtrack=backtrack, allSolutions=allSolutions
    )
    # Problem = tb2.CFN(top, resolution, vac=True)
    n_var = W.shape[0]
    n_domain = int(W.shape[-1] ** 0.5)

    # Create variables
    for i in range(n_var):
        Problem.AddVariable("x" + str(i), range(1, n_domain + 1))
    # costs
    for i in range(n_var):
        for j in range(i + 1, n_var):
            Problem.AddFunction([i, j], W[i, j])

    # unary costs
    if unary is not None:
        for i in range(n_var):
            if np.max(unary[i]) > 0:
                Problem.AddFunction([i], unary[i])  # *2*top)

    return Problem


def add_hints(problem, hints, solution, grid_size=9, top=999999, margin=1):
    for i in range(grid_size**2):
        sol = int(solution[i]) if solution is not None else 0
        hint = int(hints[i])
        if sol or hint:
            costs = np.zeros(grid_size)
            if sol:
                costs[sol - 1] = margin  # hyperparameter to tune

            if hint > 0:
                extra_costs = 2 * top * np.ones(grid_size)
                extra_costs[hint - 1] = margin
                costs += extra_costs
            problem.AddFunction([i], costs)


def solver(
    W,
    sudoku_in_line,
    solution=None,
    random=False,
    grid_size=9,
    top=999999,
    resolution=2,
    margin=1,
    setUB=False,
    all_solutions=False,
):
    """
    Solve the sudoku with constraints from matrix W and hints in sudoku_in_line
    If solution is given, the solution is penalized for the Hinge Loss
    """

    if random:
        print("random")
        return random_solver(W, sudoku_in_line)
    else:
        allSolutions = 1000 if all_solutions else 0
        Problem = make_CFN(W, top=top, resolution=resolution, allSolutions=allSolutions)
        add_hints(Problem, sudoku_in_line, solution, grid_size, top, margin)

        if setUB:
            Problem.SetUB(10 ** (-resolution))

        sol = Problem.Solve()
        if (sol is None) or sol[0][0] == 10:
            return random_solver(W, sudoku_in_line)

        if all_solutions:
            all_sol = Problem.GetSolutions()
            return all_sol
        else:
            return sol


def gradH(sol, pred, grid_size):
    """
    sol is the solution of the learned problem
    pred is the solution of the Loss augmented problem
    """
    nb_var = grid_size**2
    sol = sol - 1
    pred = pred - 1
    grad = np.zeros((nb_var, nb_var, grid_size, grid_size))
    i, j = np.triu_indices(nb_var, k=1)
    grad[i, j, sol[i], sol[j]] = 1.0
    grad[i, j, pred[i], pred[j]] -= 1.0
    grad[j, i, sol[j], sol[i]] = 1.0
    grad[j, i, pred[j], pred[i]] -= 1.0
    return grad.reshape(nb_var, nb_var, nb_var)


def through_solver(
    W,
    data,
    target,
    random=False,
    nb_var_to_pred="all",
    resolution=2,
    margin=1,
    setUB=False,
):
    y_true = Sudoku(target).sudoku_in_line.numpy()
    inp_sudoku = Sudoku(data).sudoku_in_line
    grid_size = int(W.shape[-1] ** 0.5)
    if nb_var_to_pred == "all":
        sol = solver(
            W,
            inp_sudoku,
            solution=y_true,
            random=random,
            grid_size=grid_size,
            resolution=resolution,
            margin=margin,
            setUB=setUB,
        )
        sudoku_solved = Sudoku(sol[0]).sudoku_in_line.detach().numpy()
    if nb_var_to_pred == 1:
        sudoku_solved = pred1var(W, inp_sudoku, y_true)
    if isinstance(nb_var_to_pred, int) and nb_var_to_pred > 1:
        sudoku_solved = predNvar(
            W, inp_sudoku, y_true, nb_var_to_pred, resolution=resolution
        )
    L = Hamming(y_true, sudoku_solved)
    grad_w = torch.tensor(gradH(y_true, sudoku_solved, grid_size), requires_grad=True)

    return (grad_w, L)


def pred1var(W, inp_sudoku, y_true):
    """
    Chose randomly 1 variable (not a hint) of inp_sudoku and predict its value knowing
    all other variables true value.
    """

    nb_var = W.shape[-1]
    grid_size = int(nb_var**0.5)

    fixed_var = np.random.randint(1, nb_var + 1)
    while inp_sudoku[fixed_var - 1] != 0:
        fixed_var = np.random.randint(1, nb_var + 1)

    L_cost = []
    for j in range(nb_var):
        if j > fixed_var - 1:
            L_cost.append(
                W[fixed_var - 1, int(j)].reshape(grid_size, grid_size)[
                    :, int(y_true[j]) - 1
                ]
            )

        else:  # invert i and j
            L_cost.append(
                W[int(j), fixed_var - 1].reshape(grid_size, grid_size)[
                    int(y_true[j]) - 1, :
                ]
            )

    L_cost = np.array(L_cost)  # torch.stack(L_cost)
    # pred_value = int(torch.argmin(torch.sum(L_cost, dim = 0))) + 1
    pred_value = int(np.argmin(np.sum(L_cost, axis=0))) + 1

    y_pred = np.copy(y_true)
    y_pred[fixed_var - 1] = pred_value

    return y_pred


def predNvar(W, inp_sudoku, y_true, n=10, resolution=2):
    nb_var = W.shape[-1]
    grid_size = int(nb_var**0.5)
    while torch.sum(inp_sudoku == 0) > n:
        i = np.random.randint(grid_size**2)
        inp_sudoku[i] = int(y_true[i])
    sol = solver(
        W,
        inp_sudoku,
        solution=y_true,
        random=False,
        grid_size=grid_size,
        resolution=resolution,
    )
    sudoku_solved = Sudoku(sol[0]).sudoku_in_line.detach().numpy()

    return sudoku_solved
