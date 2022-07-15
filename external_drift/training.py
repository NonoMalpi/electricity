import time

from typing import Dict, Tuple

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from torchdiffeq import odeint

from external_drift.utils import ScenarioParams, get_multivariate_batch, get_mean_tensor_from_training_set
from neural_ode import NeuralODEfunc, RunningAverageMeter
from plot.external_drift import plot_training_evaluation


def train_neural_ode_step(k: int,
                          neural_ode: NeuralODEfunc,
                          optimizer: torch.optim.Optimizer,
                          loss_meter: RunningAverageMeter,
                          batch_y0: torch.Tensor,
                          batch_t: torch.Tensor,
                          batch_y: torch.Tensor
                          ) -> Tuple[int, NeuralODEfunc, torch.optim.Optimizer, RunningAverageMeter]:
    """ Compute the neural ODE through the specified time steps, calculate loss and update neural ODE parameters.

    Parameters
    ----------
    k: int
        Auxiliary index to indicate the neural ODE to train.

    neural_ode: NeuralODEfunc
        Class containing the neural ODE architecture.

    optimizer: torch.optim.Optimizer
        Torch optimizer to use in the training process.

    loss_meter: RunningAverageMeter
        Class computing and storing the average loss.

    batch_y0: torch.Tensor
        Tensor containing the initial values for the neural ODE (batch_size, 1, obs_dim).

    batch_t: torch.Tensor
        Tensor containing the time steps to use in the neural ODE (time_period + 1, ).

    batch_y: torch.Tensor
        Tensor containing the whole trajectory (time_period + 1, batch_size, 1, obs_dim).

    Returns
    -------
     k: int
        Auxiliary index to indicate the neural ODE to train.

    neural_ode: NeuralODEfunc
        Class containing the neural ODE architecture with the updated parameters.

    optimizer: torch.optim.Optimizer
        Torch optimizer used and updated in the training process.

    loss_meter: RunningAverageMeter
        Class storing the calculated loss.

    """
    optimizer.zero_grad()

    pred_y = odeint(neural_ode, batch_y0, batch_t)

    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()

    loss_meter.update(loss.item())
    return k, neural_ode, optimizer, loss_meter


def train_multivariate_neural_ode_external_drift(params: ScenarioParams,
                                                 hidden_layer_neurons: int,
                                                 learning_rate: float,
                                                 train_df: pd.DataFrame) -> Dict[int, torch.Tensor]:
    """ Train a simple 1 hidden layer neural ODE with input dimension equal to params.obs_dim and evaluate test time steps.

    The neural ODE is trained following a time-sequential approach, incrementally adding observations.
    See: https://sebastiancallh.github.io/post/neural-ode-weather-forecast/.
    Each time step is fitted through params.epochs. At each epoch, the neural ODE evaluates a random batch of
    size params.batch_size and the neurons weights are updated. After finishing all epochs,
    the neural ODE evaluates the following time step as test case.
    The process is repeated to learn the next time step keeping the same neurons weights of the previous time step
    as the initial case, i.e., the neurons weights at the end of the last epoch of time step 1 are kept as the
    initial weights params of the neural ODE at epoch 0 to learn time step 2. This sequential process improves convergence.

    Parameters
    ----------
    params: ScenarioParams
        Class containing simulation parameters and auxiliary information.

    hidden_layer_neurons: int
        Number of neurons in the hidden layer.

    learning_rate: float
        Learning rate to use in the optimizer.

    train_df: pd.DataFrame
        The training set with MultiIndex(hour, time step) and number of simulation as columns.

    Returns
    -------
    pred_ext_drift_dict: Dict[int, torch.Tensor]
        Dictionary containing training time step as keys and predicted tensor
        of shape (training time step + 2, 1, 1, obs_dim) as values.
    """

    device = torch.device("cpu")

    pred_ext_drift_dict = {}

    init_window_length = 0

    func = NeuralODEfunc(obs_dim=params.obs_dim, hidden_layer_1=hidden_layer_neurons).to(device)
    optimizer = torch.optim.RMSprop(func.parameters(), lr=learning_rate)

    start = time.time()
    for j in range(1, params.sim_periods - 1):
        training_ts = init_window_length + params.delta_t * j

        loss_meter = RunningAverageMeter(0.97)
        for itr in range(0, params.epochs + 1):
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_multivariate_batch(train_df=train_df,
                                                                time_period=training_ts,
                                                                params=params)
            batch_y0 = batch_y0.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            pred_y = odeint(func, batch_y0, batch_t).to(device)

            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            end = time.time()
            training_time = (end - start) / 60

            if itr % (params.epochs // 8) == 0:
                print(f"Training time step {j} - Iteration: {itr:04d} | Total loss {loss_meter.avg:.6f} | Time: {training_time:.2f} mins")

            if itr % (params.epochs // 2) == 0:

                # evaluate training and test set
                true_y0 = get_mean_tensor_from_training_set(train_df=train_df, time_step=0).to(device)
                batch_test_t = torch.from_numpy(np.arange(training_ts + 2, dtype=float))
                true_test_y = get_mean_tensor_from_training_set(train_df=train_df, time_step=training_ts + 1).to(device)

                with torch.no_grad():
                    pred_test_y = odeint(func, true_y0, batch_test_t)
                    plot_training_evaluation(pred_tensor=pred_test_y,
                                             train_df=train_df,
                                             training_ts=training_ts,
                                             params=params,
                                             rows=3,
                                             columns=8)
                    plt.show()

                    test_loss = torch.mean(torch.abs(true_test_y - pred_test_y[-1]))
                    print(f"Mean absolute value error for test: {test_loss:.2f}")
                    if itr == params.epochs:
                        pred_ext_drift_dict[training_ts] = pred_test_y

                print("\n" + "=" * 115 + "\n")

    return pred_ext_drift_dict


def train_univariate_neural_ode_external_drift(params: ScenarioParams,
                                               hidden_layer_neurons: int,
                                               learning_rate: float,
                                               train_df: pd.DataFrame):
    """ Train as many simple 1 hidden layer neural ODE with input dimension 1 as params.obs_dim and evaluate test time steps.

    The neural ODE is trained following a time-sequential approach, incrementally adding observations.
    See: https://sebastiancallh.github.io/post/neural-ode-weather-forecast/.
    Each time step is fitted through params.epochs. At each epoch, the neural ODE evaluates a random batch of
    size params.batch_size and the neurons weights are updated. After finishing all epochs,
    the neural ODE evaluates the following time step as test case.
    The process is repeated to learn the next time step keeping the same neurons weights of the previous time step
    as the initial case, i.e., the neurons weights at the end of the last epoch of time step 1 are kept as the
    initial weights params of the neural ODE at epoch 0 to learn time step 2. This sequential process improves convergence.

    Parameters
    ----------
    params: ScenarioParams
        Class containing simulation parameters and auxiliary information.

    hidden_layer_neurons: int
        Number of neurons in the hidden layer.

    learning_rate: float
        Learning rate to use in the optimizer.

    train_df: pd.DataFrame
        The training set with MultiIndex(hour, time step) and number of simulation as columns.

    Returns
    -------
    pred_ext_drift_dict: Dict[int, torch.Tensor]
        Dictionary containing training time step as keys and predicted tensor
        of shape (training time step + 2, 1, 1, obs_dim) as values.
    """
    device = torch.device("cpu")

    init_window_length = 0

    pred_ext_drift_dict = {}

    neural_odes_dict = {}
    optimizer_dict = {}

    # initialise as many univariate neural ODEs as obs_dim
    for k in range(1, params.obs_dim + 1):
        func = NeuralODEfunc(obs_dim=1, hidden_layer_1=hidden_layer_neurons).to(device)
        neural_odes_dict[k] = func
        optimizer_dict[k] = torch.optim.RMSprop(func.parameters(), lr=learning_rate)

    start = time.time()
    for j in range(1, params.sim_periods - 1):
        training_ts = init_window_length + params.delta_t * j

        loss_meter_dict = {k: RunningAverageMeter(0.97) for k in range(1, params.obs_dim + 1)}
        for itr in range(0, params.epochs + 1):
            batch_y0, batch_t, batch_y = get_multivariate_batch(train_df=train_df,
                                                                time_period=training_ts,
                                                                params=params)
            batch_y0 = batch_y0.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            train_step_list = Parallel(n_jobs=-1, verbose=0)(
                delayed(train_neural_ode_step)(
                    k=k,
                    neural_ode=neural_odes_dict[k],
                    optimizer=optimizer_dict[k],
                    loss_meter=loss_meter_dict[k],
                    batch_y0=batch_y0[:, :, k - 1].reshape(params.batch_size, 1, 1),
                    batch_t=batch_t,
                    batch_y=batch_y[:, :, :, k - 1].reshape(-1, params.batch_size, 1, 1)
                ) for k in range(1, params.obs_dim + 1)
            )

            # update dictionaries, this is a faster operation that sharing memory to update dictionaries inside
            # parallel computation through Parallel(..., backend="threading")
            for element in train_step_list:
                k = element[0]
                neural_odes_dict[k] = element[1]
                optimizer_dict[k] = element[2]
                loss_meter_dict[k] = element[3]

            end = time.time()
            training_time = (end - start) / 60
            if itr % (params.epochs // 8) == 0:
                loss_array = np.array([value.avg for k, value in loss_meter_dict.items()])
                loss_mean = loss_array.mean()
                print(f"Training time step {j} - Iteration: {itr:04d} | Total loss {loss_mean:.6f} | Time: {training_time:.2f} mins")

            if itr % (params.epochs // 2) == 0:
                true_y0 = get_mean_tensor_from_training_set(train_df=train_df, time_step=0).to(device)
                batch_test_t = torch.from_numpy(np.arange(training_ts + 2, dtype=float))
                true_test_y = get_mean_tensor_from_training_set(train_df=train_df, time_step=training_ts + 1).to(device)

                with torch.no_grad():
                    pred_test_y_list = []
                    for k in range(1, params.obs_dim + 1):
                        pred_test_y_k = odeint(neural_odes_dict[k], true_y0[:, :, k - 1].reshape(1, 1, 1), batch_test_t)
                        pred_test_y_list.append(pred_test_y_k)
                    pred_test_y = torch.hstack(pred_test_y_list).reshape(training_ts + 2, 1, 1, -1)
                    plot_training_evaluation(pred_tensor=pred_test_y,
                                             train_df=train_df,
                                             training_ts=training_ts,
                                             params=params,
                                             rows=3,
                                             columns=8)
                    plt.show()

                    test_loss = torch.mean(torch.abs(true_test_y - pred_test_y[-1]))
                    print(f"Mean absolute value error for test: {test_loss:.2f}")
                    if itr == params.epochs:
                        pred_ext_drift_dict[training_ts] = pred_test_y

                print("\n" + "=" * 115 + "\n")

    return pred_ext_drift_dict
