import argparse
import logging
import pickle
import time

from typing import AnyStr, NoReturn, Tuple

import emd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from electricity.ml.neural_network import MeanSquaredError, NeuralNetFunc

from electricity.external_drift import ScenarioParams, SingleMultivariateNeuralODE, MultipleUnivariateNeuralODE

from utils import NeuralOdeEMDResult

logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s|%(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("NeuralODE-EMD-training")
parser.add_argument("--hour", type=int, choices=range(1, 25))
parser.add_argument("--train_length", type=int, choices=range(10, 10000), default=60)
parser.add_argument("--epochs", type=int, default=2_000)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()



def load_file(input_file_path: AnyStr) -> pd.DataFrame:
    return pd.read_csv(input_file_path)


def plot_imfs(imfs: np.ndarray) -> NoReturn:
    fig, axis = plt.subplots(imfs.shape[1], 1, figsize=(20, 15))

    max_value = np.abs(imfs).max()

    for i, ax in enumerate(axis):
        ax.plot(np.arange(0, imfs.shape[0]), imfs[:, i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlim(0, imfs.shape[0])
        ax.set_ylim(-max_value, max_value)

        ax.set_title(f"IMF - {i + 1}", fontsize=18)
        ax.axhline(y=0, color="grey", alpha=0.2)
        ax.tick_params(axis="both", labelsize=18)
        if i != imfs.shape[1] - 1:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xlabel("time", fontsize=18)
            ax.set_xticks(ticks=np.arange(0, imfs.shape[0]+1, 5))

    plt.savefig(f"scripts/neurips/figures/imfs/imfs_hour_{args.hour}.jpeg")


def apply_emd(df: pd.DataFrame) -> np.ndarray:
    imfs = emd.sift.sift(df.iloc[:, 0].values)
    logging.info(f"Number of IMFs: {imfs.shape[-1]}")

    return imfs


def preprocess_imfs(imfs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = imfs.mean(axis=0)
    std = imfs.std(axis=0)

    standard_imfs = (imfs - mean) / std
    return standard_imfs, mean, std


def get_training_arrays(dataset: np.ndarray, ts: int, params: ScenarioParams):
    batch_t = torch.from_numpy(np.arange(ts, dtype=float)) / 100
    y = dataset[:ts, :]
    batch_y = torch.from_numpy(y.reshape(y.shape[0], 1, 1, params.obs_dim)).float()
    batch_y0 = batch_y[0]

    return batch_y0, batch_t, batch_y


def get_batch_training_arrays(dataset: np.ndarray, params: ScenarioParams, wl: int):
    s = torch.from_numpy(
        np.random.choice(np.arange(dataset.shape[0] - wl, dtype=np.int64), params.batch_size, replace=False))
    batch_y0 = torch.from_numpy(dataset[s].reshape(params.batch_size, 1, params.obs_dim)).float()
    batch_t = torch.from_numpy(np.arange(wl, dtype=float)) / 100
    batch_y = torch.stack([torch.from_numpy(dataset[s + i]) for i in range(wl)], dim=0).reshape(
        wl, params.batch_size, 1, params.obs_dim).float()

    return batch_y0, batch_t, batch_y


def save_result(result: NeuralOdeEMDResult) -> NoReturn:
    with open(f"scripts/neurips/data/result_hour_{result.hour}_v3.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    #df = load_file(input_file_path="scripts/neurips/data/electricity_time_series.csv")

    #df = load_file(input_file_path="/Users/am2121/Downloads/monthly_csv.csv")

    #hour_df = load_file(input_file_path="scripts/neurips/test_time_delay.csv")
    hour_df = load_file(input_file_path="scripts/neurips/test_time_delay_electricity.csv")


    hour = args.hour
    #hour_df = df.query(f"hour == {hour}")[["date", "spain"]].set_index("date")
    #hour_df = df.query(f"Source == 'GCAG'").set_index("Date")[["Mean"]].sort_index()

    train_length = args.train_length + 1
    train_hour_df, test_hour_df = hour_df.iloc[:train_length], hour_df.iloc[train_length:]

    #imfs = apply_emd(df=train_hour_df)
    imfs = train_hour_df.values

    plot_imfs(imfs=imfs)

    standard_imfs, mean_train, std_train = preprocess_imfs(imfs=imfs)

    hidden_layer_neurons = [500] * 2
    activation_functions = [torch.nn.Tanh()] * 2
    loss_func = MeanSquaredError()

    params = ScenarioParams(sim_periods=20_000,
                            n_sim=1_000,
                            delta_t=5,
                            seed=1_024,
                            batch_size=30,
                            obs_dim=standard_imfs.shape[1],
                            epochs=args.epochs,
                            lr=args.lr)

    device = torch.device("cpu")

    init_window_length = 52

    #assert params.sim_periods * params.delta_t + init_window_length == standard_imfs.shape[0], \
    #    f"shapes do not match: {params.sim_periods * params.delta_t + init_window_length} and {standard_imfs.shape[0]}"

    func = NeuralNetFunc(obs_dim=standard_imfs.shape[1],
                         hidden_layer_neurons=hidden_layer_neurons,
                         activation_functions=activation_functions)

    node = SingleMultivariateNeuralODE(params=params,
                                       neural_ode_template=func,
                                       optimizer=torch.optim.Adam,
                                       loss_func=loss_func,
                                       loss_momentum=0.97)

    #node = MultipleUnivariateNeuralODE(params=params,
    #                                   neural_ode_template=func,
    #                                   optimizer=torch.optim.Adam,
    #                                   loss_func=loss_func,
    #                                   loss_momentum=0.97)

    test_size = 4
    node.initialize_loss_meter()
    start = time.time()

    result = NeuralOdeEMDResult(hour=hour,
                                time_series_df=hour_df,
                                training_df=train_hour_df,
                                test_df=test_hour_df,
                                train_imfs=imfs,
                                standard_imfs=standard_imfs,
                                mean_train=mean_train,
                                std_train=std_train,
                                init_window_length=init_window_length,
                                test_size=test_size,
                                step_length=params.delta_t,
                                periods=params.sim_periods)

    for j in range(0, params.sim_periods):
        training_ts = init_window_length + params.delta_t * j

        #batch_y0, batch_t, batch_y = get_training_arrays(dataset=standard_imfs, ts=training_ts, params=params)
        batch_y0, batch_t, batch_y = get_batch_training_arrays(dataset=standard_imfs, params=params, wl=20)

        batch_y0 = batch_y0.to(device)
        batch_t = batch_t.to(device)
        batch_y = batch_y.to(device)

        #for itr in range(0, params.epochs + 1):

        node.train(batch_y0=batch_y0, batch_t=batch_t, batch_y=batch_y)

        if j % (params.sim_periods // 40) == 0:
            end = time.time()
            training_time = (end - start) / 60
            logging.info(
                f"Training time step {j} - Iteration: {j:04d} | "
                f"Total loss {node.loss():.6f} | Time: {training_time:.2f} mins"
            )

            #true_y0, true_t, true_y = get_training_arrays(
            #    dataset=standard_imfs, ts=training_ts + test_size, params=params
            #)

            true_y0, true_t, true_y = get_training_arrays(
                dataset=standard_imfs, ts=400, params=params
            )

            with torch.no_grad():
                pred_test_y = node.solve_initial_value(batch_y0=true_y0, batch_t=true_t)

                pred_test_y = pred_test_y.reshape(true_t.shape[0], standard_imfs.shape[1])

                result.update_training(j=j,
                                       itr=j,
                                       ts=training_ts,
                                       pred_test_y=pred_test_y,
                                       loss=node.loss(),
                                       t=training_time)

    result.save_neural_ode(node=node)

    save_result(result=result)
