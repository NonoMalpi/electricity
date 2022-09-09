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

from electricity.external_drift import ScenarioParams, SingleMultivariateNeuralODE

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser("NeuralODE-EMD-training")
parser.add_argument("--hour", type=int, choices=range(1, 25))
parser.add_argument("--train_length", type=int, choices=range(20, 100), default=60)
parser.add_argument("--epochs", type=int, default=2_000)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()


class NeuralOdeEMDResult:

    def __init__(self,
                 hour: int,
                 time_series_df: pd.DataFrame,
                 training_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 train_imfs: np.ndarray,
                 standard_imfs: np.ndarray,
                 mean_train: np.ndarray,
                 std_train: np.ndarray,
                 init_window_length: int,
                 test_size: int,
                 step_length: int,
                 periods: int
                 ):

        self.hour = hour
        self.time_series_df = time_series_df
        self.training_df = training_df
        self.test_df = test_df
        self.train_imfs = train_imfs
        self.standard_imfs = standard_imfs
        self.mean_train = mean_train
        self.std_train = std_train
        self.init_window_length = init_window_length
        self.test_size = test_size
        self.step_length = step_length
        self.prediction_dict = {j: {} for j in range(periods)}
        self.node: SingleMultivariateNeuralODE = None

    def update_training(self, j: int, itr: int, ts: int, pred_test_y: torch.Tensor, loss: float, t: float) -> NoReturn:
        self.prediction_dict[j][itr] = {
            "ts": ts,
            "pred_test_y": pred_test_y.numpy(),
            "loss": round(loss, 4),
            "time": round(t, 2)
        }

    def save_neural_ode(self, node: SingleMultivariateNeuralODE) -> NoReturn:
        self.node = node

    def predict_imfs(self, ts: int) -> np.ndarray:
        y0 = torch.from_numpy(self.standard_imfs[0, :]).float()
        t = torch.from_numpy(np.arange(ts, dtype=float)) / 10

        trajectories = self.node.solve_initial_value(batch_y0=y0, batch_t=t).detach().numpy()
        trajectories = trajectories.reshape(ts, self.standard_imfs.shape[-1])

        trajectories = (trajectories * self.std_train) + self.mean_train

        return trajectories

    def predict_time_series(self, ts: int) -> np.ndarray:
        imfs_trajectories = self.predict_imfs(ts=ts)

        return imfs_trajectories.sum(axis=1)


def load_file(input_file_path: AnyStr) -> pd.DataFrame:
    return pd.read_csv(input_file_path)


def plot_imfs(imfs: np.ndarray) -> NoReturn:
    fig, axis = plt.subplots(imfs.shape[1], 1, figsize=(20, 15))

    max_value = np.abs(imfs.shape[0]).max()

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
    batch_t = torch.from_numpy(np.arange(ts, dtype=float)) / 10
    y = dataset[:ts, :]
    batch_y = torch.from_numpy(y.reshape(y.shape[0], 1, 1, params.obs_dim)).float()
    batch_y0 = batch_y[0]

    return batch_y0, batch_t, batch_y


def save_result(result: NeuralOdeEMDResult) -> NoReturn:
    with open(f"scripts/neurips/data/result_hour_{result.hour}.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    df = load_file(input_file_path="scripts/neurips/data/electricity_time_series.csv")

    hour = args.hour
    hour_df = df.query(f"hour == {hour}")[["date", "spain"]].set_index("date")

    train_length = args.train_length + 1
    train_hour_df, test_hour_df = hour_df.iloc[:train_length], hour_df.iloc[train_length:]

    imfs = apply_emd(df=train_hour_df)

    plot_imfs(imfs=imfs)

    standard_imfs, mean_train, std_train = preprocess_imfs(imfs=imfs)

    hidden_layer_neurons = [50]
    activation_functions = [torch.nn.Tanh()]
    loss_func = MeanSquaredError()

    params = ScenarioParams(sim_periods=28,
                            n_sim=1_000,
                            delta_t=2,
                            seed=1_024,
                            batch_size=10,
                            obs_dim=standard_imfs.shape[1],
                            epochs=args.epochs,
                            lr=args.lr)

    device = torch.device("cpu")

    init_window_length = 5

    assert params.sim_periods * params.delta_t + init_window_length == standard_imfs.shape[0], \
        f"shapes do not match: {params.sim_periods * params.delta_t + init_window_length} and {standard_imfs.shape[0]}"

    func = NeuralNetFunc(obs_dim=standard_imfs.shape[1],
                         hidden_layer_neurons=hidden_layer_neurons,
                         activation_functions=activation_functions)

    node = SingleMultivariateNeuralODE(params=params,
                                       neural_ode_template=func,
                                       optimizer=torch.optim.Adam,
                                       loss_func=loss_func,
                                       loss_momentum=0.97)

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

        batch_y0, batch_t, batch_y = get_training_arrays(dataset=standard_imfs, ts=training_ts, params=params)

        batch_y0 = batch_y0.to(device)
        batch_t = batch_t.to(device)
        batch_y = batch_y.to(device)

        for itr in range(0, params.epochs + 1):

            node.train(batch_y0=batch_y0, batch_t=batch_t, batch_y=batch_y)

            if itr % (params.epochs // 4) == 0:
                end = time.time()
                training_time = (end - start) / 60
                logging.info(
                    f"Training time step {j} - Iteration: {itr:04d} | "
                    f"Total loss {node.loss():.6f} | Time: {training_time:.2f} mins"
                )

                true_y0, true_t, true_y = get_training_arrays(
                    dataset=standard_imfs, ts=training_ts + test_size, params=params
                )

                with torch.no_grad():
                    pred_test_y = node.solve_initial_value(batch_y0=true_y0, batch_t=true_t)

                    pred_test_y = pred_test_y.reshape(true_t.shape[0], standard_imfs.shape[1])

                    result.update_training(j=j,
                                           itr=itr,
                                           ts=training_ts,
                                           pred_test_y=pred_test_y,
                                           loss=node.loss(),
                                           t=training_time)

    result.save_neural_ode(node=node)

    save_result(result=result)
