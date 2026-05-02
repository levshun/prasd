import os
import math
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


class ForecastEstimator:
    """
    Class for evaluating the quality of the forecasting model.

    Parameters
    -------
    features_names: list or None, default=None
        Name of features.

    Attributes
    ----------
    quality: Pandas `DataFrame`
        Matrix of forecasting quality metrics.
    features_names: list or None, default=None
        Name of features.
    true: Numpy array,
        Data of true values.
    pred: dict of Numpy arrays,
        Data of predicted values for each forecasting models.
    first_batch: Numpy array
        Data of first time window in input data.
    """

    def __init__(self, features_names: list = None):
        """
        """
        self.first_batch = None
        self.true = None
        self.pred = {}
        self.features_names = features_names
        self.quality = pd.DataFrame()

    def set_true_values(self, true: np.ndarray):
        """
        Set true output data.
        """
        assert len(true.shape) in [1, 2, 3], 'True data must be 1D, 2D or 3d numpy array'
        if len(true.shape) == 1:
            true = np.reshape(true, (1, true.shape[0], 1))
        if len(true.shape) == 2:
            true = np.reshape(true, (1, true.shape[0], true.shape[1]))
        self.true = true

    def set_pred_values(self,
                        pred: np.ndarray,
                        model_name: str = 'naive'):
        """
        Set forecasted output data.
        """
        assert len(pred.shape) in [1, 2, 3], 'Predicted data must be 1D, 2D or 3d numpy array'
        if len(pred.shape) == 1:
            pred = np.reshape(pred, (1, pred.shape[0], 1))
        if len(pred.shape) == 2:
            pred = np.reshape(pred, (1, pred.shape[0], pred.shape[1]))

        self.pred[model_name] = pred

    def set_first_batch(self, first_batch: np.ndarray):
        """
        Set first input time window data.
        """
        self.first_batch = first_batch

    def estimate(self, how: str = 'features'):
        """
        Evaluation.
        """
        assert how in ['features', 'timesteps'], 'Unknown "how" argument'

        assert self.true is not None, 'No true values for estimate'
        self.quality = pd.DataFrame()

        if how == 'features':
            if self.features_names is None:
                features_names = [str(i) for i in range(self.true.shape[2])]
            else:
                features_names = self.features_names.copy()
            features_names.append('ALL_FEATURES')

            true_reshaped = self.true.reshape((self.true.shape[0] * self.true.shape[1], self.true.shape[2]))

            for model_name, pred_vals in self.pred.items():
                assert (len(self.true) == len(pred_vals)), f'The length of ' + model_name + \
                                                           ' result not equal to true values'
                pred_reshaped = pred_vals.reshape((self.true.shape[0] * self.true.shape[1], self.true.shape[2]))

                mse = mean_squared_error(true_reshaped, pred_reshaped, multioutput='raw_values')
                mse = np.round(np.append(mse, mean_squared_error(true_reshaped, pred_reshaped)), 8)

                rmse = root_mean_squared_error(true_reshaped, pred_reshaped, multioutput='raw_values')
                rmse = np.append(rmse, root_mean_squared_error(true_reshaped, pred_reshaped)).round(5)
                mae = mean_absolute_error(true_reshaped, pred_reshaped, multioutput='raw_values')
                mae = np.append(mae, mean_absolute_error(true_reshaped, pred_reshaped)).round(5)
                r2 = r2_score(true_reshaped, pred_reshaped, multioutput='raw_values')
                r2 = np.append(r2, r2_score(true_reshaped, pred_reshaped)).round(3)

                self.quality = pd.concat(
                    [self.quality, pd.DataFrame(np.concatenate((mse.reshape(-1, 1), rmse.reshape(-1, 1),
                                                                mae.reshape(-1, 1), r2.reshape(-1, 1)), axis=1),
                                                columns=[model_name + '_MSE', model_name + '_RMSE',
                                                         model_name + '_MAE', model_name + '_R2'])],
                    axis=1)
            self.quality.index = features_names

        if how == 'timesteps':
            self.quality.index = range(self.true.shape[0])
            columns = []
            true_reshaped = self.true.reshape((self.true.shape[0], self.true.shape[1] * self.true.shape[2]))
            for model_name, pred_vals in self.pred.items():
                pred_reshaped = pred_vals.reshape((self.true.shape[0], self.true.shape[1] * self.true.shape[2]))
                mse = np.mean(np.square(np.subtract(pred_reshaped, true_reshaped)), axis=1)
                mae = np.mean(np.abs(pred_reshaped - true_reshaped), axis=1)

                self.quality = pd.concat(
                    [self.quality, pd.DataFrame(np.concatenate((mse.reshape(-1, 1), mae.reshape(-1, 1)), axis=1))],
                    axis=1)
                columns = columns + [f'{model_name}_MSE', f'{model_name}_MAE']
            self.quality.columns = columns

    def save_quality(self, filename: str):
        """
        Save evaluation results to file.
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        self.quality.to_csv('forecaster_results/' + filename + '.csv',
                            index_label='feature')

    def save_pred_result(self, dataset_name):
        """
        Save forecasting models results to file.
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        for model_name, pred_vals in self.pred.items():
            filename = 'forecaster_results/' + dataset_name + '_' + model_name + '.npy'
            np.save(filename, pred_vals)
            print('Save ' + filename)

    def _draw_feature(self, i_feature: int, ax, data_size: int, draw_horizon: int, model_names=None):
        """
        Draw forecasting results for all models and one feature.
        """

        def draw_windows(data, start_x=0, color='indigo', label='Data', alpha=1.0):
            timeline = []
            for p in range(data.shape[0]):
                y = data[p]
                timeline.append(y[0])
                x = range(p + start_x, p + y.shape[0] + start_x)
                if p == (data.shape[0] - 1):
                    ax.plot(x, y, marker='.', color=color, label=label, alpha=alpha, markersize=2)
                else:
                    ax.plot(x, y, marker='.', color=color, alpha=alpha, markersize=2)
            x = range(start_x, len(timeline) + start_x)
            ax.plot(x, timeline, color=color, alpha=alpha)

        target_start_point = 0
        connection_line = {}

        if self.first_batch is not None:
            draw_windows(self.first_batch[:, :, i_feature], color='blue', label='Inputs')
            target_start_point = self.first_batch.shape[1]
            connection_line = {'x': [target_start_point - 1, target_start_point],
                               'y': [self.first_batch[0][-1, i_feature], None]}

        if self.true is not None:
            if connection_line:
                connection_line['y'][1] = self.true[0][0, i_feature]
                plt.plot(connection_line['x'], connection_line['y'], marker='.', color='green')
            draw_windows(self.true[:data_size, :draw_horizon, i_feature], start_x=target_start_point, color='green',
                         label='True')

        if self.pred:
            if model_names is None:
                model_names = list(self.pred.keys())
            color_dicts = {}
            if 'naive' in model_names:
                model_names.remove('naive')
                color_dicts['naive'] = 'grey'

            if len(model_names) == 1:
                color_dicts[model_names[0]] = 'orange'
            else:
                color_map = plt.cm.get_cmap('twilight', len(model_names) + 2)
                for i, model_name in enumerate(model_names):
                    color_dicts[model_name] = color_map(i)

            for model_name, pred_vals in self.pred.items():
                if model_name not in model_names:
                    continue
                if connection_line:
                    connection_line['y'][1] = pred_vals[0][0, i_feature]
                    plt.plot(connection_line['x'], connection_line['y'], marker='.', color=color_dicts[model_name])
                draw_windows(pred_vals[:data_size, :draw_horizon, i_feature], start_x=target_start_point,
                             color=color_dicts[model_name],
                             label=model_name.capitalize())

        ax.legend(fontsize=8)

    def draw(self, size: int = 1000, draw_horizon: int = None, features_names: list | None = None, title: str = '',
             model_names: list | None = None):
        """
        Draw forecasting results.
        """
        if features_names is None:
            if self.features_names is None:
                features_names = [str(i) for i in range(self.true.shape[2])]
            else:
                features_names = self.features_names.copy()

        if draw_horizon is None:
            draw_horizon = self.true.shape[1]

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams.update({'font.size': 14})

        n = len(features_names)
        n_cols = 1
        if n <= 3:
            n_rows = n
        elif n % 3 == 0:
            n_cols = 3
            n_rows = int(n / 3)
        elif n % 2 == 0:
            n_cols = 2
            n_rows = int(n / 2)
        else:
            n_cols = 2
            n_rows = int(math.floor(n / 2))

        fig, axs = plt.subplots(n_rows, n_cols, sharex='col', figsize=(n_cols * 7, n_rows * 4))
        fig.tight_layout(pad=2.3)
        fig.supxlabel('Time')
        fig.suptitle(title)

        i_feature = 0
        for i in range(n_rows):
            if n_cols == 1:
                if n_rows == 1:
                    self._draw_feature(i_feature, axs, size, draw_horizon, model_names=model_names)
                    axs.set_ylabel(features_names[i_feature])
                else:
                    self._draw_feature(i_feature, axs[i], size, draw_horizon, model_names=model_names)
                    axs[i].set_ylabel(features_names[i_feature])
                    axs[i].grid(linestyle=':')
                    if i == n_rows - 1:
                        axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
                    else:
                        axs[i].legend().remove()
                i_feature = i_feature + 1
            else:
                for j in range(n_cols):
                    if i_feature > n:
                        break
                    self._draw_feature(i_feature, axs[i, j], size, draw_horizon, model_names=model_names)
                    axs[i, j].set_ylabel(features_names[i_feature])
                    axs[i, j].grid()
                    i_feature = i_feature + 1
        plt.tight_layout()
        # plt.savefig('output.png', dpi=200)
        plt.show()


def evaluate(y_true: np.ndarray | list,
             y_pred: np.ndarray | list,
             balanced=False) -> dict:
    """
    Evaluate the performance of the classifiers or anomaly detection models.
    Parameters
    -------
    y_true: Numpy array or list
        True labels of data.
    y_pred: Numpy array or list
        Predicted labels of data.
    balanced: boolean, default=False
        Use balanced accuracy or balanced F-score for evaluation.

    Returns
    -------
    Dictionary with evaluation results.

    """
    if balanced:
        counts = {val: y_true.count(val) for val in set(y_true)}
        idx = []
        for val in counts:
            subset = [i for i, x in enumerate(y_true) if x == val]
            idx.extend(random.sample(subset, min(counts.values())))
        y_true = [y_true[i] for i in idx]
        y_pred = [y_pred[i] for i in idx]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=True), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=True), 3),
        "f-measure": round(f1_score(y_true, y_pred, zero_division=True), 3),
        'FPR': round(fp / (fp + tn), 3),
        'FNR': round(fn / (fn + tp), 3)
    }
