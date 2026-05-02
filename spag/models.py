import json
import pandas as pd

import keras_tuner
from keras.models import load_model, Sequential
from keras.layers import Dense, Reshape, LSTM, Input, Dropout, SimpleRNN, GRU
import numpy as np
from keras.src.callbacks import EarlyStopping
from tqdm import tqdm
import sys
from copy import deepcopy

from support.preproc import RegWindows


class BaseForecaster:
    """
    Baseline forecasting model.

    Parameters
    ----------
    model_params: dict or None, default=None
        Dictionary of model parameters.
    how: string, default='naive'
        Type of forecasting method. Options are 'naive' (default) or 'mean'.

    Attributes
    ----------
    model_params: dict or None, default=None
        Dictionary of model parameters.
    how: string, default='naive'
        Type of forecasting method. Options are 'naive' (default) or 'mean'.
    """

    def __init__(self, model_params: dict | None = None,
                 how='naive'):
        self.model_params = model_params
        assert how in ['naive', 'mean'], "How must be 'naive' or 'mean'"
        self.how = how

    def _predict(self,
                 data: np.ndarray,
                 verbose: int = 0) -> np.ndarray:
        """
        Base forecasting function for model.

        Parameters
        -------
        data: Numpy array
            Time series data for forecasting.
        verbose: int, default=0
            Verbosity mode, where 0 = silent, 1 = progress bar.

        Returns
        -------
        predictions: Numpy array
            Forecasting result.
        """
        n_features = data.shape[2]
        predictions = np.empty(shape=(0, self.model_params['horizon'], n_features))
        if verbose == 1:
            pbar = tqdm(desc='Forecasting', total=data.shape[0], file=sys.stdout)
        else:
            pbar = None
        for batch in data:
            current_pred = np.empty(shape=(0, self.model_params['horizon']))
            if self.how == 'naive':
                current_pred = np.array([batch[-1] for i in range(self.model_params['horizon'])])
            if self.how == 'mean':
                current_pred = np.array([np.mean(batch, axis=0) for i in range(self.model_params['horizon'])])
            current_pred = np.reshape(current_pred, (1,) + current_pred.shape)
            predictions = np.concatenate((predictions, current_pred))
            if verbose == 1:
                pbar.update(1)
        if verbose == 1:
            pbar.close()
        return predictions

    def forecasting(self,
                    data: np.ndarray,
                    forecasting_data_length: int = None,
                    verbose: int = 0) -> np.ndarray:
        """
        Main forecasting functions.

        Parameters
        -------
        data: Numpy array
            Data for forecasting.
        forecasting_data_length: integer or None
            Time window size for forecasting.
        verbose: int, default=0
            Verbosity mode, where 0 = silent, 1 = progress bar.

        Returns
        -------
        predictions: Numpy array
            Forecasting result.
        """
        if not forecasting_data_length:
            forecasting_data_length = self.model_params['horizon']
        assert isinstance(forecasting_data_length, int), f"Argument 'forecasting_data_length' must be an integer"
        assert len(data.shape) in [1, 2, 3], 'Data must be 1D, 2D or 3d numpy array'
        if len(data.shape) == 1:
            data = np.reshape(data, (1, data.shape[0], 1))
        if len(data.shape) == 2:
            data = np.reshape(data, (1, data.shape[0], data.shape[1]))

        # Single-step prediction.
        if forecasting_data_length <= self.model_params['horizon']:
            predictions = self._predict(data, verbose=verbose)
            if forecasting_data_length < self.model_params['horizon']:
                for i in range(predictions.shape[0]):
                    predictions[i] = predictions[i][:forecasting_data_length]
            return predictions

        # Multi-step prediction.
        else:
            predictions = np.empty(shape=(data.shape[0], 0, data.shape[2]))
            batch = data
            while predictions.shape[1] < forecasting_data_length:
                current_pred = self._predict(batch, verbose=1)
                predictions = np.append(predictions, current_pred, axis=1)
                batch = np.append(batch[:, self.model_params['horizon']:, :], current_pred, axis=1)
            predictions = predictions[:, :forecasting_data_length, :]
            return predictions


class DeepForecaster(BaseForecaster):
    """
    Deep forecasting models based on RNNs.

    Parameters
    -------
    model_params: dict or None, default=None
        Parameters of forecasting model.
    from_file: string, default=''
        Name of keras model file.
    from_file_config: string, default=''
        Name of keras model configuration JSON file.
    from_config: dict or None, default=None
        Name of keras model configuration file.

    Attributes
    ----------
    model_params: dict or None, default=None
        Parameters of forecasting model.
    model: keras Sequential or None, default=None
        groups of layers.
    model_config: dict, default=None
        Model configuration.
    history: keras.callbacks.History or None, default=None
        History of model training.
    default_name: string, default=''
        Model name as <`block_type`_`units`_`dropout`>.
    """

    def __init__(self,
                 model_params: dict | None = None,
                 from_file: str = '',
                 from_file_config: str = '',
                 from_config: dict = None):
        super().__init__(model_params)
        self.model = None
        self.model_config = None
        self.default_name = ''
        if model_params is not None:
            self.build_model()
        if from_file:
            self.load_from_file(from_file)
        if from_file_config:
            self.load_from_model_config(filename=from_file_config)
        if from_config:
            self.load_from_model_config(config=from_config)
        self.history = None

    def load_from_file(self, filename: str = ''):
        """
        Open the forecasting model from file.
        """
        self.model = load_model(filename)
        self.model_config = self.model.get_config()
        units = [layer['config']['units']
                 for layer in self.model_config['layers'] if layer['class_name'] in ['RNN', 'LSTM', 'GRU']]

        dropout = None
        for layer in self.model_config['layers']:
            if layer['class_name'] == 'Dropout':
                dropout = layer['config']['rate']
                break

        hidden_activation = None
        for layer in self.model_config['layers']:
            if layer['class_name'] in ['RNN', 'LSTM', 'GRU']:
                hidden_activation = layer['config']['activation']
                break

        horizon = None
        for layer in self.model_config['layers']:
            if layer['class_name'] == 'Reshape':
                horizon = layer['config']['target_shape'][0]

        self.model_params = {'n_features': self.model_config['layers'][0]['config']['batch_input_shape'][2],
                             'look_back_length': self.model_config['layers'][0]['config']['batch_input_shape'][1],
                             'horizon': horizon,
                             'n_rec_layers': len(units),
                             'units': {f'units_{i}': u for i, u in enumerate(units)},
                             'dropout': dropout,
                             'block_type': self.model_config['layers'][1]['class_name'],
                             'hidden_activation': hidden_activation,
                             'output_activation': 'tanh',
                             'optimizer': 'adam',
                             'loss': 'mse'}

        self.default_name = self.model_params['block_type'].lower() + '_' + \
                            '_'.join(str(u) for u in self.model_params['units'].values()) + \
                            '_d' + str(self.model_params['dropout']).replace('.', '')

    def load_from_model_config(self, filename: str = '', config: dict = None):
        """
        Create model by keras configuration.
        """
        if filename:
            with open(filename) as f:
                self.model_config = json.load(f)
        if config:
            self.model_config = config

        self.model = Sequential.from_config(self.model_config)
        self.model.compile(optimizer=self.model_params['optimizer'], loss=self.model_params['loss'])

    def save_model_config(self, filename: str):
        """
        Save model configuration in the file.
        """
        assert self.model, 'Model does not exist'
        self.model_config = self.model.get_config()

        with open(filename, 'w') as outfile:
            json_string = json.dumps(self.model_config)
            outfile.write(json_string)
            outfile.write('\n')

    def save_model(self, filename: str):
        """
        Save model in keras format.
        """
        self.model.save(filename)

    def build_model(self):
        """
        Build model in keras format.
        """
        self.model = DeepModel(self.model_params).model
        self.model_config = self.model.get_config()
        self.default_name = self.model_params['block_type'].lower() + '_' + \
                            '_'.join(str(u) for u in self.model_params['units'].values()) + \
                            '_d' + str(self.model_params['dropout']).replace('.', '')

    def summary(self):
        return self.model.summary()

    def train(self,
              data: np.ndarray,
              timestamp_id: int | None = None,
              n_epochs: int = 100,
              batch_size: int = 128,
              verbose: int = 1,
              validation_data: np.ndarray | None = None,
              early_stop_patience: int = 1):
        """
        Training and validation of a forecasting model on data.

        Parameters
        -------
        data: Numpy array
            Training input data.
        timestamp_id: int, default=None
            Index of the timestamp.
        n_epochs: integer, default=100
            Number of epochs to train the model.
        batch_size: integer, default=128
            Number of samples per gradient update.
        verbose: int, default=0
            Verbosity mode, where 0 = silent, 1 = progress bar.
        validation_data:  Tuple `(x_val, y_val)` of Numpy arrays  or `None`, default=None
            Data on which to evaluate the loss at the end of each epoch.
        early_stop_patience: integer, default=1
            Number of epochs with no improvement after which training will be stopped.
        """
        windows = RegWindows(data, self.model_params['look_back_length'], self.model_params['horizon'], step=1,
                             timestamp_id=timestamp_id)
        if validation_data is not None:
            valid_windows = RegWindows(validation_data, self.model_params['look_back_length'],
                                       self.model_params['horizon'], step=1, timestamp_id=timestamp_id)
            valid = (valid_windows.data, valid_windows.targets)
            monitor = 'val_loss'
        else:
            valid = None
            monitor = 'loss'
        early_stop = EarlyStopping(monitor=monitor, patience=early_stop_patience)
        self.history = self.model.fit(windows.data, windows.targets, epochs=n_epochs,
                                      callbacks=[early_stop],
                                      batch_size=batch_size,
                                      validation_data=valid,
                                      shuffle=False,
                                      verbose=verbose)

    def _predict(self, data: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Base forecasting function for deep learning model.
        """
        return self.model.predict(data, batch_size=10, verbose=verbose)


class DeepModel:
    """
    Deep model builder by parameters.

    Parameters
    ----------
    model_params: dict or None, default=None
        Parameters of forecasting model.

    Attributes
    ----------
    model_params: dict or None, default=None
        Parameters of forecasting model.
    model: keras Sequential or None, default=None
        groups of layers.
    """

    def __init__(self, model_params: dict):
        self.model = None
        self.model_params = model_params
        self.build_model()

    def build_model(self):
        """
        Build model by parameters.
        """
        self.model = Sequential()
        self.model.add(Input(shape=(self.model_params['look_back_length'],
                                    self.model_params['n_features'])))

        for n in range(self.model_params['n_rec_layers']):
            units = self.model_params['units'][f'units_{n}']
            activation = self.model_params['hidden_activation']
            last_layer = False
            if n == (self.model_params['n_rec_layers'] - 1):
                last_layer = True
            self._add_hidden_layer(units, activation, last_layer)

            if 'dropout' in self.model_params:
                self.model.add(Dropout(self.model_params['dropout']))

        output_activation = self.model_params['output_activation']
        self.model.add(Dense(self.model_params['horizon'] * self.model_params['n_features'],
                             activation=output_activation))
        # Shape => [batch, out_steps, features].
        self.model.add(Reshape([self.model_params['horizon'], self.model_params['n_features']]))

        self.model.compile(optimizer=self.model_params['optimizer'], loss=self.model_params['loss'])

    def _add_hidden_layer(self, units: int, activation: str, last_layer: bool = False):
        """
        Add new hidden recurrent layer into the model.
        """
        return_sequences = not last_layer

        if self.model_params['block_type'] == 'SimpleRNN':
            self.model.add(SimpleRNN(units=units, activation=activation,
                                     return_sequences=return_sequences))
        if self.model_params['block_type'] == 'LSTM':
            self.model.add(LSTM(units=units, activation=activation,
                                return_sequences=return_sequences))
        if self.model_params['block_type'] == 'GRU':
            self.model.add(GRU(units=units, activation=activation,
                               return_sequences=return_sequences))


class DeepForecasterTuner:
    """
    Tuner for optimizing forecasting model hyperparameters.

    Parameters
    ----------
    params: dict
        Tuned parameters of forecasting model.
    fit_args: dict or None, default=None
        Fixed parameters of forecasting model.
    tuner_type: str, default='RandomSearch'
        Type of optimization algorithm.
    project_name: str, default='spag_tuners/tuner'
        Name of the project to save the tuner results.
    overwrite: bool, default=True
        Whether to overwrite existing tuner.

    Attributes
    ----------
    params: dict
        Parameters of forecasting model, including the tuning parameters.
    fit_args: dict or None, default=None
        Fixed parameters of forecasting model.
    tuner_type: str, default='RandomSearch'
        Type of optimization algorithm.
    project_name: str, default='spag_tuners/tuner'
        Name of the project to save the tuner results.
    overwrite: bool, default=True
        Whether to overwrite existing tuner.
    tuner: Tuner keras class or 'None', default=None
        Model for hyperparameters optimization.
    """

    def __init__(self, params: dict,
                 fit_args: dict | None = None,
                 tuner_type: str = 'RandomSearch',
                 project_name: str = 'spag_tuners/tuner',
                 overwrite=True):
        self.params = params
        self.tuner_type = tuner_type
        if fit_args is None:
            self.fit_args = {}
        else:
            self.fit_args = fit_args
        self.project_name = project_name
        self.overwrite = overwrite
        self.tuner = None

    def get_hps(self, hp: keras_tuner.HyperParameters):
        """
        Set parameters variables for tuning.
        """
        hps = deepcopy(self.params)
        for key, val in self.params.items():
            assert key in ['n_features', 'look_back_length', 'horizon', 'units', 'block_type', 'dropout',
                           'hidden_activation', 'output_activation', 'loss', 'optimizer', 'n_rec_layers'], \
                f"Invalid attribute '{key}'"
            if key != 'units':
                if isinstance(val, list):
                    hps[key] = hp.Choice(key, val)
            else:
                for u, v in val.items():
                    if isinstance(v, list):
                        hps[key][u] = hp.Choice(u, v)
        return hps

    def build_hypermodel(self, hp: keras_tuner.HyperParameters) -> Sequential:
        """
        Build hypermodel takes an argument from which to sample hyperparameters.
        """
        hps = self.get_hps(hp)
        model = DeepModel(hps)
        return model.model

    def _create_tuner(self, tuner_type: str, max_trials: int, project_name: str, use_validation_data: bool):
        """
        Create tuner.
        """
        assert tuner_type in ['GridSearch', 'RandomSearch', 'BayesianOptimization', 'Hyperband'], \
            f'Invalid tuner type, should be GridSearch or RandomSearch or BayesianOptimization or Hyperband'

        # Initialize tuner to run the model.
        if use_validation_data:
            objective = 'val_loss'
        else:
            objective = 'loss'
        if tuner_type == 'RandomSearch':
            self.tuner = keras_tuner.RandomSearch(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,  # the number of different models to try
                project_name=project_name,
                overwrite=self.overwrite
            )
        elif tuner_type == 'BayesianOptimization':
            self.tuner = keras_tuner.BayesianOptimization(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,
                project_name=project_name,
                overwrite=self.overwrite
            )
        elif tuner_type == 'Hyperband':
            self.tuner = keras_tuner.Hyperband(
                hypermodel=self.build_hypermodel,
                objective=objective,
                project_name=project_name,
                overwrite=self.overwrite
            )
        elif tuner_type == 'GridSearch':
            self.tuner = keras_tuner.GridSearch(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,
                project_name=project_name,
                overwrite=self.overwrite
            )

    def _search(self,
                x: np.ndarray,
                y: np.ndarray,
                validation_data: tuple,
                batch_size: int,
                epochs: int,
                patience: int,
                verbose: int):
        """
        Performs a search for best hyperparameter configuations.
        """
        if validation_data is not None:
            if patience != 0:
                self.tuner.search(x, y, validation_data=validation_data,
                                  batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  callbacks=[EarlyStopping(monitor='val_loss',
                                                           patience=patience, restore_best_weights=True)],
                                  verbose=verbose)
            else:
                self.tuner.search(x, y, validation_data=validation_data,
                                  batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=verbose)
        else:
            if patience != 0:
                self.tuner.search(x, y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  callbacks=[EarlyStopping(monitor='loss',
                                                           patience=patience, restore_best_weights=True)],
                                  verbose=verbose)
            else:
                self.tuner.search(x, y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=verbose)

    def search_space_summary(self):
        """
        Print search space summary.
        """
        return self.tuner.search_space_summary()

    def tuning(self,
               data: np.ndarray,
               validation_data: np.ndarray | None = None,
               timestamp_id: int | None = None
               ):
        max_trials = self.fit_args['max_trials'] if 'max_trials' in self.fit_args else 100
        n_models = self.fit_args['n_models'] if 'n_models' in self.fit_args else max_trials
        batch_size = self.fit_args['batch_size'] if 'batch_size' in self.fit_args else 32
        epochs = self.fit_args['epochs'] if 'epochs' in self.fit_args else 100
        patience = self.fit_args['patience'] if 'patience' in self.fit_args else 0
        verbose = self.fit_args['verbose'] if 'verbose' in self.fit_args else 1

        windows = RegWindows(data, self.params['look_back_length'], self.params['horizon'], step=1,
                             timestamp_id=timestamp_id)
        if validation_data is not None:
            valid_windows = RegWindows(validation_data, self.params['look_back_length'], self.params['horizon'], step=1,
                                       timestamp_id=timestamp_id)
            valid = (valid_windows.data, valid_windows.targets)
        else:
            valid = None

        self.params['n_features'] = windows.data.shape[2]

        self._create_tuner(self.tuner_type, max_trials, self.project_name, validation_data is not None)
        self._search(windows.data, windows.targets, valid, batch_size, epochs, patience, verbose)

        results = []
        for trial in self.tuner.oracle.get_best_trials(n_models):
            hyperparameters = trial.hyperparameters.values
            result_hps = deepcopy(self.params)
            if 'units' not in result_hps.keys():
                result_hps['units'] = {}
            for key, value in hyperparameters.items():
                if 'units' in key:
                    if 'n_rec_layers' in hyperparameters.keys():
                        n_rec_layers = hyperparameters['n_rec_layers']
                    else:
                        n_rec_layers = self.params['n_rec_layers']
                    if int(key.split('_')[1]) < n_rec_layers:
                        result_hps['units'][key] = value
                    else:
                        del result_hps['units'][key]
                else:
                    result_hps[key] = value
            results.append({'hyperparameters': result_hps,
                            'score': round(trial.score, 10)})
        return results


def window_tuning(data: np.ndarray,
                  look_back_lengths: list | int,
                  tuner: DeepForecasterTuner,
                  validation: np.ndarray | None,
                  timestamp_id: int | None = None
                  ):
    results = []
    for look_back_length in look_back_lengths:
        for horizon in [
            look_back_length // 4, look_back_length // 2, look_back_length
        ]:
            print(f'look_back_length={look_back_length}', f'horizon={horizon}')
            tuner.params['look_back_length'] = look_back_length
            tuner.params['horizon'] = horizon
            best_hps = tuner.tuning(data=data, validation_data=validation, timestamp_id=timestamp_id)
            results.extend(best_hps)
    df = _get_hps_table(results)
    return df


def _get_hps_table(hps_list: list):
    df = pd.DataFrame([hps['hyperparameters'] for hps in hps_list])
    df['score'] = [hps['score'] for hps in hps_list]
    df = df.sort_values(by='score')
    df['temp'] = df['units'].apply(lambda x: frozenset(x.items()))
    dcolumns = list(df.columns)
    dcolumns.remove('units')
    dcolumns.remove('score')
    df = df.drop_duplicates(subset=dcolumns).drop(columns=['temp'])
    df = df.reset_index(drop=True)
    return df


def _state_decision(state_dict: dict,
                    hp_names: list,
                    ):
    ranks_list = []
    for state, hps in state_dict.items():
        ranks = pd.DataFrame()
        ranks['hp'] = hps[hp_names].astype(str).agg(', '.join, axis=1)
        ranks['rank'] = hps.index.values
        ranks_list.append(ranks)
    df = pd.concat(ranks_list).groupby('hp').sum()
    df = df / len(state_dict)
    df = df.sort_values('rank')
    return df
