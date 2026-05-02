import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view



def get_time_windows(x: np.ndarray,
                     window_size: int,
                     step: int | None = 1,
                     timestamp_id: int | None = None) -> (np.ndarray, np.ndarray):
    """
    Get time windows for a series of time steps.
    Parameters
    -------
    x: Numpay array
        Time steps.
    window_size: integer
        Size of the window.
    step: integer, default=1
        Step of the window.
    timestamp_id: integer or None, default=None
        Index of the timestamp.

    Returns
    -------
    Array of time windows (np.ndarray),
    Array of timestamps (np.ndarray)
    """
    if timestamp_id is not None:
        timestamps = sliding_window_view(x[:, timestamp_id], window_size)
        x = np.delete(x, timestamp_id, axis=1)
    else:
        timestamps = sliding_window_view(range(x.shape[0]), window_size)
    if len(x.shape) == 1:
        data = sliding_window_view(x, window_size)
    else:
        data = sliding_window_view(x, window_size, axis=0)
    if step != 1:
        data = data[::step]
        timestamps = timestamps[::step]
    data = np.swapaxes(data, 1, 2)
    data = data.astype('float32')
    return data, timestamps


class RegWindows:
    """
    Time windows generator for regression models.

    Parameters
    ----------
    x: np.ndarray or None, default=None
        Array of time steps.
    look_back_length: int, default=10
        Length of the input time window.
    horizon: int, default=1
        Length of the output time window.
    step: int, default=1
        Length of the time step between windows.
    timestamp_id: int, default=None
        Index of the timestamp.

    Attributes
    ----------
    data: np.ndarray
        Input time windows.
    targets: np.ndarray
        Output time windows.
    data_times: np.ndarray
        Timestamps of input time windows.
    targets_times: np.ndarray
        Timestamps of output time windows.
    """

    def __init__(self,
                 x: np.ndarray | None = None,
                 look_back_length: int = 10,
                 horizon: int = 1,
                 step: int = 1,
                 timestamp_id: int | None = None
                 ):
        if x is not None:
            self.data, self.data_times = get_time_windows(x[:-horizon, :], look_back_length, step, timestamp_id)
            self.targets, self.targets_times = get_time_windows(x[look_back_length:, :], horizon, step, timestamp_id)
        else:
            self.data, self.data_times = None, None
            self.targets, self.targets_times = None, None

    def save(self, folder_name):
        self.save_data(folder_name)
        self.save_targets(folder_name)

    def save_data(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        np.save(f'{folder_name}/data.npy', self.data)
        np.save(f'{folder_name}/data_times.npy', self.data_times)

    def save_targets(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        np.save(f'{folder_name}/targets.npy', self.targets)
        np.save(f'{folder_name}/targets_times.npy', self.targets_times)

    def load_data(self, folder_name):
        self.data = np.load(f'{folder_name}/data.npy', allow_pickle=True)
        if 'data_times.npy' in os.listdir(folder_name):
            self.data_times = np.load(f'{folder_name}/data_times.npy', allow_pickle=True)

    def load_targets(self, folder_name):
        self.targets = np.load(f'{folder_name}/targets.npy', allow_pickle=True)
        if 'targets_times.npy' in os.listdir(folder_name):
            self.targets_times = np.load(f'{folder_name}/targets_times.npy', allow_pickle=True)


def load_reg_windows(folder_name: str) -> RegWindows:
    """Load preprocessed reg windows from files.
    Parameters
    -------
    folder_name: string
        Name of the directory containing the preprocessed reg windows.

    Returns
    -------
    rw: RegWindows
        Time windows for regression.
    """
    rw = RegWindows()
    rw.load_data(folder_name=folder_name)
    rw.load_targets(folder_name=folder_name)
    return rw
