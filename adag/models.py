import random
import os
from support.evaluate import ForecastEstimator, evaluate
import scipy.stats as stats
import numpy as np
import pandas as pd
import json


class AnomalyDetectorPro:
    """
    Class to detect anomalies in the dataset.

    Parameters
    -------
    percentile: float, default=0.99
        Threshold value of anomaly probability.
    classes: dict or None, default=None
        Dictionary of classes to use for anomaly detection.
    alpha: float, default=0.4
        Coefficient alpha determines the influence degree of the anomaly probability adjustment
        from the S-SCAg and P-SCAg agents results.
    beta: float, default=0.3
        Coefficient beta determines the degree of reduction in the anomaly probability when agents disagree.
    gamma: float, default=0.2
        Coefficient gamma determines the degree of reduction or increase in the anomaly probability when agents agree.
    shape: float or None, default=None
        Shape parameter controls the profile of the distribution.
    loc: float or None, default=None
        Location parameter shifts the distribution along the x-axis.
    scale: float or None, default=None
        Scale parameter controls the spread of the distribution.

    Attributes
    ----------
    percentile: float, default=0.99
        Threshold value of anomaly probability.
    classes: dict or None, default={0: [0], 1: [1]}
        Dictionary of classes to use for anomaly detection.
    alpha: float, default=0.4
        Coefficient alpha determines the influence degree of the anomaly probability adjustment
        from the S-SCAg and P-SCAg agents results.
    beta: float, default=0.3
        Coefficient beta determines the degree of reduction in the anomaly probability when agents disagree.
    gamma: float, default=0.2
        Coefficient gamma determines the degree of reduction or increase in the anomaly probability when agents agree.
    shape: float
        Shape parameter controls the profile of the distribution.
    loc: float
        Location parameter shifts the distribution along the x-axis.
    scale: float
        Scale parameter controls the spread of the distribution.
    """

    def __init__(self,
                 percentile: float = 0.99,
                 classes: dict | None = None,
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 shape: float | None = None,
                 loc: float | None = None,
                 scale: float | None = None):
        super().__init__()
        self.percentile = percentile
        if classes is None:
            self.classes = {0: [0], 1: [1]}
        else:
            self.classes = classes
        self.alpha = alpha
        self.beta = beta
        if gamma + beta > 1:
            gamma = 1 - beta
        self.gamma = gamma
        self.shape = shape
        self.loc = loc
        self.scale = scale

    def save(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(f'{filename}.json', 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def train(self, true, pred):
        fe = ForecastEstimator()
        fe.set_true_values(true)
        fe.set_pred_values(pred, model_name='model')
        fe.estimate(how='timesteps')
        mse = fe.quality[f'model_MSE']
        # Gamma distribution is ideal because MSE is always positive
        self.shape, self.loc, self.scale = stats.gamma.fit(mse, floc=0)

    def choose_percentile(self,
                          percentiles: list,
                          true_values,
                          pred_values,
                          true_labels,
                          balanced=True,
                          metric='f-measure'):
        results = {}
        idx = []
        if balanced:
            counts = {val: true_labels.count(val) for val in set(true_labels)}
            for val in counts:
                subset = [i for i, x in enumerate(true_labels) if x == val]
                idx.extend(random.sample(subset, min(counts.values())))
            true_labels = [true_labels[i] for i in idx]

        for perc in percentiles:
            self.percentile = perc
            preds = self.simply_run(true_data=true_values, pred_data=pred_values)
            if balanced:
                preds['labels'] = [preds['labels'][i] for i in idx]
            results[perc] = evaluate(true_labels, preds['labels'], balanced=False)
        df = pd.DataFrame.from_dict(results, orient='index')
        self.percentile = df[metric].idxmax()
        print('Percentile: {}'.format(self.percentile))
        return df

    def get_anomaly_probs(self, true, pred):
        fe = ForecastEstimator()
        fe.set_true_values(true)
        fe.set_pred_values(pred, model_name='model')
        fe.estimate(how='timesteps')
        mse = fe.quality[f'model_MSE']
        return stats.gamma.cdf(mse, self.shape, self.loc, self.scale)

    def simply_run(self, true_data, pred_data):
        anomaly_probs = self.get_anomaly_probs(true_data, pred_data)
        return {'labels': (anomaly_probs > self.percentile).astype(int),
                'probability': anomaly_probs}

    def run(self, true_data, pred_data,
            current_labels, pred_labels,
            current_proba, pred_proba):
        start_anomaly_probs = self.get_anomaly_probs(true_data, pred_data)
        anomaly_probs = [0 for _ in range(len(start_anomaly_probs))]
        for i in range(len(current_labels)):
            confA = self._confidence_from_entropy(current_proba[i])
            confB = self._confidence_from_entropy(pred_proba[i])
            avg_conf = 0.5 * (confA + confB)

            agree_is_norm = (current_labels[i] in self.classes[0]) and (pred_labels[i] in self.classes[0])
            agree_is_anomaly = (current_labels[i] in self.classes[1]) and (pred_labels[i] in self.classes[1])
            disagree = current_labels[i] != pred_labels[i]

            if agree_is_anomaly or not (agree_is_norm ^ agree_is_anomaly):
                o = 1.0 if disagree else 1 - self.gamma
            else:
                o = -self.beta if disagree else -self.beta - self.gamma
            factor = 1.0 + self.alpha * avg_conf * o
            anomaly_probs[i] = np.clip(start_anomaly_probs[i] * factor, 0.0, 1.0)
        anomaly_probs = np.array(anomaly_probs)
        return {'labels': (anomaly_probs > self.percentile).astype(int),
                'probability': anomaly_probs}

    @staticmethod
    def _confidence_from_entropy(p):
        p = np.clip(p, 1e-12, 1)
        H = stats.entropy(p, base=2)
        Hmax = np.log2(len(p))
        return 1.0 - H / (Hmax + 1e-12)


def load_anomaly_detector(filename: str) -> AnomalyDetectorPro:
    with open(filename) as f:
        detector_params = json.load(f)
    detector_params['classes'][0] = detector_params['classes'].pop('0')
    detector_params['classes'][1] = detector_params['classes'].pop('1')
    detector = AnomalyDetectorPro(**detector_params)
    return detector
