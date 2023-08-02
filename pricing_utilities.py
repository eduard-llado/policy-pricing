import numpy as np
import pandas as pd
from data_utilities import get_clean_data, get_split
from lightgbm import LGBMRegressor


def best_model():
    model = LGBMRegressor(
        n_estimators=700, learning_rate=0.04, verbose=-1, random_state=57
    )
    return model


def evaluate_pricing(y_pred, y_true):
    residual = y_pred - y_true
    print(f"Percent sold: {(residual < 0).mean() * 100:.2f}")
    sold = residual[residual < 0]
    print(f"MAE of sold policies: {-sold.mean():.2f}")


def estimate_bias(y_pred, y_true, bias=0.31):
    bias = -(y_pred - y_true).quantile(bias)
    return bias


def get_price_intervals(y_pred, quantiles=6):
    quantiles = pd.Series(y_pred).quantile(np.arange(0, 1.00001, 1 / quantiles))
    intervals = [[a, b] for a, b in zip(quantiles.values, quantiles.values[1:])]
    intervals[0][0] = float("-inf")
    intervals[-1][1] = float("inf")
    return intervals


def get_indices_by_quantile(y_pred, quantiles=6):
    pred_indices_by_quantile = [
        np.where((l <= y_pred) & (y_pred <= r))[0]
        for l, r in get_price_intervals(y_pred, quantiles=quantiles)
    ]
    return pred_indices_by_quantile


def optimize_biases_greedy(
    y_true,
    y_pred,
    biases=None,
    step=5.0,
    to_sell=0.31,
    n_iters=1000,
    quantiles=6,
):
    if biases is None:
        biases = [0] * quantiles
    to_sell = to_sell * len(y_true)
    residuals = y_pred - y_true
    indices_by_quantile = get_indices_by_quantile(y_pred, quantiles=quantiles)
    residuals_by_quantile = [residuals.take(idx) for idx in indices_by_quantile]
    sold = sum([(r + b < 0).sum() for r, b in zip(residuals_by_quantile, biases)])

    def iterate(step):
        misprice_deltas = []
        sold_deltas = []
        for b, r in zip(biases, residuals_by_quantile):
            current_sold = (r + b < 0).sum()
            new_sold = (r + b + step < 0).sum()
            sold_deltas.append(new_sold - current_sold)
            current_misprice = (r[r + b < 0] + b).sum()
            new_misprice = (r[r + b + step < 0] + b + step).sum()
            misprice_deltas.append(new_misprice - current_misprice)
        gradient = np.array(sold_deltas) / misprice_deltas
        if step > 0:
            best_quantile = gradient.argmax()
        else:
            best_quantile = gradient.argmin()
        biases[best_quantile] += step
        return sold_deltas[best_quantile]

    rng = np.random.RandomState(seed=57)
    for _ in range(n_iters):
        if rng.rand() < 0.5:
            sold += iterate(step)
        else:
            sold += iterate(-step)

    while sold > to_sell:
        sold += iterate(step)
    while sold < to_sell:
        sold += iterate(-step)

    estimated_mae = sum(
        [-(r[r + b < 0] + b).sum() for b, r in zip(biases, residuals_by_quantile)]
    ) / sum([(r + b < 0).sum() for b, r in zip(biases, residuals_by_quantile)])
    return estimated_mae, biases


def apply_biases(prediction, biases, quantiles=6):
    indices_by_quantile = get_indices_by_quantile(prediction, quantiles=quantiles)
    bias_per_pred = [None] * len(prediction)
    for i, idx in enumerate(indices_by_quantile):
        for j in idx:
            bias_per_pred[j] = biases[i]
    assert not any([b is None for b in bias_per_pred])
    return prediction + bias_per_pred
