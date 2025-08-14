import numpy as np

# TODO: currently non-zero mask may not make sense if there are both (P_g, Q_g) and (V_r, V_i)

def r2_by_metric(Y_pred, Y):
    """
    Compute R^2 score for each metric, and then average
    """
    # Y shape = (batch size B, num nodes N, num variable D)
    B, N, D = Y.shape
    Y_pred = Y_pred.reshape(B * N, D)
    Y = Y.reshape(B * N, D)

    total_mse = ((Y - Y.mean(axis=0, keepdims=True))**2).mean(axis=0)  # N * D
    Y_res = Y - Y_pred
    residual_mse = ((Y_res - Y_res.mean(axis=0, keepdims=True))**2).mean(axis=0)  # N * D

    mask = total_mse > 0

    r2_mat = 1 - (residual_mse[mask] / total_mse[mask]).mean()

    return r2_mat


def r2_by_metric_by_bus(Y_pred, Y):
    """
    Compute R^2 score for each (bus, metric), and then average

    """
    # Y shape = (batch size B, num nodes N, num variable D)
    # B, N, D = Y.shape
    # Y_pred = Y_pred.reshape(B * N, D)
    # Y = Y.reshape(B * N, D)

    total_mse = ((Y - Y.mean(axis=0, keepdims=True))**2).mean(axis=0)  # N * D
    Y_res = Y - Y_pred
    residual_mse = ((Y_res - Y_res.mean(axis=0, keepdims=True))**2).mean(axis=0)  # N * D

    mask = total_mse > 0

    r2_mat = 1 - (residual_mse[mask] / total_mse[mask]).mean()

    return r2_mat
