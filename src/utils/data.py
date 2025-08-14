import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import os


def load_env_mat(mat_path: str, device="cpu", dtype=torch.float32):
    """
    Load env saved in MATLAB (save_env) and keep only S_max for line limits.

    Returns a dict with tensors already on `device`:
      - counts: n_bus, n_line, n_gbus, n_lbus, sbus_idx
      - indices: gbus_idx (list[int] for compatibility with your loss), branches (LongTensor)
      - matrices/vectors: G, B, G_line, B_line, S_max, V_min, V_max, P_g_min/Max, Q_g_min/Max
    """
    M = sio.loadmat(mat_path, squeeze_me=True)

    def T(x, as_long=False):
        if as_long:
            return torch.as_tensor(np.array(x, dtype=np.int64), device=device)
        return torch.as_tensor(np.array(x, dtype=np.float32), device=device, dtype=dtype)

    G = T(M["G"])
    B = T(M["B"])
    branches = T(M["branches"], as_long=True)  # (n_line, 2), 0-based from MATLAB
    G_line = T(M["G_line"])
    B_line = T(M["B_line"])

    V_min = T(M["V_min"])
    V_max = T(M["V_max"])

    P_g_min = T(M["P_g_min"])
    P_g_max = T(M["P_g_max"])
    Q_g_min = T(M["Q_g_min"])
    Q_g_max = T(M["Q_g_max"])

    gen_bus_idx = T(M["gen_bus_idx"], as_long=True)
    load_bus_idx = T(M["load_bus_idx"], as_long=True)
    slack_bus_idx = T(M["slack_bus_idx"], as_long=True)
    baseMVA = float(M["baseMVA"])

    # Base case for P_d, Q_d, P_g, Q_g, V_r, V_i

    P_d_base = M["P_d_base"]
    Q_d_base = M["Q_d_base"]

    V_r_base = M["V_r_base"]
    V_i_base = M["V_i_base"]

    # S_max present in MATLAB file
    if "S_max" not in M:
        raise KeyError("env.mat doesn't contain S_max. Please save it in MATLAB (RATE_A).")
    S_max = T(M["S_max"])

    n_bus = G.shape[0]
    n_line = branches.shape[0]
    n_gbus = gen_bus_idx.numel()
    n_lbus = load_bus_idx.numel()

    if slack_bus_idx.numel() != 1:
        raise ValueError(f"Expect exactly one slack bus. Got {slack_bus_idx.tolist()}")

    sbus_idx = int(slack_bus_idx.item())

    # TODO: This 100 is hardcoded. Please check it (should that be MVA?)

    env = dict(
        n_bus=n_bus,
        n_line=n_line,
        n_gbus=n_gbus,
        n_lbus=n_lbus,
        sbus_idx=sbus_idx,
        gbus_idx=gen_bus_idx,
        G=G,
        B=B,
        branches=branches,
        G_line=G_line,
        B_line=B_line,
        V_min=V_min,
        V_max=V_max,
        P_g_min=P_g_min / 100,
        P_g_max=P_g_max / 100,
        Q_g_min=Q_g_min / 100,
        Q_g_max=Q_g_max / 100,
        V_r_base=V_r_base,
        V_i_base=V_i_base,
        P_d_base=P_d_base / 100,
        Q_d_base=Q_d_base / 100,
        S_max=S_max,
        baseMVA=baseMVA,
    )
    return env


def load_csv(csv_path, n_bus):
    df = pd.read_csv(csv_path)
    PQ_cols = [f"{metric}_{bus_id + 1}" for metric in ['p_d', 'q_d', 'p_g', 'q_g'] for bus_id in range(n_bus)]
    df[PQ_cols] /= 100.0

    return df
