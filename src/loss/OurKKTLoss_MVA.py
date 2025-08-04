"""
Rewrite the KKT Loss
https://github.com/RahulNellikkath/Physics-Informed-Neural-Networks-for-AC-Optimal-Power-Flow/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io as sio

mat_path = "data/case39/case39_env.mat"


class OurKKTLoss(nn.Module):
    """
    The current version assume that the topology is the same for all samples in the same batch.
    """

    def __init__(self, env: dict):
        super().__init__()
        self.n_bus = int(env["n_bus"])
        self.n_line = int(env["n_line"])
        self.sbus_idx = int(env["sbus_idx"])
        # register buffers
        self.register_buffer("gbus_idx", env["gbus_idx"].to(torch.long))
        self.register_buffer("branches", env["branches"].to(torch.long))

        for k in [
            "G",
            "B",
            "G_line",
            "B_line",
            "V_min",
            "V_max",
            "P_g_min",
            "P_g_max",
            "Q_g_min",
            "Q_g_max",
            "S_max",
        ]:
            self.register_buffer(k, env[k])

    def forward(
        self,
        # Model Prediction
        V_r: torch.Tensor,  # (bs, n_bus), real part of voltage
        V_i: torch.Tensor,  # (bs, n_bus), imaginary part of voltage
        P_g: torch.Tensor,  # (bs, n_bus), active power generation
        Q_g: torch.Tensor,  # (bs, n_bus), reactive power generation
        #
        P_l: torch.Tensor,  # (bs, n_bus), active power loads
        Q_l: torch.Tensor,  # (bs, n_bus), reactive power loads
        # Dual variables
        n_o_lam_p=None,  # active power injection
        n_o_lam_q=None,  # reactive power injection
        n_o_mu_g_u=None,  # power generation upper bound
        n_o_mu_g_d=None,  # power generation lower bound
        n_o_mu_v_u=None,  # voltage upper bound
        n_o_mu_v_d=None,  # voltage lower bound
        n_o_mu_s_u=None,  # line flow upper bound (MVA)
    ):
        """
        Return: KKT_error: torch.Tensor, size = (bs, )
        """
        ######## ######## ######## ######## ######## ######## ######## ########
        # Constraints
        ######## ######## ######## ######## ######## ######## ######## ########

        # TODO: (P_g, Q_g) on non-generator buses MUST be zero. Should we learn it or manually enforce it?

        # Notice that `KKT_error` should be a Tensor of shape (bs, ), bs is batch size

        # 1. Reference bus
        # equality constraints
        KKT_error = torch.abs(
            V_i[:, self.sbus_idx]
        )  # imaginary part should be zero, i.e., angle = 0

        print('1', KKT_error.mean())

        # 2. Powerflow Equation
        # This might be further accelerated with torch sparse
        # since G and B are typically very sparse,
        # might not be necessary for small graphs

        P_inj = (
            V_r.T * (self.G @ V_r.T - self.B @ V_i.T) + V_i.T * (self.G @ V_i.T + self.B @ V_r.T)
        ).T  # (bs, n_bus), active power injection
        Q_inj = (
            V_i.T * (self.G @ V_r.T - self.B @ V_i.T) - V_r.T * (self.G @ V_i.T + self.B @ V_r.T)
        ).T  # (bs, n_bus), reactive power injection

        # equality constraints
        KKT_error += (P_inj + P_l - P_g).abs().sum(dim=1)
        KKT_error += (Q_inj + Q_l - Q_g).abs().sum(dim=1)

        print('2', KKT_error.mean())

        # 3. Power Generation Violation
        # Power generation on generator buses only
        P_g_gbus = P_g[:, self.gbus_idx]
        Q_g_gbus = Q_g[:, self.gbus_idx]

        # inequality constraints
        KKT_error += (torch.relu(P_g_gbus - self.P_g_max)).sum(dim=1)
        KKT_error += (torch.relu(self.P_g_min - P_g_gbus)).sum(dim=1)
        KKT_error += (torch.relu(Q_g_gbus - self.Q_g_max)).sum(dim=1)
        KKT_error += (torch.relu(self.Q_g_min - Q_g_gbus)).sum(dim=1)

        print('3', KKT_error.mean())

        # 4. Voltage Violation
        V_mag_sq = V_r**2 + V_i**2

        # inequality constraints
        KKT_error += (torch.relu(V_mag_sq - self.V_max**2)).sum(dim=1)
        KKT_error += (torch.relu(self.V_min**2 - V_mag_sq)).sum(dim=1)

        print('4', KKT_error.mean())

        # 5. Line Flow Violation (use S_max only)
        V_r_diff = V_r[:, self.branches[:, 0]] - V_r[:, self.branches[:, 1]]  # (bs, n_line)
        V_i_diff = V_i[:, self.branches[:, 0]] - V_i[:, self.branches[:, 1]]
        I_r = -V_r_diff * self.G_line + V_i_diff * self.B_line
        I_i = -V_i_diff * self.G_line - V_r_diff * self.B_line
        I_mag_sq = I_r**2 + I_i**2

        # voltage magnitude squared at each bus
        V_from_mag_sq = V_mag_sq[:, self.branches[:, 0]]  # (bs, n_line)
        V_to_mag_sq = V_mag_sq[:, self.branches[:, 1]]  # (bs, n_line)

        # apparent power |S|^2 = |V|^2 * |I|^2
        S_from_sq = V_from_mag_sq * I_mag_sq
        S_to_sq = V_to_mag_sq * I_mag_sq

        S_max_sq = self.S_max**2  # (n_line,)
        # Make sure broadcast shapes align: (bs, n_line) - (n_line,) -> ok
        KKT_error += torch.relu(S_from_sq - S_max_sq).sum(dim=1)
        KKT_error += torch.relu(S_to_sq - S_max_sq).sum(dim=1)

        print('5', KKT_error.mean())

        return KKT_error


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
        P_g_min=P_g_min,
        P_g_max=P_g_max,
        Q_g_min=Q_g_min,
        Q_g_max=Q_g_max,
        S_max=S_max,
        baseMVA=baseMVA,
    )
    return env
