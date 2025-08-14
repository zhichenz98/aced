"""
The KKT Loss that is only defined on voltage (V_r, V_i)
P_g and Q_g is calculateed from V_r, V_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io as sio


# mat_path = "data/case39/case39_env.mat"


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
            "V_r_base",
            "V_i_base",
            "P_d_base",
            "Q_d_base",
            "P_g_base",
            "Q_g_base",
        ]:
            self.register_buffer(k, env[k])

    def forward(
            self,
            # Model Prediction
            V_r: torch.Tensor,  # (bs, n_bus), real part of voltage
            V_i: torch.Tensor,  # (bs, n_bus), imaginary part of voltage
            #
            P_d: torch.Tensor,  # (bs, n_bus), active power loads
            Q_d: torch.Tensor,  # (bs, n_bus), reactive power loads
    ):
        """
        Since our
        """
        # First Calculate the P_g, Q_g
        P_inj = (
                V_r.T * (self.G @ V_r.T - self.B @ V_i.T) + V_i.T * (self.G @ V_i.T + self.B @ V_r.T)
        ).T  # (bs, n_bus), active power injection
        Q_inj = (
                V_i.T * (self.G @ V_r.T - self.B @ V_i.T) - V_r.T * (self.G @ V_i.T + self.B @ V_r.T)
        ).T  # (bs, n_bus), reactive power injection

        P_g = P_inj + P_d
        Q_g = Q_inj + Q_d

        # 1. Reference bus
        # equality constraints
        KKT_error = torch.abs(
            V_i[:, self.sbus_idx]
        )

        # 3. Power Generation Violation
        # Power generation on generator buses only
        P_g_gbus = P_g[:, self.gbus_idx]
        Q_g_gbus = Q_g[:, self.gbus_idx]

        # inequality constraints
        KKT_error += (torch.relu(P_g_gbus - self.P_g_max)).sum(dim=1)
        KKT_error += (torch.relu(self.P_g_min - P_g_gbus)).sum(dim=1)
        KKT_error += (torch.relu(Q_g_gbus - self.Q_g_max)).sum(dim=1)
        KKT_error += (torch.relu(self.Q_g_min - Q_g_gbus)).sum(dim=1)

        # 4. Voltage Violation
        V_mag_sq = V_r ** 2 + V_i ** 2

        # inequality constraints
        KKT_error += (torch.relu(V_mag_sq - self.V_max ** 2)).sum(dim=1)
        KKT_error += (torch.relu(self.V_min ** 2 - V_mag_sq)).sum(dim=1)

        # 5. Line Flow Violation (use S_max only)
        V_r_diff = V_r[:, self.branches[:, 0]] - V_r[:, self.branches[:, 1]]  # (bs, n_line)
        V_i_diff = V_i[:, self.branches[:, 0]] - V_i[:, self.branches[:, 1]]
        I_r = -V_r_diff * self.G_line + V_i_diff * self.B_line
        I_i = -V_i_diff * self.G_line - V_r_diff * self.B_line
        I_mag_sq = I_r ** 2 + I_i ** 2

        # voltage magnitude squared at each bus
        V_from_mag_sq = V_mag_sq[:, self.branches[:, 0]]  # (bs, n_line)
        V_to_mag_sq = V_mag_sq[:, self.branches[:, 1]]  # (bs, n_line)

        # apparent power |S|^2 = |V|^2 * |I|^2
        S_from_sq = V_from_mag_sq * I_mag_sq
        S_to_sq = V_to_mag_sq * I_mag_sq

        S_max_sq = self.S_max ** 2  # (n_line,)
        # Make sure broadcast shapes align: (bs, n_line) - (n_line,) -> ok
        KKT_error += torch.relu(S_from_sq - S_max_sq).sum(dim=1)
        KKT_error += torch.relu(S_to_sq - S_max_sq).sum(dim=1)

        return KKT_error
