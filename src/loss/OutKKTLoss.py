"""
Rewrite the KKT Loss
https://github.com/RahulNellikkath/Physics-Informed-Neural-Networks-for-AC-Optimal-Power-Flow/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OurKKTLoss(nn.Module):
    """
    The current version assume that the topology is the same for all samples in the same batch.
    """

    def __init__(self):
        super().__init__()

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
            # n_o_l_p,
            # n_o_mu_g_u: torch.Tensor,  # power generation upper bound
            # n_o_mu_g_d: torch.Tensor,  # power generation lower bound
            # n_o_mu_v_u: torch.Tensor,  # voltage upper bound
            # n_o_mu_v_d: torch.Tensor,  # voltage lower bound
            # n_o_mu_i_u: torch.Tensor,  # line flow upper bound

            # Topology info
            env: dict = None,
    ):
        ######## ######## ######## ######## ######## ######## ######## ########
        # Preparation for the Environment
        ######## ######## ######## ######## ######## ######## ######## ########

        n_bus: int = env['n_bus']  # number of all buses
        n_gbus: int = env['n_gbus']  # number of generators
        n_lbus: int = env['n_lbus']  # number of loads

        sbus_idx: int = env['sbus_idx']  # index of (the only) slack bus
        gbus_idx = torch.LongTensor(env['gbus_idx'])  # list of int -> torch.LongTensor, indices of generators

        n_line: int = env['n_line']  # int, number of lines

        G = torch.Tensor(env['G'])  # (n_bus, n_bus), conductance
        B = torch.Tensor(env['B'])  # (n_bus, n_bus), susceptance
        branches = torch.LongTensor(env['branches'])  # (n_line, 2), starting and ending indices of branches
        G_line = G[branches[:, 0], branches[:, 1]]
        B_line = B[branches[:, 0], branches[:, 1]]

        P_g_max = torch.Tensor(env['P_g_max'])  # (n_gbus, ), active power generation upper bound
        P_g_min = torch.Tensor(env['P_g_min'])  # (n_gbus, ), active power generation lower bound
        Q_g_max = torch.Tensor(env['Q_g_max'])  # (n_gbus, ), reactive power generation upper bound
        Q_g_min = torch.Tensor(env['Q_g_min'])  # (n_gbus, ), reactive power generation lower bound

        V_max = torch.Tensor(env['V_max'])  # (n_bus, ), voltage upper bound
        V_min = torch.Tensor(env['V_min'])  # (n_bus, ), voltage lower bound

        I_max = torch.Tensor(env['I_max'])  # (n_line, )

        ######## ######## ######## ######## ######## ######## ######## ########
        # Constraints
        ######## ######## ######## ######## ######## ######## ######## ########

        # TODO: (P_g, Q_g) on non-generator buses MUST be zero. Should we learn it or manually enforce it?

        # Notice that `KKT_error` should be a Tensor of shape (bs, ), bs is batch size

        # 1. Reference bus
        KKT_error = torch.abs(V_i[:, sbus_idx])  # imaginary part should be zero, i.e., angle = 0

        # 2. Powerflow Equation
        # This might be further accelerated with torch sparse
        # since G and B are typically very sparse,
        # might not be necessary for small graphs

        P_inj = (V_r.T * (G @ V_r.T - B @ V_i.T) + V_i.T * (
                    G @ V_i.T + B @ V_r.T)).T  # (bs, n_bus), active power injection
        Q_inj = (V_i.T * (G @ V_r.T - B @ V_i.T) - V_r.T * (
                    G @ V_i.T + B @ V_r.T)).T  # (bs, n_bus), reactive power injection

        KKT_error += (P_inj + P_l - P_g).abs().sum(dim=1)
        KKT_error += (Q_inj + Q_l - Q_g).abs().sum(dim=1)

        # 3. Power Generation Violation
        # TODO: Defined on (P_g, Q_g) or (P_inj, Q_inj)? (P_g, Q_g) might be more direct

        # Power generation on generator buses only
        P_g_gbus = P_g[:, gbus_idx]
        Q_g_gbus = Q_g[:, gbus_idx]

        KKT_error += (torch.relu(P_g_gbus - P_g_max)).sum(dim=1)
        KKT_error += (torch.relu(P_g_min - P_g_gbus)).sum(dim=1)
        KKT_error += (torch.relu(Q_g_gbus - Q_g_max)).sum(dim=1)
        KKT_error += (torch.relu(Q_g_min - Q_g_gbus)).sum(dim=1)

        # 4. Voltage Violation
        V_mag_sq = V_r ** 2 + V_i ** 2

        KKT_error += (torch.relu(V_mag_sq - V_max ** 2)).sum(dim=1)
        KKT_error += (torch.relu(V_min ** 2 - V_mag_sq)).sum(dim=1)

        # 5. Line Flow Violation
        # TODO: In the PINN paper, line flow constraint is defined on current.
        # However, in our dataset, it seems to be defined on apparent power.

        V_r_diff = V_r[:, branches[:, 0]] - V_r[:, branches[:, 1]]  # (bs, n_line)
        V_i_diff = V_i[:, branches[:, 0]] - V_i[:, branches[:, 1]]  # (bs, n_line)

        I_r = - V_r_diff * G_line + V_i_diff * B_line
        I_i = - V_i_diff * G_line - V_r_diff * B_line

        I_mag_sq = I_r ** 2 + I_i ** 2

        KKT_error += (torch.relu(I_mag_sq - I_max ** 2)).sum(dim=1)

        return KKT_error



