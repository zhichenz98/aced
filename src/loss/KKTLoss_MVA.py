"""
Adapted from
https://github.com/RahulNellikkath/Physics-Informed-Neural-Networks-for-AC-Optimal-Power-Flow/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class KKTLoss(nn.Module):
    """
    The current version assume that the topology is the same for all samples in the same batch.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Volt: torch.Tensor,  # voltage prediction
        PQ_Gens: torch.Tensor,  # power generation prediction
        PQ_Loads: torch.Tensor,  # power load, ground truth
        # Dual variables
        n_o_l_p,
        n_o_mu_g_u: torch.Tensor,  # power generation upper bound
        n_o_mu_g_d: torch.Tensor,  # power generation lower bound
        n_o_mu_v_u: torch.Tensor,  # voltage upper bound
        n_o_mu_v_d: torch.Tensor,  # voltage lower bound
        n_o_mu_i_u: torch.Tensor,  # line flow upper bound
        # Topology info
        # load_bus_idxs: list = None,
        # slack_bus_idx: int = 1,
        # generator_bus_idxs: list = None,
        # All the non-differentiable things
        env: dict = None,
        # computation config
        foreach: bool = True,  # whether to use foreach computation in the powerflow
    ):
        """
        It will be much more clear to split Volt and PQ...
        """

        batch_size = Volt.shape[0]

        device = Volt.device
        dtype = Volt.dtype

        ######## ######## ######## ######## ######## ######## ######## ########
        # Configs in environment
        ######## ######## ######## ######## ######## ######## ######## ########

        n_bus = env["n_bus"]  # int, Number of buses (all types)
        n_gbus = env["n_gbus"]  # int, Number of generator
        n_lbus = env["n_lbus"]  # int, Number of loads

        slack_bus_idx = env["slack_bus_idx"]  # int, Index of the slack bus
        gbus_idx = torch.LongTensor(env["gbus_idx"])
        gbus_idx = torch.cat([gbus_idx, gbus_idx + n_bus], dim=0)  #

        n_line = env["n_line"]  # int, Number of lines

        G = env["G"]
        B = env["B"]

        Y = torch.vstack([torch.hstack([G, -B]), torch.hstack([B, G])])

        Yconj = torch.vstack([torch.hstack([-B, G]), torch.hstack([G, -B])])

        Gen_max = env["Gen_max"]  # torch.Tensor, power generation upper bound
        Gen_min = env["Gen_min"]  # power generation lower bound

        V_max = env["V_max"]  # voltage upper bound
        V_min = env["V_min"]  # voltage lower bound

        # TODO: Unsure about things below

        Ybr = env["Ybr"]
        IM = env["IM"]  # torch.Tensor, (2 * n_line, n_bus), start and end of each line
        L_limit = env["L_limit"]  # torch.Tensor, (n_line, )

        C_Pg = env["C_Pg"]
        C_Qg = env["C_Qg"]
        Lg_Max = env["Lg_Max"]
        Map_g = env["Map_g"]
        Map_L = env["Map_L"]

        ######## ######## ######## ######## ######## ######## ######## ########
        # Constraints
        ######## ######## ######## ######## ######## ######## ######## ########

        # 1. Reference bus
        KKT_error = torch.abs(Volt[:, slack_bus_idx])  # [batch_size, ]

        # 2. PowerFlow Equation,
        # for i in range(n_bus):
        #     M = torch.zeros((2 * n_bus, 2 * n_bus), dtype=dtype, device=device)
        #     M[i, i] = 1
        #     M[n_bus + i, n_bus + 1] = 1
        #
        #     H_p = M @ Y
        #     H_q = M @ Yconj
        #
        #     vHv_p = torch.einsum('bi,ij,bj->b', Volt, H_p, Volt)
        #     vHv_q = torch.einsum('bi,ij,bj->b', Volt, H_q, Volt)
        #
        #     # Extract real/reactive power at bus i
        #     e_P = torch.zeros((2 * n_bus, 1), dtype=dtype, device=device)
        #     e_Q = torch.zeros((2 * n_bus, 1), dtype=dtype, device=device)
        #     e_P[i, 0] = 1
        #     e_Q[n_bus + i, 0] = 1
        #
        #     p_n = (PQ_Loads - PQ_Gens) @ e_P  # [batch_size, 1]
        #     q_n = (PQ_Loads - PQ_Gens) @ e_Q
        #
        #     # Fixed: add a torch.abs
        #     KKT_error += torch.abs(vHv_p + p_n.squeeze(1))
        #     KKT_error += torch.abs(vHv_q + q_n.squeeze(1))

        # A faster version for 2:
        KKT_error += (
            (
                (Volt.T * (Y @ Volt.T)).T.view(-1, 2, n_bus).sum(dim=1)
                + (PQ_Loads - PQ_Gens)[:, :n_bus]
            )
            .abs()
            .sum(1)
        )
        KKT_error += (
            (
                (Volt.T * (Yconj @ Volt.T)).T.view(-1, 2, n_bus).sum(dim=1)
                + (PQ_Loads - PQ_Gens)[:, n_bus:]
            )
            .abs()
            .sum(1)
        )

        # 3. Generation Violation
        KKT_error += torch.sum(torch.relu(PQ_Gens[:, gbus_idx] - Gen_max), dim=1)
        KKT_error += torch.sum(torch.relu(Gen_min - PQ_Gens[:, gbus_idx]), dim=1)

        # 4. Voltage
        V_r = Volt[:, :n_bus]
        V_i = Volt[:, n_bus : 2 * n_bus]
        V_mag_sq = V_r**2 + V_i**2

        KKT_error += torch.sum(torch.relu(V_mag_sq - V_max**2), dim=1)
        KKT_error += torch.sum(torch.relu(V_min**2 - V_mag_sq), dim=1)

        # 5. Line Flow Violation
        # TODO: figure out how Ybr is computed
        # Ibr = (Ybr @ (IM @ Volt.T)).T  # shape: [batch_size, 2 * n_line]
        # I_r = Ibr[:, :n_line]
        # I_i = Ibr[:, n_line:]
        # I_mag_sq = I_r**2 + I_i**2

        # KKT_error += torch.sum(torch.relu(I_mag_sq - L_limit**2), dim=1)

        Ibr = (Ybr @ (IM @ Volt.T)).T  # shape: [B, 2 * n_line]
        I_r = Ibr[:, :n_line]  # [B, n_line]
        I_i = Ibr[:, n_line:]  # [B, n_line]
        I_mag_sq = I_r**2 + I_i**2  # |I|^2 for each branch

        V_r = Volt[:, :n_bus]  # [B, n_bus]
        V_i = Volt[:, n_bus:]  # [B, n_bus]
        V_mag_sq = V_r**2 + V_i**2  # [B, n_bus]

        IM_start = IM[:n_line, :]  # from-bus selector
        IM_end = IM[n_line:, :]  # to-bus selector

        V_from_mag_sq = (IM_start @ V_mag_sq.T).T  # [B, n_line]
        V_to_mag_sq = (IM_end @ V_mag_sq.T).T  # [B, n_line]

        S_from_sq = V_from_mag_sq * I_mag_sq  # [B, n_line]
        S_to_sq = V_to_mag_sq * I_mag_sq  # [B, n_line]

        KKT_error += torch.sum(torch.relu(S_from_sq - L_limit**2), dim=1)
        KKT_error += torch.sum(torch.relu(S_to_sq - L_limit**2), dim=1)

        ######## ######## ######## ######## ######## ######## ######## ########
        # KKT Conditions
        ######## ######## ######## ######## ######## ######## ######## ########

        # Comment out this section.
        # TODO: Need re-checking the formulas and

        # # Generation Violation (upper/lower bound multipliers)
        # KKT_error += torch.sum(torch.abs(n_o_mu_g_u * (PQ_Gens - Gen_max)), dim=1) / n_gbus
        # KKT_error += torch.sum(torch.abs(n_o_mu_g_d * (Gen_min - PQ_Gens)), dim=1) / n_gbus
        #
        # # Voltage Violation
        # KKT_error += torch.sum(torch.abs(n_o_mu_v_u * (V_mag_sq - V_max ** 2)), dim=1)
        # KKT_error += torch.sum(torch.abs(n_o_mu_v_d * (V_min ** 2 - V_mag_sq)), dim=1)
        #
        # # Line Flow Violation
        # KKT_error += torch.sum(torch.abs(n_o_mu_i_u * (I_mag_sq - L_limit ** 2)), dim=1)
        #
        # # dL/ dP_Gen
        # # TODO: Do not understand this...
        # dual_term = (
        #         n_o_mu_g_u * Lg_Max[1] -
        #         n_o_mu_g_d * Lg_Max[2] +
        #         (n_o_l_p * Lg_Max[0]) @ Map_g.T -
        #         torch.cat((C_Pg, C_Qg), dim=1)  # cost coefficient
        # )
        # KKT_error += torch.sum(torch.abs(dual_term), dim=1)
        #
        # # KKT dual variables, Dual feasibility (all mu >= 0)
        # for mu in [n_o_mu_g_u, n_o_mu_g_d, n_o_mu_v_u, n_o_mu_v_d, n_o_mu_i_u]:
        #     KKT_error += torch.sum(torch.relu(-mu), dim=1)

        return KKT_error
