"""Permutation- and rigid-alignment-aware distances for n-body evaluation."""

import torch
from scipy.optimize import linear_sum_assignment


def find_rigid_alignment(A, B):
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    R = V.mm(U.T)
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    return x1[col_ind]


def dist_point_clouds(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            reordered.append(ot(x0[i], x1[j]))
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(
            x0[i][None].repeat(len(x1), 1, 1), reordered
        )
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    return torch.stack(M).squeeze()


def interatomic_dist(x, n_particles, n_spatial_dim):
    B, D = x.shape
    assert D == n_particles * n_spatial_dim
    x = x.view(B, n_particles, n_spatial_dim)
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles), device=x.device), diagonal=1)
        == 1,
    ]
    return torch.linalg.norm(distances, dim=-1)
