"""W2 metrics for identical-particle systems (DW-4, LJ-13, LJ-55)."""

from pathlib import Path

import numpy as np
import ot as pot
import torch

from bms.eval.eval_utils import dist_point_clouds, interatomic_dist
from bms.utils.geometry import subtract_mean


class NBodyEvaluator:
    def __init__(self, ref_samples_path: str, energy):
        self.energy = energy
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.spatial_dim
        path = Path(ref_samples_path)
        if not path.is_absolute():
            # Resolve relative to the package repo root (three levels above eval/).
            repo_root = Path(__file__).resolve().parents[3]
            path = repo_root / path
        ref = torch.as_tensor(np.load(path, allow_pickle=True), dtype=torch.float)
        ref = ref.reshape(-1, self.n_particles, self.n_spatial_dim)
        self.ref_samples = subtract_mean(ref)

    def __call__(self, samples: torch.Tensor) -> dict:
        samples = subtract_mean(samples.float())
        B = samples.shape[0]
        idx = torch.randperm(len(self.ref_samples))[:B]
        ref = self.ref_samples[idx].to(samples.device)

        gen_e = self.energy.energy(samples).detach()
        ref_e = self.energy.energy(ref).detach()
        energy_w2 = float(pot.emd2_1d(ref_e.cpu().numpy(), gen_e.cpu().numpy()) ** 0.5)

        gen_flat = samples.reshape(B, -1)
        ref_flat = ref.reshape(B, -1)
        gen_d = interatomic_dist(gen_flat, self.n_particles, self.n_spatial_dim)
        ref_d = interatomic_dist(ref_flat, self.n_particles, self.n_spatial_dim)
        dist_w2 = float(
            pot.emd2_1d(
                gen_d.cpu().numpy().reshape(-1), ref_d.cpu().numpy().reshape(-1)
            )
            ** 0.5
        )

        M = dist_point_clouds(samples.cpu(), ref.cpu())
        a = torch.ones(M.shape[0]) / M.shape[0]
        b = torch.ones(M.shape[0]) / M.shape[0]
        eq_w2 = float(pot.emd2(M=M**2, a=a, b=b) ** 0.5)
        return {"energy_w2": energy_w2, "eq_w2": eq_w2, "dist_w2": dist_w2}
