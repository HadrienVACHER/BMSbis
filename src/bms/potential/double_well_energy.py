import torch

from bms.potential.base import BasePotential


def compute_distances(x, n_particles, spatial_dim):
    x = x.reshape(-1, n_particles, spatial_dim)
    diff = x[:, :, None, :] - x[:, None, :, :]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    n = n_particles
    tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
    return dist[:, tri]


class DoubleWellPotential(BasePotential):
    def __init__(
        self,
        n_particles: int = 4,
        spatial_dim: int = 2,
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.a = a
        self.b = b
        self.c = c
        self.offset = offset

    def energy(self, pos: torch.Tensor) -> torch.Tensor:
        d = compute_distances(pos, self.n_particles, self.spatial_dim) - self.offset
        return (self.a * d**4 + self.b * d**2 + self.c).sum(dim=-1)

    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        # terminal_cost.potential_grad is @torch.no_grad(); re-enable for autograd.
        with torch.enable_grad():
            pos = pos.reshape(-1, self.n_particles, self.spatial_dim)
            pos = pos.detach().requires_grad_(True)
            energy = self.energy(pos)
            forces = -torch.autograd.grad(energy.sum(), pos, create_graph=False)[0]
        return {"energy": energy.detach(), "forces": forces.detach()}