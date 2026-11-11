import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (channels * channels)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, C, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=32, layers=4):
        super().__init__()
        self.lift = nn.Conv1d(2, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, 1, 1)

    def forward(self, u):
        B, _, N = u.shape
        grid = torch.linspace(0, 1, N, device=u.device).view(1, 1, N).expand(B, 1, N)
        h = self.lift(torch.cat([u, grid], dim=1))
        for i, (s, l) in enumerate(zip(self.spec, self.local)):
            h = s(h) + l(h)
            if i < len(self.spec) - 1:
                h = F.gelu(h)
        return self.proj2(F.gelu(self.proj1(h)))