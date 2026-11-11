import numpy as np

def solve_burgers(u0, t_end=0.5, nu_eff=0.01/np.pi, c=1.0, dt=2.5e-4):
    u0 = np.asarray(u0, dtype=np.float64)
    N = u0.size
    k = 2 * np.pi * np.arange(N // 2 + 1)
    mask = (np.arange(N // 2 + 1) <= N // 3)

    def nonlin(vh):
        u = np.fft.irfft(vh, n=N)
        return -c * 0.5j * k * np.fft.rfft(u * u) * mask

    steps = int(round(t_end / dt))
    dt = t_end / steps
    E = np.exp(-nu_eff * k**2 * dt / 2)
    E2 = E * E
    v = np.fft.rfft(u0) * mask
    for _ in range(steps):
        a = dt * nonlin(v)
        b = dt * nonlin(E * (v + a / 2))
        cc = dt * nonlin(E * v + b / 2)
        d = dt * nonlin(E2 * v + E * cc)
        v = E2 * v + (E2 * a + 2 * E * (b + cc) + d) / 6
    return np.fft.irfft(v, n=N)