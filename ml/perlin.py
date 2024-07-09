import numpy as np
import numpy.typing as npt


def interpolant(t: npt.NDArray):
    return t**3 * (t * (t * 6 - 15) + 10)


def perlin_noise(
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False),
) -> npt.NDArray:
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    # Ramps
    grid0, grid1 = grid[:, :, 0], grid[:, :, 1]
    n00 = np.sum(np.dstack((grid0, grid1)) * g00, 2)
    n10 = np.sum(np.dstack((grid0 - 1, grid1)) * g10, 2)
    n01 = np.sum(np.dstack((grid0, grid1 - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid0 - 1, grid1 - 1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    t0, t1 = t[:, :, 0], t[:, :, 1]
    n0 = n00 * (1 - t0) + t0 * n10
    n1 = n01 * (1 - t0) + t0 * n11

    return np.sqrt(2) * ((1 - t1) * n0 + t1 * n1)
