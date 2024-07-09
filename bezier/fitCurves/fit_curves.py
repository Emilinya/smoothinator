"""
Python implementation of Algorithm for Automatically Fitting Digitized
Curves by Philip J. Schneider "Graphics Gems", Academic Press, 1990
"""

import numpy as np
import numpy.typing as npt
from fitCurves.bezier import q, q_prime, q_prime_prime


def fit_curve(
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64], max_error: float
):
    """Fit one (ore more) Bezier curves to a set of points"""

    if not points[0].shape == (2,):
        raise ValueError("'points' must be a list of 2d points")

    left_tangent = normalize(points[1] - points[0])
    right_tangent = normalize(points[-2] - points[-1])
    return fit_cubic(points, left_tangent, right_tangent, max_error)


def fit_cubic(
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    left_tangent: npt.NDArray[np.float64],
    right_tangent: npt.NDArray[np.float64],
    error: float,
):
    # Use heuristic if region only has two points in it
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.0
        bezier_curve: list[npt.NDArray[np.float64]] = [
            points[0],
            points[0] + left_tangent * dist,
            points[1] + right_tangent * dist,
            points[1],
        ]
        return [bezier_curve]

    # Parameterize points, and attempt to fit curve
    u = chord_length_parameterize(points)
    bezier_curve = generate_bezier(points, u, left_tangent, right_tangent)
    # Find max deviation of points to fitted curve
    max_error, split_point = compute_max_error(points, bezier_curve, u)
    if max_error < error:
        return [bezier_curve]

    # If error not too large, try some reparameterization and iteration
    if max_error < error**2:
        for _ in range(20):
            u_prime = reparameterize(bezier_curve, points, u)
            bezier_curve = generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_error, split_point = compute_max_error(points, bezier_curve, u_prime)
            if max_error < error:
                return [bezier_curve]
            u = u_prime

    # Fitting failed -- split at max error point and fit recursively
    beziers: list[list[npt.NDArray[np.float64]]] = []
    center_tangent = normalize(points[split_point - 1] - points[split_point + 1])
    # for some reason, if I use +=, the type checker thinks
    # fit_cubic can return Unknown, so I have to use append
    for bezier in fit_cubic(
        points[: split_point + 1], left_tangent, center_tangent, error
    ):
        beziers.append(bezier)
    for bezier in fit_cubic(
        points[split_point:], -center_tangent, right_tangent, error
    ):
        beziers.append(bezier)

    return beziers


def generate_bezier(
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    parameters: list[float],
    left_tangent: npt.NDArray[np.float64],
    right_tangent: npt.NDArray[np.float64],
):
    bezier_curve: list[npt.NDArray[np.float64]] = [
        points[0],
        np.array([]),
        np.array([]),
        points[-1],
    ]

    # compute the A's
    A = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = left_tangent * 3 * (1 - u) ** 2 * u
        A[i][1] = right_tangent * 3 * (1 - u) * u**2

    # Create the C and X matrices
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)

        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent newton_raphson_root_find() call. */
    seg_length = np.linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * seg_length
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezier_curve[1] = bezier_curve[0] + left_tangent * (seg_length / 3.0)
        bezier_curve[2] = bezier_curve[3] + right_tangent * (seg_length / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezier_curve[1] = bezier_curve[0] + left_tangent * alpha_l
        bezier_curve[2] = bezier_curve[3] + right_tangent * alpha_r

    return bezier_curve


def reparameterize(
    bezier: list[npt.NDArray[np.float64]],
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    parameters: list[float],
):
    return [
        newton_raphson_root_find(bezier, point, u)
        for point, u in zip(points, parameters)
    ]


def newton_raphson_root_find(
    bezier: list[npt.NDArray[np.float64]], point: npt.NDArray[np.float64], u: float
):
    """
    Newton's root finding algorithm calculates f(x)=0 by reiterating
    x_n+1 = x_n - f(x_n)/f'(x_n)

    We are trying to find curve parameter u for some point p that minimizes
    the distance from that point to the curve. Distance point to curve is d=q(u)-p.
    At minimum distance the point is perpendicular to the curve.
    We are solving
    f = q(u)-p * q'(u) = 0
    with
    f' = q'(u) * q'(u) + q(u)-p * q''(u)

    gives
    u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """
    d = q(bezier, u) - point
    numerator = (d * q_prime(bezier, u)).sum()
    denominator = (q_prime(bezier, u) ** 2 + d * q_prime_prime(bezier, u)).sum()

    if denominator == 0.0:
        return u
    return u - float(numerator / denominator)


def chord_length_parameterize(
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + float(np.linalg.norm(points[i] - points[i - 1])))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]

    return u


def compute_max_error(
    points: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    bez: list[npt.NDArray[np.float64]],
    parameters: list[float],
):
    max_dist = 0.0
    # This was previously len(points) / 2, which means
    # split_point could be a float. Was this intentional?
    split_point = len(points) // 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = float(np.linalg.norm(q(bez, u) - point) ** 2)
        if dist > max_dist:
            max_dist = dist
            split_point = i

    return max_dist, split_point


def normalize(v: npt.NDArray[np.float64]):
    return v / np.linalg.norm(v)
