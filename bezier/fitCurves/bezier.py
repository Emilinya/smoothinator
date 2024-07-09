import numpy as np
import numpy.typing as npt


def q(bezier: list[npt.NDArray[np.float64]], t: float):
    """evaluates cubic bezier at t, return point"""
    return (
        (1.0 - t) ** 3 * bezier[0]
        + 3 * (1.0 - t) ** 2 * t * bezier[1]
        + 3 * (1.0 - t) * t**2 * bezier[2]
        + t**3 * bezier[3]
    )


def q_prime(bezier: list[npt.NDArray[np.float64]], t: float):
    """evaluates cubic bezier first derivative at t, return point"""
    return (
        3 * (1.0 - t) ** 2 * (bezier[1] - bezier[0])
        + 6 * (1.0 - t) * t * (bezier[2] - bezier[1])
        + 3 * t**2 * (bezier[3] - bezier[2])
    )


def q_prime_prime(bezier: list[npt.NDArray[np.float64]], t: float):
    """evaluates cubic bezier second derivative at t, return point"""
    return 6 * (1.0 - t) * (bezier[2] - 2 * bezier[1] + bezier[0]) + 6 * (t) * (
        bezier[3] - 2 * bezier[2] + bezier[1]
    )
