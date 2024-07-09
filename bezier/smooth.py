import queue
import argparse

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.signal import convolve2d
from PIL import Image, UnidentifiedImageError

from fitCurves.fit_curves import fit_curve


def load(filename: str) -> npt.NDArray[np.uint8]:
    try:
        image = Image.open(filename).convert("L")
        return np.array(image)
    except UnidentifiedImageError:
        image = np.load(filename)[::-1, :]
        minv, maxv = np.min(image), np.max(image)
        return (((maxv - image) / (maxv - minv)) * 255).astype(np.uint8)


def save(array: npt.NDArray[np.uint8], filename: str) -> None:
    image = Image.fromarray(array)
    image.save(filename)


def project(array: npt.NDArray, minv=0, maxv=255, avg: None | float = None):
    if avg is None:
        avg = (maxv - minv) / 2
    return np.where(array > avg, np.full_like(array, maxv), np.full_like(array, minv))


def detect_edges(array: npt.NDArray[np.uint8]) -> list[list[tuple[int, int]]]:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ]
    )

    convolution = project(
        -convolve2d(array, kernel, mode="same", fillvalue=255), 255, 0, 0
    ).astype(np.uint8)

    def get_neighbors(point: tuple[int, int]):
        return [
            (point[0] - 1, point[1]),
            (point[0] + 1, point[1]),
            (point[0], point[1] - 1),
            (point[0], point[1] + 1),
            (point[0] - 1, point[1] - 1),
            (point[0] - 1, point[1] + 1),
            (point[0] + 1, point[1] + 1),
            (point[0] + 1, point[1] - 1),
        ]

    edge_points = {
        (int(v[1]), int(v[0])) for v in np.array(np.where(convolution == 0)).T
    }
    boundary_points_list: list[list[tuple[int, int]]] = []
    while len(edge_points) > 0:
        point = edge_points.pop()
        boundary_points = [point]

        search_points: queue.SimpleQueue[tuple[int, int]] = queue.SimpleQueue()
        for neighbor in get_neighbors(point):
            if neighbor in edge_points:
                search_points.put(neighbor)
                edge_points.remove(neighbor)
                break

        while not search_points.empty():
            point = search_points.get()
            boundary_points.append(point)

            for neighbor in get_neighbors(point):
                if neighbor in edge_points:
                    search_points.put(neighbor)
                    edge_points.remove(neighbor)
        boundary_points_list.append(boundary_points)

    return boundary_points_list


def lerp(a, b, t: float):
    return a + (b - a) * t


def cubic(curve_points: npt.NDArray, t) -> npt.NDArray:
    if curve_points.shape != (4, 2):
        raise ValueError("Curve points must be four 2d points!")

    p1, p2, p3, p4 = curve_points
    d1l1 = lerp(p1, p2, t)
    d1l2 = lerp(p2, p3, t)
    d1l3 = lerp(p3, p4, t)

    d2l1 = lerp(d1l1, d1l2, t)
    d2l2 = lerp(d1l2, d1l3, t)

    return lerp(d2l1, d2l2, t)


def bezier_test_plot(
    image: npt.NDArray[np.uint8],
    boundary_arrays: list[npt.NDArray],
    beziers_list: list[npt.NDArray],
):
    plt.imshow(image, cmap="gray")

    cmap = colormaps.get_cmap("hsv")
    t_ray = np.linspace(0.0, 1.0, 100)
    for boundary_array, beziers in zip(boundary_arrays, beziers_list):
        for j, bezier in enumerate(beziers):
            curve = np.zeros((2, t_ray.size))
            for i, t in enumerate(t_ray):
                curve[:, i] = cubic(bezier, t)

            colors = ["r", "g", "b", "c", "m", "y"]
            plt.plot(*curve, color=colors[j % len(colors)])

        for i, point in enumerate(boundary_array):
            plt.plot(*point, ".", color=cmap(i / (len(boundary_array) - 1)))

    plt.savefig("beziers.png", dpi=200, bbox_inches="tight")


def bezier_collision(curve_points: npt.NDArray, y: float, debug=False) -> list[float]:
    if curve_points.shape != (4, 2):
        raise ValueError("Curve points must be four 2d points!")

    # We are interested in collisions at fixed y: only need y-axis of the curve
    p1, p2, p3, p4 = curve_points
    y1, y2, y3, y4 = p1[1], p2[1], p3[1], p4[1]

    # if all bezier points are above/below y, it can't intersect
    if (y1 > y and y2 > y and y3 > y and y4 > y) or (
        y1 < y and y2 < y and y3 < y and y4 < y
    ):
        return []

    # cubic bezier curve is a*t^3 + b*t^2 + c*t + d, where
    a = -(y1 - 3 * y2 + 3 * y3 - y4)
    b = 3 * (y1 - 2 * y2 + y3)
    c = -3 * (y1 - y2)
    d = y1
    if debug:
        print(a, b, c, d)

    # want to find intersection with y => subtract y and find root
    d -= y
    roots: list[float] = []

    # see if any coefficients are zero, and if so, find simpler roots
    if a == 0:
        if b == 0:
            if c == 0:
                if d == 0:
                    # all coefficients are zero, any point is a root
                    roots.append(0.5)
                # if d!= 0, no solution exists
            else:
                # c*t + d = 0 => t = -d/c
                roots.append(-d / c)
        else:
            # use quadratic formula
            discriminant = c**2 - 4 * b * d
            if discriminant < -1e-6:
                # negative discriminant => both solutions are complex
                return []
            if abs(discriminant) <= 1e-6:
                roots.append(-c / (2 * b))
            else:
                sqrt = abs(discriminant) ** (1 / 2)
                roots.append((-c + sqrt) / (2 * b))
                roots.append((-c - sqrt) / (2 * b))
    else:
        # equation for roots of cubic polynomials
        delta_0 = b**2 - 3 * a * c
        delta_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

        if debug:
            print(delta_0, delta_1)

        roots: list[float] = []
        if abs(delta_0) < 1e-16 and abs(delta_1) < 1e-16:
            roots.append(-b / (3 * a))
        else:
            unity = (-1 + float(np.sqrt(3)) * 1j) / 2
            C: complex = (
                (delta_1 + (delta_1**2 - 4 * delta_0**3 + 0j) ** (1 / 2)) / 2
            ) ** (1 / 3)

            for k in [0, 1, 2]:
                complex_root: complex = (
                    -1 / (3 * a) * (b + unity**k * C + delta_0 / (unity**k * C))
                )
                if abs(complex_root.imag) < 1e-6:
                    roots.append(complex_root.real)

    return [cubic(curve_points, root)[0] for root in roots if -1e-6 <= root <= 1 + 1e-6]


def rasterize_beziers(
    beziers_list: list[npt.NDArray], image_shape: tuple[int, int], scale: int
):
    h, w = image_shape
    smooth_image = np.empty((h * scale, w * scale), dtype=np.uint8)
    problematic_collisions: list[int] = []
    for y in range(h * scale):
        collisions: list[float] = []
        for beziers in beziers_list:
            for bezier in beziers:
                collisions += bezier_collision(bezier, y / scale - 0.5)

        if len(collisions) == 0:
            smooth_image[y, :] = 255
            continue

        collisions = sorted(
            [int(round(v)) for v in {scale * (round(v, 4) + 0.5) for v in collisions}]
        )
        if len(collisions) % 2 != 0:
            problematic_collisions.append(y)

        positive_collisions = [c for c in collisions if c >= 0]
        negative_count = len(collisions) - len(positive_collisions)
        sign = 1 if negative_count % 2 == 0 else 0

        smooth_image[y, : positive_collisions[0]] = 255 * sign
        sign = 1 if sign == 0 else 0
        for i in range(len(positive_collisions) - 1):
            smooth_image[y, positive_collisions[i] : positive_collisions[i + 1]] = 255 * sign
            sign = 1 if sign == 0 else 0
        smooth_image[y, positive_collisions[-1] :] = 255 * sign

    for y in problematic_collisions:
        print(f"##### {y} #####")
        all_collisions = []
        for beziers in beziers_list:
            for bezier in beziers:
                collisions = bezier_collision(bezier, y / scale - 0.5, debug=True)
                if len(collisions) > 0:
                    print()
                all_collisions += collisions
        print(all_collisions)
        print(
            sorted(
                [
                    int(round(v))
                    for v in {scale * (round(v, 4) + 0.5) for v in all_collisions}
                ]
            )
        )

    return smooth_image


def main(filename: str, max_error: float, scale: int, debug: bool):
    image = load(filename)
    print(np.mean(image) / 255.0)

    projection = project(image)
    if debug:
        save(projection, "projected.png")

    # get edge points
    boundary_points_list = detect_edges(projection)
    if debug:
        edge_figure = np.full((*image.shape, 3), 255, dtype=np.uint8)
        for i, boundary_points in enumerate(boundary_points_list):
            colors = [
                [255, 0, 0],
                [255, 255, 0],
                [0, 255, 0],
                [0, 255, 255],
                [0, 0, 255],
                [255, 0, 255],
            ]
            color = colors[i % len(colors)]
            for ix, iy in boundary_points:
                edge_figure[iy, ix, :] = color
        save(edge_figure, "edges.png")

    # Sort edge points by angle and find beziers using fitCurves program
    boundary_arrays: list[npt.NDArray] = []
    beziers_list: list[npt.NDArray] = []
    for boundary_points in boundary_points_list:
        points = np.array(boundary_points).astype(np.float64)

        tangent_direction: int | None = None
        for i, point in enumerate(points):
            pre = points[(i - 1) % len(points)]
            nex = points[(i + 1) % len(points)]
            between = nex - pre
            tangent = np.array([between[1], -between[0]]) / np.linalg.norm(between)
            if tangent_direction is None:
                index_tangent = np.sign(tangent)

                test_x = int(point[0] + index_tangent[0])
                test_y = int(point[1] + index_tangent[1])
                if projection[test_y, test_x] == 0:
                    tangent_direction = 1
                else:
                    tangent_direction = -1

            point += tangent / (2 * np.sqrt(2)) * (-1) * tangent_direction
        points = np.append(points, [points[0]], axis=0)

        boundary_arrays.append(points)
        beziers_list.append(np.array(fit_curve(points, max_error)))
    if debug:
        bezier_test_plot(image, boundary_arrays, beziers_list)

    # rasterize bezier curves to create high resolution image
    w, h = image.shape
    smooth_image = rasterize_beziers(beziers_list, (w, h), scale)
    print(np.mean(smooth_image) / 255.0)
    save(smooth_image, "smooth.png")


def parse_arguments() -> tuple[str, float, int, bool]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_file",
        type=argparse.FileType("r"),
        help="path to image/data you want to smooth",
    )
    parser.add_argument(
        "-e",
        "--max_error",
        type=float,
        required=False,
        default=0.2,
        help="the maximum error for the bezier curve approximation. A larger value will give a "
        + "smoother result, but will conform to the original shape less. (Default: 0.2)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        required=False,
        default=20,
        help="how many times larger the output will be compared to the input. (Default: 20)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="make the program create figures depicting intermediate results",
    )

    args = parser.parse_args()
    filename = args.data_file.name
    args.data_file.close()

    return filename, args.max_error, args.scale, args.debug


if __name__ == "__main__":
    main(*parse_arguments())
