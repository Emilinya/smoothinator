import argparse

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError


def load(filename: str) -> npt.NDArray[np.uint8]:
    try:
        image = Image.open(filename).convert("L")
        return np.array(image)
    except UnidentifiedImageError:
        image = np.load(filename)[::-1, :]
        minv, maxv = np.min(image), np.max(image)
        return (((maxv - image) / (maxv - minv)) * 255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_file",
        type=argparse.FileType("r"),
        help="path to image/data you want to smooth",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        required=False,
        default=30,
        help="how many times larger the output will be compared to the input. (Default: 30)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        choices=("ml", "bezier"),
        default="ml",
        help="wether the input is upscaled using a neural network or curve fitting. "
        + "The neural network works much better, so it is the default",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="make the program create figures depicting intermediate results",
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

    args = parser.parse_args()
    filename = args.data_file.name
    args.data_file.close()

    image = load(filename)

    if args.mode == "bezier":
        raise ValueError(":(")
    else:
        from ml.smooth import main

        main(image, filename)
