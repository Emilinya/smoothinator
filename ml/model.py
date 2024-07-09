import torch
import numpy as np
from scipy.optimize import brentq
from PIL import Image, ImageFilter

from ml.perlin import perlin_noise

DEVICE = torch.device("mps")


class CNN(torch.nn.Module):
    def __init__(self, scale: int):
        super().__init__()

        self.scale = scale
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, self.scale**2, 3),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        # we must pad input in a way that "makes sense" (TODO: explain)
        x = torch.nn.functional.pad(x, (5, 5, 5, 5), mode="constant", value=0.5)
        _, _, h, w = x.shape

        def smart_average(x: torch.Tensor):
            avg_kernel = torch.tensor([1.0, 1.0, 1.0]).reshape(1, 1, -1) / 3.0
            average = torch.conv1d(x, avg_kernel)

            return torch.where(average > 0.3, 1.0, average)

        for i in range(5, 0, -1):
            ip = i + 1
            im = i - 1

            # fill sides
            x[:, 0, i - 1, ip:-ip] = smart_average(x[:, 0, i, i:-i])
            x[:, 0, h - i, ip:-ip] = smart_average(x[:, 0, h - ip, i:-i])
            x[:, 0, ip:-ip, i - 1] = smart_average(x[:, 0, i:-i, i])
            x[:, 0, ip:-ip, w - i] = smart_average(x[:, 0, i:-i, w - ip])

            # fill corners
            corners = [(im, im), (w - i, im), (w - i, h - i), (im, h - i)]
            normals = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
            for (cx, cy), (nx, ny) in zip(corners, normals):
                cxp, cyp = cx + nx, cy + ny
                x[:, 0, cy, cxp] = (x[:, 0, cyp, cxp] + x[:, 0, cy, cx + 2 * nx]) / 2
                x[:, 0, cyp, cx] = (x[:, 0, cyp, cxp] + x[:, 0, cy + 2 * ny, cx]) / 2
                x[:, 0, cy, cx] = (x[:, 0, cy, cxp] + x[:, 0, cyp, cx]) / 2

        # convolve input and shuffle pixels to get an upscaled image
        y = torch.pixel_shuffle(self.model(x), self.scale)

        return y.squeeze()

    def clean_output(self, x: torch.Tensor) -> Image.Image:
        radius = min(x.squeeze().shape) / 4
        blurred_image = tensor2image(self.forward(x)).filter(
            ImageFilter.GaussianBlur(radius)
        )

        # ensure volume constraint is followed by finding optimal threshold
        volume_fraction = float(torch.mean(x))

        def volume_error(threshold: float):
            thresholded_image = blurred_image.point(
                lambda p: 255 if p > threshold else 0
            )
            return np.mean(np.array(thresholded_image) / 255.0) - volume_fraction

        optimal_threshold = brentq(volume_error, 0, 255)
        assert isinstance(optimal_threshold, float)
        image = blurred_image.point(lambda p: 255 if p > optimal_threshold else 0)

        return image


def tensor2image(tensor: torch.Tensor):
    return Image.fromarray((tensor.squeeze().numpy(force=True) * 255).astype(np.uint8))


def generate_training_data(
    shape: tuple[int, int], scale: int, batches: int, noise_scale: int
):
    large_shape = (shape[0] * scale, shape[1] * scale)
    res = (-1, -1)
    for i in range(noise_scale, max(large_shape) + 1):
        if res[0] == -1 and large_shape[0] % i == 0:
            res = (i, res[1])
        if res[1] == -1 and large_shape[1] % i == 0:
            res = (res[0], i)
        if min(res) > -1:
            break

    batched_image_tensor = torch.zeros((batches, *large_shape))
    batched_small_image_tensor = torch.zeros((batches, *shape))
    for i in range(batches):
        noise = perlin_noise((shape[0] * scale, shape[1] * scale), res)
        image = np.where(
            noise > 0, np.full_like(noise, 255), np.full_like(noise, 0)
        ).astype(np.uint8)

        small_image = Image.fromarray(image)
        small_image.thumbnail(
            (shape[1], shape[0]), Image.Resampling.NEAREST, reducing_gap=None
        )

        batched_image_tensor[i] = torch.from_numpy(image / 255.0).float()
        batched_small_image_tensor[i] = torch.from_numpy(
            np.array(small_image) / 255.0
        ).float()

    return batched_image_tensor.to(DEVICE), batched_small_image_tensor.to(DEVICE)


def train_network(size: tuple[int, int], scale: int):
    network = CNN(scale).to(DEVICE)
    optimizer_adam = torch.optim.Adam(network.parameters(), lr=0.001)

    epochs = 1000
    batches = 5
    min_loss = float("Infinity")
    for i in range(epochs):
        image, small_image = generate_training_data(
            (size[1], size[0]), scale, batches, 2
        )
        image_prediction = network.forward(small_image)
        loss = torch.sum((image - image_prediction) ** 2) / np.prod(image.shape)

        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
        loss = float(loss)

        if loss < min_loss:
            min_loss = loss

        print(
            f"\r{i+1}/{epochs}: loss: {float(loss):.3g}, min: {min_loss:.3g}{' '*10}",
            end="",
        )
    print()

    torch.save(network.state_dict(), "ml/state_dict.pt")

    for i in range(2):
        image_tensor, small_image_tensor = generate_training_data(
            (size[1], size[0]), scale, 1, 2
        )
        upscaled_image_tensor = network.forward(small_image_tensor)

        image = tensor2image(image_tensor)
        small_image = tensor2image(small_image_tensor)
        upscaled_image = tensor2image(upscaled_image_tensor)
        clean_upscaled_image = network.clean_output(small_image_tensor)

        image.save(f"ml/ai_out/{i}_image.png")
        small_image.save(f"ml/ai_out/{i}_small_image.png")
        upscaled_image.save(f"ml/ai_out/{i}_upscaled_image.png")
        clean_upscaled_image.save(f"ml/ai_out/{i}_zlean_upscaled_image.png")


def plot_test():
    image_tensor, small_image_tensor = generate_training_data((40, 80), 30, 1, 2)
    image = Image.fromarray(
        (image_tensor.squeeze().numpy(force=True) * 255).astype(np.uint8)
    )
    small_image = Image.fromarray(
        (small_image_tensor.squeeze().numpy(force=True) * 255).astype(np.uint8)
    )

    image.save("test.png")
    small_image.save("test_small.png")


def main():
    train_network((80, 40), 30)


if __name__ == "__main__":
    main()
