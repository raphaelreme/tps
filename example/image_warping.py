import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import scipy.ndimage  # type: ignore[import-untyped]
from PIL import Image, ImageDraw  # type: ignore[import-untyped]

from tps import ThinPlateSpline


def increase_ctrl_points():
    """Generate ctrl points that increase the center of the image.

    (In proportion of the desired shapes)
    """
    input_ctrl = np.array(
        [
            [0.25, 0.25],  # (i_0, j_0)
            [0.25, 0.75],  # (i_0, j_1)
            [0.75, 0.25],  # (i_1, j_0)
            [0.75, 0.75],  # (i_1, j_1)
        ]
    )

    output_ctrl = np.array(
        [
            [0.15, 0.15],
            [0.15, 0.85],
            [0.85, 0.15],
            [0.85, 0.85],
        ]
    )

    corners = np.array(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    return np.concatenate((input_ctrl, corners)), np.concatenate((output_ctrl, corners))


def decrease_ctrl_points():
    """Generate ctrl points that decrease the center of the image.

    (In proportion of the desired shapes)
    """
    input_ctrl = np.array(
        [
            [0.25, 0.25],  # (i_0, j_0)
            [0.25, 0.75],  # (i_0, j_1)
            [0.75, 0.25],  # (i_1, j_0)
            [0.75, 0.75],  # (i_1, j_1)
        ]
    )

    output_ctrl = np.array(
        [
            [0.35, 0.35],
            [0.35, 0.65],
            [0.65, 0.35],
            [0.65, 0.65],
        ]
    )

    corners = np.array(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    return np.concatenate((input_ctrl, corners)), np.concatenate((output_ctrl, corners))


def random_ctrl_points():
    """Generate random ctrl points.

    (In proportion of the desired shapes)
    """
    np.random.seed(777)
    input_ctrl = np.random.rand(10, 2)
    output_ctrl = input_ctrl + np.random.randn(10, 2) * 0.05

    corners = np.array(  # Add corners ctrl points
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    return np.concatenate((input_ctrl, corners)), np.concatenate((output_ctrl, corners))


def main():
    """Warp an image."""
    # Load the image and draw a rectangle in the middle
    image = Image.open("images/dog.jpeg")
    width, height = image.size
    ImageDraw.Draw(image).rectangle([width * 0.25, height * 0.25, width * 0.75, height * 0.75])

    # Build control points
    # input_ctrl, output_ctrl = increase_ctrl_points()
    # input_ctrl, output_ctrl = decrease_ctrl_points()
    input_ctrl, output_ctrl = random_ctrl_points()
    input_ctrl *= [height, width]
    output_ctrl *= [height, width]

    # Fit the thin plate spline from output to input
    tps = ThinPlateSpline(0.5)
    tps.fit(output_ctrl, input_ctrl)

    # Create the 2d meshgrid of indices for output image
    output_indices = np.indices((height, width), dtype=np.float64).transpose(1, 2, 0)  # Shape: (H, W, 2)

    # Transform it into the input indices
    input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(height, width, 2)

    # Interpolate the resulting image
    warped = np.concatenate(
        [
            scipy.ndimage.map_coordinates(np.array(image)[..., channel], input_indices.transpose(2, 0, 1))[..., None]
            for channel in (0, 1, 2)
        ],
        axis=-1,
    )

    plt.figure()
    plt.imshow(warped)

    plt.figure()
    plt.imshow(image)

    plt.show()


if __name__ == "__main__":
    main()
