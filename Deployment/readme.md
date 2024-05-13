# Snowpack Identification Script

This script, `identifysnowpack.py`, is used to identify snowpack in images using a pre-trained model (`model.pth`). Here's how you can run the script:

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- [Pillow](https://python-pillow.org/): `pip install pillow`
- [Torch](https://pytorch.org/): Follow the installation instructions specific to your platform and Python version. Make sure to install the same version of Python as listed in the Torch installation instructions.

## Usage

To run the script, follow these steps:

1. Place the `identifysnowpack.py` script and the `model.pth` file in the same directory.
2. Create a separate python file and add `from identifysnowpack import identify_image`
3. Run the function `identify_image` by providing a path to the input image, a path to a shapefile for clipping the input image, and specifying a path to save the output image.

## Example

For an example of how to use the `identifysnowpack.py` script, refer to the `example-script.py` file in this repository.

That's it! You should now be able to run the script and identify snowpack in images. If you encounter any issues, please make sure you have the correct dependencies installed and that the Python version matches the requirements specified by Torch.
