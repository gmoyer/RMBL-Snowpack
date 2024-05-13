from torch import nn, load, from_numpy
from torchvision import transforms
from torchvision.utils import save_image
import fiona
import rasterio
import rasterio.mask
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

FEATURE_IMAGE_HEIGHT = 256*4

def Model1():
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=5, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 8, kernel_size=5, padding=2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


# From https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
def clip_raster(raster_path, vector_path):
    with fiona.open(vector_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    return out_image, out_transform

def preprocess_image(raster_img, raster_transform):
    image = from_numpy(raster_img)
    # image = image.permute(1, 2, 0)

    print(image.shape)

    width, height = image.shape[2], image.shape[1]
    scale_x = abs(raster_transform[0])
    scale_y = abs(raster_transform[4])

    feature_width = int(width * scale_x * 4)
    feature_height = int(height * scale_y * 4)

    transform_feature = transforms.Compose([
        transforms.Resize((feature_height, feature_width)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        transforms.Normalize((0.46, 0.39, 0.29), (0.16, 0.14, 0.13))
    ])

    
    # print(img, "Feature", image.width, image.height, image.width/image.height)
    image = transform_feature(image)
    return image.view(1, 3, feature_height, feature_width)

model = Model1()
model.load_state_dict(load("model.pth"))

def identify_image(raster_path, vector_path, output_path):
    # Preprocessing
    raster_image, raster_transform = clip_raster(raster_path, vector_path)
    feature = preprocess_image(raster_image, raster_transform)

    model.eval()
    prediction = model(feature)
    binary_prediction = (prediction > 0.5).float()
    save_image(binary_prediction, output_path)


# Testing
identify_image("DeerCreekTrail_2019_05_11_ortho_3cm.tif", "DeerCreekTrail_Clip.gpkg", "output.png")