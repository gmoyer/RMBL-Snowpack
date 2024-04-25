from torch import nn, load
from torchvision import transforms
from torchvision.utils import save_image
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

def preprocess_image(img):
    image = Image.open(img)

    width, height = image.size

    feature_width = int(FEATURE_IMAGE_HEIGHT / height * width)
    feature_height = FEATURE_IMAGE_HEIGHT

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

def identify_image(input_path, output_path):
    model.eval()
    feature = preprocess_image(input_path)
    prediction = model(feature)
    binary_prediction = (prediction > 0.5).float()
    save_image(binary_prediction, output_path)


