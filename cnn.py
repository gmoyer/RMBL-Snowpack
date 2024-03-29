import os
import torch
from torchvision import transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

featureDir = "Input"
labelDir = "Expected"
featureFiles = os.listdir(featureDir)
labelFiles = os.listdir(labelDir)

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])
print(inputFiles[0])
sampleImage = Image.open(inputFiles[0])

imageTensor = transform(sampleImage).unsqueeze(0)

print(imageTensor)