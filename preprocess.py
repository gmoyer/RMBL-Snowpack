from torchvision import transforms
from PIL import Image
import os
import torch
import time

Image.MAX_IMAGE_PIXELS = None

LABEL_IMAGE_SIZE = 256
FEATURE_IMAGE_SIZE = 4*LABEL_IMAGE_SIZE

def load_data(feature_dir, label_dir):

    print("Loading data...")
    t = time.time()

    # Get the list of files in the input and expected directories
    feature_files = os.listdir(feature_dir)
    label_files = os.listdir(label_dir)

    # Get the prefixes for file matching
    feature_prefixes = ['_'.join(feature.split('_')[:4]) for feature in feature_files]
    label_prefixes = [label.rsplit('_', 1)[0] for label in label_files]

    # Get the list of files that have both input and expected files
    features = []
    labels = []

    for i in range(len(feature_files)):
        if feature_prefixes[i] in label_prefixes:
            features.append(feature_files[i])
            labels.append(label_files[label_prefixes.index(feature_prefixes[i])])

    # Prepend the directory to the file names
    features = [os.path.join(feature_dir, feature) for feature in features]
    labels = [os.path.join(label_dir, label) for label in labels]

    print(f"Number of data points: {len(features)}")

    elapsed = time.time() - t
    print(f"Data loaded in {elapsed:.2f} seconds.")

    return features, labels


def preprocess_data(features, labels):
    print ("Preparing data...")
    t = time.time()
    # Prepare the images for the CNN
    transform_feature = transforms.Compose([
        transforms.Resize((FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE)),
        transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])
    transform_label = transforms.Compose([
        transforms.Resize((LABEL_IMAGE_SIZE, LABEL_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x>0.5).float()),
    ])

    def prepare_feature(img):
        image = Image.open(img)
        image = transform_feature(image)
        return image
    
    def prepare_label(img):
        image = Image.open(img)
        image = transform_label(image)
        return image
    

    features = torch.stack([prepare_feature(feature) for feature in features])
    labels = torch.stack([prepare_label(label) for label in labels])

    elapsed = time.time() - t

    print(f"Data prepared in {elapsed:.2f} seconds.")

    return features, labels