from torchvision import transforms
from PIL import Image
import os
import torch
import time
import matplotlib.pyplot as plt

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
        transforms.Normalize((0.46, 0.39, 0.29), (0.16, 0.14, 0.13))
    ])

    transform_label = transforms.Compose([
        transforms.Resize((LABEL_IMAGE_SIZE, LABEL_IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    def prepare_feature(img):
        image = Image.open(img)
        image = transform_feature(image)
        return image
    
    def prepare_label(img):
        image = Image.open(img)
        image = transform_label(image)
        image[image < 0] = -1
        image[torch.logical_and(image >= 0, image < 0.5)] = 0
        image[image >= 0.5] = 1
        return image
    

    features = torch.stack([prepare_feature(feature) for feature in features])
    labels = torch.stack([prepare_label(label) for label in labels])

    elapsed = time.time() - t

    print(f"Data prepared in {elapsed:.2f} seconds.")

    return features, labels


# Exploratory Data Analysis

# features, labels = load_data("Input", "Expected")
# features, labels = preprocess_data(features, labels)

# means = []
# stds = []
# for c in range(3):
#     mean_row = []
#     std_row = []
#     for i in range(len(features)):
#         feature = features[i]
#         label = labels[i]
#         feature = feature[c:c+1, :, :]
#         feature = feature[label >= 0]
#         mean_row.append(feature.mean())
#         std_row.append(feature.std())
#     means.append(mean_row)
#     stds.append(std_row)


# # Create three scatter plots
# for c in range(3):
#     plt.scatter(means[c], stds[c])
#     plt.xlabel('Mean')
#     plt.ylabel('Standard Deviation')
#     plt.title(f'Scatter Plot for Channel {c+1}')
#     plt.show()

# # Print out the averages of each mean row and std row
# for c in range(3):
#     mean_row = means[c]
#     std_row = stds[c]
#     mean_avg = sum(mean_row) / len(mean_row)
#     std_avg = sum(std_row) / len(std_row)
#     print(f"Average mean of channel {c+1}: {mean_avg}")
#     print(f"Average std of channel {c+1}: {std_avg}\n")