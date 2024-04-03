import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import create_model, Model
import torchvision
import preprocess

from preprocess import load_data, preprocess_data

# Load the data

features, labels = load_data("Input", "Expected")

# Preprocess the data

features, labels = preprocess_data(features[0:5], labels[0:5])

# Display the label mask as gray
labels[labels < 0] = 0.5

num_validation_samples = features.shape[0]

features = features.reshape([num_validation_samples, 3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE])

# Load the saved model
model = Model()
model.load_state_dict(torch.load("model2.pth"))
model.eval()

# Run the model on the input data

predictions = model(features)

# Convert predictions to binary values
binary_predictions = (predictions > 0.5).float()

for i in range(predictions.shape[0]):
    prediction = binary_predictions[i]
    label = labels[i]
    feature = features[i]

    # print(f"Label min: {label.min()}, max: {label.max()}")
    # print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")

    mask = label != 0.5

    # Compare prediction and label images
    accuracy = torch.mean((prediction[mask] == label[mask]).float()) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Save the prediction and label images
    prediction[label == 0.5] = 0.5
    prediction_image = prediction.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
    label_image = label.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
    feature_image = feature.view(3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE)
    torchvision.utils.save_image(prediction_image, f"Validation/prediction_{i}.png")
    torchvision.utils.save_image(label_image, f"Validation/label_{i}.png")
    torchvision.utils.save_image(feature_image, f"Validation/feature_{i}.png")

