import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model1 import Model1
from model2 import Model2
from model3 import Model3
import torchvision
import preprocess
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess_data

# Load the data

features, labels = load_data("Clipped-Input", "Expected")

# Preprocess the data

features, labels = preprocess_data(features[:5], labels[:5])

# Display the label mask as gray
labels[labels < 0] = 0.5

num_validation_samples = features.shape[0]

features = features.reshape([num_validation_samples, 3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE])


def verify_model(model):
    # Run the model on the input data

    predictions = model(features)

    # Convert predictions to binary values
    binary_predictions = (predictions > 0.5).float()

    accuracies = []

    for i in range(predictions.shape[0]):
        prediction = binary_predictions[i]
        label = labels[i]
        feature = features[i]

        # print(f"Label min: {label.min()}, max: {label.max()}")
        # print(f"Prediction min: {prediction.min()}, max: {prediction.max()}")

        mask = label != 0.5

        # Compare prediction and label images
        accuracy = torch.mean((prediction[mask] == label[mask]).float()) * 100
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.2f}%")

        # Save the prediction and label images
        prediction[label == 0.5] = 0.5
        prediction_image = prediction.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
        label_image = label.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
        feature_image = feature.view(3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE)
        torchvision.utils.save_image(prediction_image, f"Validation/prediction_{i}.png")
        torchvision.utils.save_image(label_image, f"Validation/label_{i}.png")
        torchvision.utils.save_image(feature_image, f"Validation/feature_{i}.png")
    return accuracies

# Verify Model1
# Load the saved model
model = Model3()
model.load_state_dict(torch.load("model3.pth"))
model.eval()

# model2 = Model2()
# model2.load_state_dict(torch.load("model2.2.pth"))
# model2.eval()

accuracies = verify_model(model)
# # accuracies2 = verify_model(model2)
# # accuracies = [accuracies1, accuracies2]
# # labels = ['Model 1', 'Model 2']
# # Plot a histogram of model 1 and 2 accuracies
# plt.hist(accuracies, bins=10, histtype='bar')
# plt.xlabel('Accuracy')
# plt.ylabel('Frequency')
# plt.title('Model Accuracies')
# plt.legend()
# plt.show()

