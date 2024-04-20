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

features, labels = preprocess_data(features[75:80], labels[75:80])

print(f"Number of features: {features.shape[0]}")

# Display the label mask as gray
labels[labels < 0] = 0.5

# num_validation_samples = features.shape[0]

# features = features.reshape([num_validation_samples, 3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE])

verification_data_set = TensorDataset(features, labels)
verification_data_loader = DataLoader(verification_data_set, 
                                      batch_size=1, 
                                      shuffle=False)


def verify_model(model, filename, save_images=0):
    model.load_state_dict(torch.load(filename))
    model.eval()


    accuracies = []
    # Run the model on the input data
    for i, (feature, label) in enumerate(verification_data_loader):
        prediction = model(feature)

        # Convert predictions to binary values
        binary_prediction = (prediction > 0.5).float()

        mask = label != 0.5

        # Compare prediction and label images
        accuracy = torch.mean((binary_prediction[mask] == label[mask]).float()) * 100
        accuracies.append(accuracy)
        print(f"Image {i}, Accuracy: {accuracy:.2f}%")

        # Save the first few prediction and label images
        if i < save_images:
            binary_prediction[label == 0.5] = 0.5
            prediction_image = binary_prediction.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
            label_image = label.view(1, preprocess.LABEL_IMAGE_SIZE, preprocess.LABEL_IMAGE_SIZE)
            feature_image = feature.view(3, preprocess.FEATURE_IMAGE_SIZE, preprocess.FEATURE_IMAGE_SIZE)
            torchvision.utils.save_image(prediction_image, f"Validation/{i}_prediction.png")
            torchvision.utils.save_image(label_image, f"Validation/{i}_label.png")
            torchvision.utils.save_image(feature_image, f"Validation/{i}_feature.png")
    return accuracies




# Comparing models with histogram

accuracies1 = verify_model(Model1(), "model1_clipped.pth")
accuracies2 = verify_model(Model2(), "model2_clipped.pth")
accuracies3 = verify_model(Model3(), "model3.pth", 5)
# accuracies3 = verify_model(Model3(), "model3.pth")

print(f"Model 1 Mean Accuracy: {torch.mean(torch.tensor(accuracies1)):.3f}%")
print(f"Model 2 Mean Accuracy: {torch.mean(torch.tensor(accuracies2)):.3f}%")
print(f"Model 3 Mean Accuracy: {torch.mean(torch.tensor(accuracies3)):.3f}%")

# accuracies = [accuracies1, accuracies2]
# labels = ['Before Clipping', 'After Clipping']
# # # Plot a histogram of model 1 and 2 accuracies
# plt.hist(accuracies, bins=10, histtype='bar', label=labels)
# plt.xlabel('Accuracy')
# plt.ylabel('Frequency')
# plt.title('Model Accuracies')
# plt.legend()
# plt.show()

