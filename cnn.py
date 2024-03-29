import os
import time
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

feature_dir = "Input"
label_dir = "Expected"

# Get the list of files in the input and expected directories
feature_files = os.listdir(feature_dir)
label_files = os.listdir(label_dir)

# Get the prefixes for file matching
feature_prefixes = [feature.rsplit('_', 1)[0] for feature in feature_files]
label_prefixes = [label.rsplit('_', 1)[0] for label in label_files]

# Get the list of files that have both input and expected files
features = []
labels = []

for i in range(len(feature_files)):
    if feature_prefixes[i] in label_prefixes:
        features.append(feature_files[i])
        labels.append(label_files[label_prefixes.index(feature_prefixes[i])])

# Prepend the directory to the file names
features = [os.path.join(feature_dir, feature) for feature in features][0:3]
labels = [os.path.join(label_dir, label) for label in labels][0:3]

# Prepare the images for the CNN
transform = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor()
])

def prepareImage(img):
    image = Image.open(img)
    image = transform(image)
    return image

print ("Preparing data...")

features = torch.stack([prepareImage(feature) for feature in features])
labels = torch.stack([prepareImage(label) for label in labels])

# Finish the data preparation
training_data_set = TensorDataset(features, labels)
training_data_loader = DataLoader(training_data_set, 
                                  batch_size=4, 
                                  shuffle=True)

print("Data preparation complete.")

# Define the CNN
model = torch.nn.Sequential(
    torch.nn.Conv2d(4, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 1, 3, padding=1),
    torch.nn.Sigmoid()
)

# Define other parameters
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the CNN
print("Training the CNN...")
num_epochs = 5
total_num_batches = len(training_data_loader)

for epoch in range(num_epochs):
    t = time.time()

    for batch_num, (features, labels) in enumerate(training_data_loader):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        if (batch_num + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_num + 1}/{total_num_batches}, Loss: {loss:.4f}")
    elapsed = time.time() - t

    print(f"Epoch {epoch + 1} completed in: {elapsed:.2f} seconds")

print("Training complete. Saving the model...")

# Save the model
torch.save(model.state_dict(), "model.pth")

print("Model saved.")