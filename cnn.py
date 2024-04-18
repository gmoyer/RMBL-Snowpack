import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model3 import Model3

from preprocess import load_data, preprocess_data
# Load the data

features, labels = load_data("Clipped-Input", "Expected")

# Preprocess the data

features, labels = preprocess_data(features[0:25], labels[0:25])


training_data_set = TensorDataset(features, labels)
training_data_loader = DataLoader(training_data_set, 
                                  batch_size=5, 
                                  shuffle=True)

# Define the CNN
model = Model3()
# Define other parameters

bce_loss = torch.nn.BCELoss()

def loss_function(prediction, label):
    mask = label != -1
    return bce_loss(prediction[mask], label[mask])

optimizer = torch.optim.Adam(model.parameters())

# Train the CNN
print("Training the CNN...")
num_epochs = 100
total_num_batches = len(training_data_loader)

for epoch in range(num_epochs):
    t = time.time()

    for batch_num, (feature, label) in enumerate(training_data_loader):
        prediction = model(feature)

        loss = loss_function(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t

    print(f"Epoch {epoch + 1} completed in: {elapsed:.2f} seconds with loss: {loss:.4f}")

print("Training complete. Saving the model...")

# Save the model
torch.save(model.state_dict(), "model3.pth")

print("Model saved.")