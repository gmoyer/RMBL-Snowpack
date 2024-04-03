import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Model

from preprocess import load_data, preprocess_data

# Load the data

features, labels = load_data("Input", "Expected")

# Preprocess the data

features, labels = preprocess_data(features[0:5], labels[0:5])


training_data_set = TensorDataset(features, labels)
training_data_loader = DataLoader(training_data_set, 
                                  batch_size=1, 
                                  shuffle=True)


# Define the CNN
model = Model()
# Define other parameters

def loss_function(prediction, label):
    mask = label >= 0
    out = (prediction[mask]-label[mask])**2
    return out.mean()

optimizer = torch.optim.Adam(model.parameters())

# Train the CNN
print("Training the CNN...")
num_epochs = 150
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
torch.save(model.state_dict(), "model2.pth")

print("Model saved.")