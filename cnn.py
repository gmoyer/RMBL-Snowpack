import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import create_model

from preprocess import load_data, preprocess_data

# Load the data

features, labels = load_data("Input", "Expected")

# Preprocess the data

features, labels = preprocess_data(features[0:10], labels[0:10])

training_data_set = TensorDataset(features, labels)
training_data_loader = DataLoader(training_data_set, 
                                  batch_size=4, 
                                  shuffle=True)


# Define the CNN
model = create_model()
# Define other parameters
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

# Train the CNN
print("Training the CNN...")
num_epochs = 100
total_num_batches = len(training_data_loader)

for epoch in range(num_epochs):
    t = time.time()

    for batch_num, (feature, label) in enumerate(training_data_loader):
        optimizer.zero_grad()
        prediction = model(feature)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()

        # if (batch_num + 1) % 10 == 0:
        #     print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_num + 1}/{total_num_batches}, Loss: {loss:.4f}")
    elapsed = time.time() - t

    print(f"Epoch {epoch + 1} completed in: {elapsed:.2f} seconds with loss: {loss:.4f}")

print("Training complete. Saving the model...")

# Save the model
torch.save(model.state_dict(), "model.pth")

print("Model saved.")