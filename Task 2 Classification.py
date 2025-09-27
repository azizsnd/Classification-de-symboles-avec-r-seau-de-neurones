import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import accuracy_score

# Generate a dataset of symbols ("*" or "/") with corresponding labels (0 for "*", 1 for "/")
def generate_data(n_samples=1000):
    symbols = ["*", "/"]
    data = [random.choice(symbols) for _ in range(n_samples)]
    labels = [0 if s == "*" else 1 for s in data]
    return data, labels

# Convert symbols into one-hot encoded tensors
def encode_data(data):
    mapping = [[1.0, 0.0] if c == '*' else [0.0, 1.0] for c in data]
    return torch.tensor(mapping, dtype=torch.float32)


# Convert label list into a tensor of integers
def encode_labels(labels):
    return torch.tensor(labels, dtype=torch.long)

# Define a simple neural network classifier
class SymbolClassifier(nn.Module):
    def __init__(self):
        super(SymbolClassifier, self).__init__()
        self.fc = nn.Sequential(
            # Input layer to hidden layer, gets linearly transformed into 16 features
            nn.Linear(2, 16),
            # Replace negative values with 0
            nn.ReLU(),
            # Hidden layer to output layer, outputs scores for the sample
            nn.Linear(16, 2)
        )   # The higher score determines the predicted class
    def forward(self, x):
        return self.fc(x)

# Create dataset and encode inputs and labels
data, labels = generate_data(1000)
X = encode_data(data)
y = encode_labels(labels)

# Split the data into 80% training and 20% testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Initialize the model, loss function, and optimizer
model = SymbolClassifier()

# loss function used to measure how well the model's predictions match the true labels.
criterion = nn.CrossEntropyLoss()

# Updates the model's weights during training to minimize the loss.
# model.parameters() gives it the weights to update, and lr sets the learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model for 50 epochs
for epoch in range(50):
    optimizer.zero_grad()             # Reset gradients
    outputs = model(X_train)          # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()                   # Backward pass
    optimizer.step()                  # Update weights

# Evaluate the model on the test set
with torch.no_grad():  # Disable gradient calculation for evaluation
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, dim=1)
    acc = accuracy_score(y_test.numpy(), predicted_classes.numpy())

    print(y_test)
    print(predicted_classes)
    print(f"Accuracy: {acc:.4f}")