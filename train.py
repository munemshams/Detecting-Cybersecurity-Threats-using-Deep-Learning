import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from model import AttackClassifier

# Load datasets
train_df = pd.read_csv("labelled_train.csv")
val_df = pd.read_csv("labelled_validation.csv")

# Split features and labels
X_train = train_df.drop("sus_label", axis=1)
y_train = train_df["sus_label"]

X_val = val_df.drop("sus_label", axis=1)
y_val = val_df["sus_label"]

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)

# Model
input_size = X_train.shape[1]
model = AttackClassifier(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "attack_model.pth")

print("Model saved as attack_model.pth")
