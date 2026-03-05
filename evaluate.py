import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from model import AttackClassifier

# Load datasets
train_df = pd.read_csv("labelled_train.csv")
test_df = pd.read_csv("labelled_test.csv")

# Split features and labels
X_train = train_df.drop("sus_label", axis=1)
y_train = train_df["sus_label"]

X_test = test_df.drop("sus_label", axis=1)
y_test = test_df["sus_label"]

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)

# Load model
input_size = X_train.shape[1]
model = AttackClassifier(input_size)

model.load_state_dict(torch.load("attack_model.pth"))

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    preds = (outputs > 0.5).float().numpy().flatten()

# Accuracy
acc = accuracy_score(y_test, preds)

print("Test Accuracy:", acc)

# Save predictions
os.makedirs("outputs", exist_ok=True)

df = pd.DataFrame({
    "prediction": preds
})

df.to_csv("outputs/predictions.csv", index=False)

print("Predictions saved to outputs/predictions.csv")
