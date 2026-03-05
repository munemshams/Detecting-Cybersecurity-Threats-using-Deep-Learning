# Detecting Cybersecurity Threats Using Deep Learning (PyTorch)

## Overview

This project builds a **deep learning model using PyTorch** to detect suspicious network activity using tabular cybersecurity data. The model is trained to classify network traffic as either **normal or suspicious** using features extracted from network events.

Cybersecurity threat detection is an important application of machine learning in modern security systems. By analyzing patterns in network traffic, machine learning models can help detect potential attacks or abnormal behavior automatically.

This project demonstrates a complete **deep learning workflow for tabular data**, including:

- Loading structured data from CSV files
- Data preprocessing and normalization
- Building a neural network classifier in PyTorch
- Training and validating a deep learning model
- Evaluating performance on a test dataset
- Exporting predictions to a CSV file

The repository provides a practical example of how **deep learning can be applied to cybersecurity threat detection**.

---

# Dataset

The dataset used in this project is hosted on Kaggle:

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/munemshariarshams/dataset-used-for-detecting-cyber-threats

Due to repository file limits and dataset size considerations, the CSV files are **not included directly in this repository**.

### Download Instructions

1. Download the dataset from the Kaggle link above.
2. Extract the files into the project directory.
3. Ensure the following files are present:

```
labelled_train.csv
labelled_validation.csv
labelled_test.csv
```

These files contain labeled network traffic data used for training, validating, and testing the model.

---

# Model Architecture

The project uses a **feedforward neural network (Multi-Layer Perceptron)** implemented in PyTorch.

### Network Structure

```
Input Features
↓
Fully Connected Layer (64 neurons)
↓
ReLU Activation
↓
Fully Connected Layer (32 neurons)
↓
ReLU Activation
↓
Output Layer (1 neuron)
↓
Sigmoid Activation
↓
Binary Prediction
```

The model outputs a value between **0 and 1**, representing the probability that a network record is suspicious.

---

# Project Workflow

The project consists of two main stages: **training the model** and **evaluating the model**.

---

## 1. Training the Model

The training process is handled by `train.py`.

The script performs the following steps:

1. Loads the training and validation datasets
2. Separates features from the target label (`sus_label`)
3. Normalizes features using **StandardScaler**
4. Converts the data into PyTorch tensors
5. Trains a neural network classifier for several epochs
6. Saves the trained model weights

After training completes, the model file is generated automatically:

```
attack_model.pth
```

This file contains the trained neural network parameters.

The model file is **not included in the repository** because it is automatically generated when the training script is executed.

---

## 2. Evaluating the Model

Model evaluation is handled by `evaluate.py`.

The script performs the following steps:

1. Loads the trained model
2. Processes the test dataset
3. Generates predictions
4. Calculates classification accuracy
5. Saves predictions to a CSV file

Output generated:

```
outputs/predictions.csv
```

### Example Output

| prediction |
|-----------|
| 0 |
| 1 |
| 0 |
| 0 |
| 1 |

Where:

| Value | Meaning |
|------|--------|
| 0 | Normal network activity |
| 1 | Suspicious network activity |

---

# Installation and Dependencies

Install the required Python libraries using:

```
python -m pip install torch pandas numpy scikit-learn
```

---

# Python Libraries Used

| Library | Purpose |
|--------|--------|
| **torch** | Core deep learning framework used to build and train neural networks |
| **pandas** | Used to load and manipulate tabular datasets |
| **numpy** | Provides numerical computation utilities |
| **scikit-learn** | Used for feature normalization and evaluation metrics |

---

# How to Run the Project

## Step 1 — Train the Model

Run the training script:

```
python train.py
```

This will:

- load the training dataset
- train the neural network
- generate the trained model file

Output generated automatically:

```
attack_model.pth
```

---

## Step 2 — Evaluate the Model

Run the evaluation script:

```
python evaluate.py
```

This will:

- load the trained model
- evaluate the test dataset
- export predictions

Output generated automatically:

```
outputs/predictions.csv
```

---

# Files Included

| File | Description |
|-----|-------------|
| `model.py` | Defines the neural network architecture |
| `train.py` | Trains the deep learning model using the training dataset |
| `evaluate.py` | Evaluates the trained model and generates predictions |
| `README.md` | Project documentation |

---

# Repository Notes

The following files are **generated automatically when running the project** and are not included in the repository:

- `attack_model.pth` — trained neural network model
- `outputs/predictions.csv` — prediction output file

These files will be created automatically when running the training and evaluation scripts.

---

# Project Objective

The objective of this project is to demonstrate how **deep learning can be applied to cybersecurity threat detection**. By analyzing patterns in network traffic features, the model learns to distinguish between normal activity and potentially suspicious events.

---

# Example Applications

Deep learning models like this can be applied to:

- **Intrusion detection systems**
- **Network anomaly detection**
- **Cybersecurity monitoring platforms**
- **Automated threat detection systems**

---

