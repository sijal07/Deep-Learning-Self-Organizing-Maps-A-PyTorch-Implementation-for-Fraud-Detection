# Deep-Learning-Self-Organizing-Maps-A-PyTorch-Implementation-for-Fraud-Detection
DLSOM (Deep Learning Self-Organizing Maps)
A GPU-accelerated implementation of Self-Organizing Maps (SOM) using PyTorch for fraud detection applications.

Overview
This project implements a Self-Organizing Map (SOM) with GPU acceleration capabilities for anomaly detection in financial transactions. The implementation includes:

GPU-accelerated SOM training using PyTorch

Synthetic fraud dataset generation for testing

Comprehensive visualization tools including U-Matrix, component planes, and fraud concentration maps

Multiple visualization techniques to analyze SOM behavior and detect anomalies

Features
Core Components
GPUSOM Class: Custom SOM implementation with PyTorch GPU acceleration

Synthetic Data Generation: Configurable fraud detection dataset with normal and anomalous transactions

Multiple Training Modes: Support for both CPU and GPU training

Visualization Suite
U-Matrix: Distance map visualization showing cluster boundaries

Component Planes: Individual feature visualization across the SOM grid

Hit Histogram: Frequency of samples mapped to each neuron

Fraud Concentration Map: Visualization of fraud distribution across the SOM

BMU Distribution: Scatter plot showing Best Matching Unit assignments

Installation
bash
pip install minisom numpy torch matplotlib seaborn scikit-learn pandas
Usage
Basic Example
python
# Generate synthetic fraud data
X, y, feature_names = generate_fraud_data(n_samples=1000, fraud_ratio=0.1)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train SOM
som = GPUSOM(x=15, y=15, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
weights = som.train(X_scaled, num_epochs=100)

# Generate visualizations
# (Visualization code as shown in the notebook)
Key Parameters
x, y: SOM grid dimensions

input_len: Number of input features

sigma: Neighborhood radius

learning_rate: Initial learning rate

gpu: Boolean flag for GPU acceleration

Dataset
The synthetic fraud dataset includes 4 features:

Transaction_Amount

Time_of_Day

Location_Distance

Frequency

By default, generates 1000 samples with 10% fraud cases.

GPU Acceleration
The implementation automatically detects and utilizes available GPUs:

Uses PyTorch tensors for computation

Falls back to CPU if GPU is unavailable

Provides significant speedup for large datasets

Visualization Examples
The notebook includes comprehensive visualizations:

U-Matrix: Identifies cluster boundaries and anomalies

Component Planes: Shows how each feature contributes to the SOM organization

Fraud Patterns: Highlights regions with high fraud concentration

Hit Analysis: Shows sample distribution across the map

Requirements
Python 3.7+

PyTorch

NumPy

Matplotlib

Seaborn

scikit-learn

MiniSom (for reference implementations)

Applications
Fraud detection in financial transactions

Anomaly detection in multivariate data

Data visualization and clustering

Feature analysis and dimensionality reduction

Performance
GPU Training: Significant speedup over CPU implementations

Scalable: Handles large datasets efficiently

Configurable: Flexible parameters for different use cases
