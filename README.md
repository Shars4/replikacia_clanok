# Replication Project: Potato Disease Classification (CNN vs. Traditional ML)

## Introduction

This project aims to replicate and extend the findings related to classifying potato leaf diseases using image analysis. The primary goal is to implement a Convolutional Neural Network (CNN) for this task and compare its performance against several traditional machine learning algorithms.

The project evaluates the effectiveness of different modeling approaches on the Potato Leaves Dataset, focusing on identifying three conditions: Early Blight, Late Blight, and Healthy leaves.

## Dataset

* **Source:** PlantVillage Potato Leaves Dataset (commonly found on Kaggle).
* **Classes:**
    1.  Potato___Early_blight
    2.  Potato___Late_blight
    3.  Potato___healthy

## Models Compared

This project implements and compares the following models:

1.  **Convolutional Neural Network (CNN):** A custom CNN architecture built using TensorFlow/Keras.
2.  **Traditional Machine Learning Models (via Scikit-learn):**
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM) with a linear kernel
    * Decision Tree
    * Random Forest
    * Logistic Regression
    * Gaussian Naive Bayes (Naive Bayes)
    * Linear Discriminant Analysis (LDA)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Install dependencies:** Ensure you have Python 3 installed. You can install the required libraries using pip:
    ```bash
    pip install tensorflow pandas matplotlib scikit-learn numpy
    ```
3.  **Dataset:** Download the ([Potato leaf dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)) (or similar source) and place it such that the script can find it at the path specified in the code (e.g., `/kaggle/input/potato-leaves/plants/`), or update the `dataset_dir` variable in the script accordingly.

## Usage

1.  **Run the Python script:** Execute the main Python script (e.g., `rep.py` or the Jupyter Notebook) from your terminal:
    ```bash
    python rep.py
    ```
    *(Replace `rep.py` with the actual filename)*

2.  **Output:** The script will:
    * Load and preprocess the data.
    * Train the CNN model and print its evaluation metrics and classification report.
    * Train the traditional ML models using scikit-learn pipelines (including scaling where appropriate) and print their evaluation metrics and classification reports.
    * Display comparison charts:
        * Overall accuracy of all models.
        * Per-class F1-scores for all models.
        * Per-class Recall scores for all models.
    * Display training history (accuracy/loss curves) for the CNN model.
    * Save the trained CNN model to `potato_disease_classifier_cnn_model.h5`.

## Results

The script generates several plots comparing the performance of the CNN against the traditional models based on overall accuracy, F1-score per class, and recall per class. These visualizations help in understanding the strengths and weaknesses of each approach for this specific image classification task. A summary table comparing accuracy and training times is also printed to the console.

*(Optional: You could embed example images of the output charts here if available)*

## Original Paper Reference

This project draws inspiration from and aims to replicate aspects of the work presented in:

* **Title:** Early stage Potato Disease Classification by analyzing Potato Plants using CNN
* **Link:** [https://ieeexplore.ieee.org/document/10212746](https://ieeexplore.ieee.org/document/10212746)
