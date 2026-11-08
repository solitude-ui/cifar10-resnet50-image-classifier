# CIFAR-10 Image Classification using ResNet50

## Overview

This project implements an image classification system using **Transfer Learning with ResNet50** on the CIFAR-10 dataset.

A pretrained ResNet50 model is used as the backbone, with a custom classification head trained to recognize 10 different object categories.

The trained model is integrated into a **Streamlit web application**, allowing users to upload images and view the predicted class and confidence score.

## Technologies Used

- Python
- TensorFlow / Keras
- ResNet50
- Transfer Learning
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab

## Dataset

The project uses the CIFAR-10 dataset.

- 60,000 colour images
- Image dimensions: 32 × 32 pixels
- 10 object categories
- 50,000 training images
- 10,000 test images

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Project Workflow

1. Environment setup and GPU verification.
2. CIFAR-10 dataset loading and exploration.
3. Data preprocessing and augmentation.
4. Loading pretrained ResNet50 with ImageNet weights.
5. Freezing the pretrained backbone.
6. Building a custom classification head.
7. Training the model using feature extraction.
8. Fine-tuning selected ResNet50 layers.
9. Evaluating model performance.
10. Saving the trained model.
11. Integrating the model into a Streamlit application.

## Project Structure

```text
cifar10-resnet50-image-classifier/
│
├── app.py
├── Transfer Learning for Image Classification.ipynb
├── best_resnet50_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-resnet50-image-classifier.git
```

### 2. Navigate to the project folder

```bash
cd cifar10-resnet50-image-classifier
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

After activation, `(venv)` should appear at the beginning of your terminal prompt.

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Using the Application

1. Open the Streamlit application.
2. Upload an image in JPG, JPEG, or PNG format.
3. Click the Predict button.
4. View the predicted object class and confidence score.

## Model Information

- Architecture: ResNet50
- Learning approach: Transfer Learning
- Dataset: CIFAR-10
- Number of output classes: 10
- Framework: TensorFlow / Keras

## Future Improvements

- Improve classification accuracy through hyperparameter tuning.
- Experiment with different pretrained architectures.
- Improve predictions on real-world images.
- Deploy the Streamlit application online.

## Author

**Adil Niham**

GitHub: https://github.com/YOUR_USERNAME
