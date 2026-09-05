# Plant Disease Classification System 🌱

A deep learning-based plant disease classification system that uses a **Convolutional Neural Network (CNN)** to identify plant diseases from leaf images. The model is trained to classify **15 plant health and disease categories** and provides predictions based on uploaded leaf images.

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and quality. Early identification of diseases can help in taking appropriate preventive and corrective measures. This project aims to automate plant disease detection using image classification and deep learning.

The system takes a plant leaf image as input, preprocesses the image, extracts relevant visual features using a CNN, and predicts the corresponding disease class.

## 🎯 Objectives

* Build a CNN-based model for multi-class plant disease classification.
* Classify leaf images into 15 different plant health/disease categories.
* Apply image preprocessing and data augmentation to improve model generalization.
* Use Batch Normalization, Global Average Pooling, Dropout, and label smoothing to improve training stability and reduce overfitting.
* Evaluate the model using accuracy, validation loss, and confusion matrix analysis.
* Develop an image-based prediction pipeline for disease identification.

## 🧠 Model Architecture

The custom CNN consists of three convolutional blocks followed by a classification head:

```text
Input Image (224 × 224 × 3)
          ↓
Conv2D (32 filters, 3×3)
          ↓
Batch Normalization
          ↓
Max Pooling (2×2)
          ↓
Conv2D (64 filters, 3×3)
          ↓
Batch Normalization
          ↓
Max Pooling (2×2)
          ↓
Conv2D (128 filters, 3×3)
          ↓
Batch Normalization
          ↓
Max Pooling (2×2)
          ↓
Global Average Pooling
          ↓
Dense (128, ReLU)
          ↓
Dropout (0.5)
          ↓
Dense (15, Softmax)
          ↓
Disease Prediction
```

### Mathematical Components

**ReLU activation:**

[
f(x) = \max(0,x)
]

ReLU introduces non-linearity and allows the network to learn complex visual patterns.

**Softmax:**

[
P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
]

Softmax converts the final layer's outputs into probabilities for the 15 classes.

**Categorical Cross-Entropy:**

[
L=-\sum_i y_i\log(p_i)
]

The loss function measures the difference between the actual class and the predicted probability distribution.

**Label Smoothing:**

[
y_{smooth}=y(1-\epsilon)+\frac{\epsilon}{K}
]

Label smoothing reduces overconfidence and can improve generalization.

## 🔬 Data Preprocessing

Images are resized to:

```text
224 × 224 × 3
```

Pixel values are normalized from:

```text
0–255 → 0–1
```

using:

```python
rescale=1./255
```

### Data Augmentation

The training pipeline uses:

* Rotation up to 20°
* Zoom up to 20%
* Horizontal and vertical shifting
* Horizontal flipping

These transformations generate variations of training images and help the model generalize better to different image orientations and positions.

## 📊 Evaluation

The model can be evaluated using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Confusion matrix
* ROC-AUC analysis

The confusion matrix is particularly useful for identifying classes that the model frequently confuses with each other, especially visually similar plant diseases.

## 🌿 Supported Classes

The model contains 15 classes:

```text
Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_Septoria_leaf_spot
Tomato_Spider_mites_Two_spotted_spider_mite
Tomato__Target_Spot
Tomato__Tomato_YellowLeaf__Curl_Virus
Tomato__Tomato_mosaic_virus
Tomato_healthy
```

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Pillow**

## 📁 Project Structure

```text
Plant-Disease-Classification/
│
├── model/
│   └── custom_model2.keras
│
├── notebooks/
│   └── training.ipynb
│
├── requirements.txt
├── README.md
└── ...
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Plant-Disease-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Model

Load the trained `.keras` model:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "custom_model2.keras",
    compile=False
)
```

An input leaf image should then be resized to `224 × 224`, converted to RGB, normalized, and passed to the model for prediction.

The model returns probabilities for all 15 classes. The class with the highest probability is selected as the predicted disease.

## ⚠️ Limitations

The custom CNN performs well on the overall classification task but can struggle with **visually similar diseases**, particularly among some tomato disease classes. Similar colors, lesions, textures, and leaf patterns can make fine-grained classification difficult.

The model is also trained on a specific image dataset, so performance may decrease on real-world images with different lighting, backgrounds, camera quality, leaf orientations, or environmental conditions.

## 🔮 Future Scope

Future improvements could include:

* Transfer learning using architectures such as MobileNet or EfficientNet.
* Training with more diverse real-world field images.
* Higher-resolution inputs for fine-grained disease identification.
* Further hyperparameter and architecture optimization.
* Deployment as a web or mobile application for practical agricultural use.

## 👨‍💻 Author

**Anshul**

This project was developed as a deep learning application for automated plant disease classification and analysis.
