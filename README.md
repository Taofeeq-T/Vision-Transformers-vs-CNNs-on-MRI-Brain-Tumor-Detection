# Vision-Transformers-vs-CNNs-on-MRI-Brain-Tumor-Detection

# Brain Tumor Classification: Comparing CNNs and Transformers

## Introduction
Medical imaging, especially Magnetic Resonance Imaging (MRI), is a cornerstone of modern healthcare, playing a pivotal role in detecting and diagnosing brain tumors. These tumors, whether benign or malignant, have significant implications for patient outcomes, emphasizing the need for accurate and timely diagnosis. Despite advancements in AI and imaging technologies, challenges remain in deploying these solutions effectively, particularly in low-resource settings.

This project explores the use of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (VTs)** for classifying brain tumors into four classes: `meningioma`, `glioma`, `pituitary tumor`, and `no tumor`. By leveraging transfer learning, the study evaluates the performance of pre-trained models, focusing on their ability to adapt to smaller datasets while mitigating computational constraints.

---

## Project Overview

### Objective
To benchmark the performance of CNNs (`VGG16`, `VGG19`) and Transformers (`ViT`, `Beit`) in brain tumor classification using transfer learning, comparing their metrics such as accuracy, sensitivity, specificity, F1-score, and ROC-AUC.

### Motivation
- **Medical Need**: Early and accurate tumor classification impacts treatment plans and outcomes.
- **Technical Gap**: Evaluate CNNs versus VTs to identify optimal approaches for medical imaging.
- **Real-World Constraints**: Address challenges posed by limited data and computational resources.

---

## Dataset
The project utilizes a labeled dataset containing MRI scans with the following classes:
1. **Meningioma**
2. **Glioma**
3. **Pituitary Tumor**
4. **No Tumor**

The dataset is found at https://doi.org/10.34740/KAGGLE/DSV/2645886

The dataset is structured in separate directories for each class. Images are preprocessed and resized to 224x224 pixels for model compatibility.

![train_data_dist](https://github.com/user-attachments/assets/2425392a-5240-4387-be39-56a566f8a46e)


![test_data_dist](https://github.com/user-attachments/assets/78625c15-d3da-4c80-969c-ee0b8e892e6e)


---

## Methodology

### 1. Data Preprocessing
- **For CNNs (TensorFlow)**: Images are normalized to `[0, 1]` and augmented using `ImageDataGenerator`.
- **For Transformers (PyTorch)**: Images are preprocessed using Hugging Face `AutoFeatureExtractor` and normalized to `[-1, 1]`.

### 2. Model Architectures
#### Convolutional Neural Networks (CNNs)
- **VGG16**: Extracts hierarchical features with frozen pre-trained layers.
- **VGG19**: A deeper variant of VGG16 with additional convolutional layers.

#### Vision Transformers (VTs)
- **ViT**: Leverages self-attention for global context understanding.
- **Beit**: A transformer model designed for high performance in image classification tasks.

### 3. Training and Evaluation
- **Transfer Learning**: Fine-tuning the classification layers of pre-trained models.
- **Hyperparameter Tuning**: Focused on learning rates to maximize the F1-score.
- **Evaluation Metrics**: Accuracy, sensitivity, specificity, F1-score, ROC-AUC, and confusion matrix.

---

## Results
**Note**: Full results are pending due to GPU constraints for comprehensive training and fine-tuning. Preliminary insights indicate:
- **CNNs** excel at extracting local image features, making them well-suited for high-detail areas in MRI scans.
- **Transformers**, while more computationally demanding, effectively capture global context and excel when trained on larger datasets.

Final results will include:
- Comparative analysis of performance metrics.
- Visualizations such as confusion matrices and ROC curves.

---

## Potential Impact
This project aims to:
- **Improve Diagnostic Efficiency**: Offer insights into model suitability for MRI classification tasks.
- **Support Low-Resource Settings**: Enable deployment of lightweight, pre-trained AI models with minimal training.
- **Advance Medical AI Research**: Highlight the strengths and limitations of CNNs and Transformers in medical imaging.

---

## How to Run

### Clone the Repository
```bash
git clone https://github.com/Taofeeq-T/Vision-Transformers-vs-CNNs-on-MRI-Brain-Tumor-Detection.git

