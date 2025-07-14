# fMRI Image Denoising Training Pipeline

This project provides a training pipeline for medical image data in NIfTI format (`.nii.gz`). The pipeline is designed for flexibility and easy experimentation using Jupyter Notebooks.

## 📁 Folder Structure

```
project-root/
│
├── final_train.ipynb     # Main training notebook
├── data/                 # Folder for training data
│   ├── case1.nii.gz
│   ├── case2.nii.gz
│   └── ...
├── validation/           # Folder for validation data
│   ├── val1.nii.gz
│   ├── val2.nii.gz
│   └── ...
```

## 🚀 Getting Started

### 1. Prepare Your Data

* Place your training `.nii.gz` files in the `data/` folder.
* Place your validation `.nii.gz` files in the `validation/` folder.

### 2. Train the Model

Run the `final_train.ipynb` notebook to:

* Load and preprocess the data
* Set up the model and training parameters
* Train the model
* Evaluate the results on the validation set

### 3. Customize Parameters

You can customize various training parameters inside the notebook:

* **Learning Rate**
* **Optimizer**
* **Batch Size**
* **Number of Epochs**

Modify these settings at in the notebook to experiment with different training configurations.

---

## 🛠 Requirements

Make sure to install all dependencies and include:

* Python 3.8+
* NumPy
* PyTorch
* Nibabel
* Matplotlib
* scikit-learn
* Jupyter

---

## 📈 Output

* The notebook will display training loss and validation metrics.
* You can modify it to save models, logs, or predictions as needed.

---
