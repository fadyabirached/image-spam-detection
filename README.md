# Image Spam Detection using Deep Learning

This project is a binary image classifier that distinguishes between **spam images** (e.g., advertisement, clickbait) and **natural/ham images** using a deep learning model based on **VGG19** and transfer learning.

## 📁 Dataset
- **SpamImages** and **NaturalImages** are used.
- Automatically downloaded and cleaned for duplicates and corruption.

## 🧠 Model
- Uses **VGG19** with `imagenet` weights (frozen layers)
- Final layers are custom: Flatten → Dense → Dropout → Dense → Sigmoid
- Binary classification (`spam` = 1, `ham` = 0)

## ⚙️ Training Pipeline
- Includes data augmentation (random flip & rotation)
- Uses `EarlyStopping`, `ModelCheckpoint`, and `TensorBoard`
- TensorFlow/Keras pipeline with `image_dataset_from_directory`

## 📊 Evaluation
- Accuracy and loss monitored via TensorBoard
- Best model saved to `saved_model/`

## 🚀 Run the project

```bash
pip install -r requirements.txt
python spam_detection.py
```

TensorBoard logs will be saved and accessible using:

```bash
tensorboard --logdir logs/fit
```

## 📦 Requirements
See `requirements.txt`

## 🤖 Author
Fady Abi Rached — 2025

