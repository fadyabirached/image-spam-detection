# 📸 Image Spam Detection with Transfer Learning

This is a complete image classification pipeline in Python that detects **spam images** (e.g., clickbait, phishing ads) versus **non-spam** content using a frozen **VGG-19** CNN model.

## 🧠 Project Features

- ✅ Automatically removes corrupted and duplicate files
- ✅ Uses VGG-19 pretrained on ImageNet
- ✅ Applies basic image augmentation
- ✅ Evaluates results using Precision, Recall, F1-score & Confusion Matrix
- ✅ Saves the trained model in both `.h5` and SavedModel formats

## 🚀 How to Use

1. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pillow matplotlib scikit-learn

Note: Dataset is private.