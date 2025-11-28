# Amazon ML Challenge: High-Performance Pricing Solution

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)

## 📖 Project Overview

This project implements a high-performance pricing solution developed for the Amazon ML Challenge. By combining deep learning-based feature extraction with a powerful **XGBoost** model, we achieved accurate and robust price predictions.

Our approach demonstrates that fusing multimodal embeddings—utilizing models like `all-mpnet-base-v2` for text and `CLIP` for images—provides a rich feature set. This hybrid methodology delivers strong results without the complexity of an end-to-end neural network architecture.

---

## 🏗️ Model Architecture

### 1. Architecture Overview
Our solution utilizes a multi-stage pipeline designed for efficiency and accuracy:

1.  **Parallel Processing:** Text and image data are processed simultaneously through their respective state-of-the-art embedding models.
2.  **Feature Fusion:** The resulting feature vectors are concatenated to form a unified representation.
3.  **Regression:** This combined vector serves as the input for an XGBoost Regressor, which outputs the final price prediction.



[Image of multimodal machine learning model architecture diagram showing text and image inputs feeding into separate encoders then concatenated for XGBoost]


### 2. Model Components

#### 📝 Text Processing Pipeline
* **Preprocessing:** Text is thoroughly cleaned (lowercasing, removal of special characters) before inference.
* **Model:** `sentence-transformers/all-mpnet-base-v2`
* **Output:** Generates a dense vector embedding with a dimension of **768**.

#### 🖼️ Image Processing Pipeline
* **Preprocessing:** * Images are downloaded from source URLs.
    * Resized to standard **224x224** dimensions.
    * Normalized via standard pixel scaling.
* **Model:** `openai/clip-vit-base-patch32`
* **Output:** Generates a visual feature embedding with a dimension of **512**.

#### 🎯 Prediction Model
* **Algorithm:** XGBoost Regressor
* **Input Features:** A fused feature vector of dimension **1280**.
    * *Calculation:* 768 (Text) + 512 (Image) = 1280 dimensions.
