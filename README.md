# AmazonMLChallenge-Test1-

Developed a high-performance pricing solution by combining deep learning-based feature extraction with a powerful XGBoost model. Our approach demonstrates that fusing multimodal embeddings from models like all-mpnet-base-v2 and CLIP provides a rich feature set that enables accurate and robust price prediction. This hybrid methodology proved to be highly effective, delivering strong results without the need for a more complex, end-to-end neural network architecture.


1. Model Architecture
1.1 Architecture Overview
Our solution is a multi-stage pipeline. First, text and image data are processed in parallel through their respective embedding models. The resulting feature vectors are concatenated and then used as input to train an XGBoost Regressor which outputs the final price prediction.
1.2 Model Components
Text Processing Pipeline:

Preprocessing steps: Text is cleaned (e.g., lowercasing, removing special characters) before being fed into the model.

Model type: sentence-transformers/all-mpnet-base-v2

Key parameters: Output embedding dimension of 768.
Image Processing Pipeline:

Preprocessing steps: Images are downloaded from URLs, resized to 224x224, and normalized.

Model type: openai/clip-vit-base-patch32

Key parameters: Output embedding dimension of 512.
Prediction Model:

Model type: XGBoost Regressor

Features Used: Concatenated 768-dim text embeddings and 512-dim image embeddings, resulting in a 1280-dimension feature vector.
