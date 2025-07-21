# 🩺 Pneumonia Detection Using CNN

This project is a deep learning-based pneumonia detection system trained on chest X-ray images. It uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify X-ray scans as either **Normal** or **Pneumonia**. The final model is deployed via a user-friendly Streamlit interface.

## 📊 Dataset

- **Training samples**: 5,216  
- **Testing samples**: 624  
- **Validation samples**: 16  
- **Classes**: `Normal`, `Pneumonia`  
- **Image size**: Resized to `32x32` pixels for training efficiency  
- All images were preprocessed using rescaling and augmentation.

## 🧠 Model Architecture

Built using the **Sequential API** with the following layers:

- `Conv2D` – Extracts spatial features from X-ray images
- `MaxPooling2D` – Reduces dimensionality while keeping key info
- `Flatten` – Converts 3D tensors to 1D vectors
- `Dense` – Fully connected layers to learn class boundaries
- `Dropout` – Prevents overfitting during training
- `Dense(1, activation='sigmoid')` – Final output layer for binary classification
- `EarlyStopping` - Monitor `val_loss` and restore best weights once performance stopped improving.


## ✅ Performance

- Final test accuracy: **~83.6%**  
- Validation accuracy peaked at **~81.3%**  
- Model trains over multiple epochs with monitored validation loss to ensure generalization.


## 🖥️ Streamlit Application

A fully interactive web interface built with **Streamlit**, now publicly deployed and accessible to all users.

### Features:

- **Upload your own X-ray image** and get instant predictions
- **Shuffle & Upload Demo Image**: View randomly selected sample X-rays
- **Clear Image**: Resets the view and prediction state
- **Confidence Score**: Displays model's certainty for each prediction

## 🧪 Try It Yourself
```
https://hemlalit-detecting-pneumonia-using-cnn.streamlit.app/
```

## 📮 Contact
Made by Hemlalit Let’s connect and build smarter healthcare together! 🩺📊
