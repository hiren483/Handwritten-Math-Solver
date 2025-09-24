# Handwritten Math Solver 📝➕✖️

A PyTorch-based CNN system that recognizes handwritten mathematical equations (digits and operators) from images and evaluates them automatically. Inspired by Photomath, this project combines computer vision with symbolic computation to solve equations in real-time.

---

## Features

- Recognizes **digits (0–9)** and **operators (+, −, ×, ÷, %)** from handwritten images.  
- Segments equations into individual symbols using OpenCV.  
- Reconstructs equations and solves them automatically.  
- Lightweight CNN trained on a 13-class dataset (digits + operators).  
- Can be extended for real-time webcam input.

---

## Dataset

- **Source:** [Mathematics Symbols Dataset](https://www.kaggle.com/datasets/amitamola/mathematics-symbols-data)  
- Contains **digits (0–9) and operators (+, −, ×, ÷, %)** in a single dataset.  
- Used for training the CNN to recognize handwritten math symbols.  


### File Descriptions
- **`model_training.ipynb`** → Preprocesses dataset, builds and trains the CNN, and saves the trained model.  
- **`main.py`** → Loads the trained model, segments symbols from a handwritten equation image, predicts each symbol using the CNN, reconstructs the equation, and evaluates it.  
- **`math_cnn.pth`** → Trained CNN model weights saved after training.  
- **`equation.jpg` & `equation2.jpg`** → Sample handwritten equation images for testing the solver.
