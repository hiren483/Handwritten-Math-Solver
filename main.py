import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import sympy as sp

# --------------------------
# Step 1: Model Definition (same as training)
# --------------------------
class MathCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(MathCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self._to_linear = None
        self._dummy_forward()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _dummy_forward(self):
        x = torch.randn(1,1,28,28)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------
# Step 2: Load Model
# --------------------------
device = torch.device("cpu")
model = MathCNN(num_classes=14)
model.load_state_dict(torch.load("math_cnn.pth", map_location=device))
model.eval()

# --------------------------
# Step 3: Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------------
# Step 4: Robust character segmentation
# --------------------------
def segment_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Adaptive thresholding (better for messy handwriting)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 8)
    
    # Optional: remove small noise
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Vertical projection
    vertical_sum = np.sum(thresh, axis=0)
    min_width = 5  # minimum width of a character
    chars = []
    start = None

    for i, val in enumerate(vertical_sum):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > min_width:
                char_img = thresh[:, start:i]
                char_img = cv2.resize(char_img, (28, 28))
                chars.append(char_img)
            start = None
    # handle last character
    if start is not None:
        char_img = thresh[:, start:]
        char_img = cv2.resize(char_img, (28, 28))
        chars.append(char_img)

    return chars

# --------------------------
# Step 5: Predict characters
# --------------------------
class_labels = ["0","1","2","3","4","5","6","7","8","9","+","-","*","/"]

def predict_character(img):
    pil_img = Image.fromarray(img)
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return class_labels[predicted.item()]

# --------------------------
# Step 6: Main Solver
# --------------------------
def solve_equation(image_path):
    chars = segment_characters(image_path)
    if not chars:
        print("No characters detected.")
        return
    
    equation = "".join([predict_character(c) for c in chars])
    print("Equation:", equation)
    
    try:
        result = sp.sympify(equation).evalf()
        print("Result:", result)
    except Exception as e:
        print("Error solving equation:", e)

# --------------------------
# Run Example
# --------------------------
if __name__ == "__main__":
    solve_equation("equation.jpg")  # replace with your handwritten image
    solve_equation("equation2.jpg")
