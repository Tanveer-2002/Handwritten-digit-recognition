import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

data = np.load('E:\Git\Handwritten-digit-recognition\with_raw_image\model_params.npz')

w1 = data["w1"]
b1 = data["b1"]
w2 = data["w2"]
b2 = data["b2"]
w3 = data["w3"]
b3 = data["b3"]
w4 = data["w4"]
b4 = data["b4"]


def prepro(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       x, y, w, h = cv2.boundingRect(largest_contour)
       print(x,y,w,h)

       # Crop to Bounding Box
       cropped_digit = thresh[y:y+h, x:x+w]

       # Making the Image Square by Adding Padding
       height, width = cropped_digit.shape
       padding = abs(height - width) // 2

       if height > width:
           padded_digit = cv2.copyMakeBorder(cropped_digit, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
       else:
           padded_digit = cv2.copyMakeBorder(cropped_digit, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

       final_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
       return final_digit
    else:
       print("No digit found in the image.")

def relu(x):
    return np.maximum(0,x)

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict_digit(img_path):
       
    final_digit = np.array(prepro(img_path))
    final_digit = final_digit.reshape(1,784)/255.0
    

    z1 = np.dot(final_digit, w1) + b1
    r1 = relu(z1)
       
    z2 = np.dot(r1, w2) + b2
    r2 = relu(z2)

    z3 = np.dot(r2, w3) + b3
    r3 = relu(z3)

    z4 = np.dot(r3, w4) + b4
    output = softmax(z4)

    predicted_digit = np.argmax(output)
    return predicted_digit

# GUI Code
class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("500x500")

        # Frame for Image Display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # Result Label
        self.result_label = tk.Label(root, text="Select an image to predict", font=("Arial", 16))
        self.result_label.pack(pady=10)

        # Button to choose image
        choose_btn = tk.Button(root, text="Choose Image", command=self.load_image)
        choose_btn.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            # Show the image
            img = Image.open(file_path).resize((200, 200))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)

            # Predict digit
            predicted = predict_digit(file_path)
            self.result_label.config(text=f"Predicted Digit: {predicted}")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()

