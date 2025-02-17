# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import cv2  

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model and calculate overall accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy on the full test set: {test_acc:.2f}%")

### Condition 1: Predict Only '9' Images from the Dataset ###
def run_condition_1():
    """Predict digit '9' from the dataset and display only correctly identified '9' images."""
    digit_9_indices = np.where(test_labels == 9)[0]
    digit_9_images = test_images[digit_9_indices]
    digit_9_labels = test_labels[digit_9_indices]

    # Make predictions for digit '9'
    predictions = model.predict(digit_9_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Ensure only correctly predicted '9' images are displayed
    correct_9_indices = np.where(predicted_classes == 9)[0]
    correct_9_images = digit_9_images[correct_9_indices]
    correct_9_labels = digit_9_labels[correct_9_indices]

    # Calculate accuracy for predicting digit '9' correctly
    accuracy_for_9 = (len(correct_9_images) / len(digit_9_labels)) * 100
    print(f"Accuracy for predicting digit '9' correctly: {accuracy_for_9:.2f}%")

    # Display only correctly predicted '9' images
    plt.figure(figsize=(10, 10))
    num_samples = min(9, len(correct_9_images))  # Display up to 9 images
    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        plt.imshow(correct_9_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"P: {predicted_classes[correct_9_indices[i]]}")
        plt.axis('off')

    plt.suptitle("Correct Predictions for Digit '9'", fontsize=16)
    plt.show()

# Run Condition 1
run_condition_1()

### Condition 2: Drawing Pad with Enhanced Detection ###
def preprocess_image(image):
    """Preprocess the drawn image for prediction with multiple checks."""
    image = image.convert('L') 
    image = ImageOps.invert(image)  
    image_array = np.array(image)

    # Apply binary thresholding to remove noise
    _, binary_image = cv2.threshold(image_array, 100, 255, cv2.THRESH_BINARY)

    # Find contours to detect individual digits
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("The digit cannot be identified. It might not be drawn well.")
        return None

    if len(contours) > 1:
        print("The input contains multiple digits. Please draw a single digit.")
        return None

    # Calculate the bounding box of the digit
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Crop the digit and add padding
    cropped_image = binary_image[y:y+h, x:x+w]
    padded_image = cv2.copyMakeBorder(cropped_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

    # Resize to 28x28 while maintaining aspect ratio
    resized_image = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_AREA)
    normalized_image = resized_image / 255.0  # Normalize the image

    return normalized_image.reshape(1, 28, 28, 1)

def predict_drawn_digit(image):
    """Predict the digit using the trained CNN model."""
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return

    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    print(f"Predicted class: {predicted_class}")

    if predicted_class == 9:
        print("The drawn digit is a 9!")
    else:
        print(f"The drawn digit is not a 9; it is a {predicted_class}.")

def create_drawing_pad():
    """Create a Tkinter drawing pad for digit prediction."""
    root = Tk()
    root.title("Draw a digit and predict it")
    root.geometry("300x400")
    canvas = Canvas(root, width=280, height=280, bg="white")
    canvas.pack()

    image = Image.new("RGB", (280, 280), "white")
    draw = ImageDraw.Draw(image)

    def paint(event):
        x, y = event.x, event.y
        canvas.create_oval(x, y, x + 8, y + 8, fill='black', outline='black')
        draw.ellipse([x, y, x + 8, y + 8], fill='black')

    def clear_canvas():
        """Clear the canvas and reset the image."""
        canvas.delete("all")
        draw.rectangle((0, 0, 280, 280), fill="white")

    def predict_digit():
        """Predict the digit drawn on the canvas."""
        predict_drawn_digit(image)

    canvas.bind("<B1-Motion>", paint)
    Button(root, text="Predict", command=predict_digit).pack(side=LEFT, padx=10)
    Button(root, text="Clear", command=clear_canvas).pack(side=RIGHT, padx=10)
    root.mainloop()

# Launch the drawing pad for Condition 2
create_drawing_pad()
