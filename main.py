import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#CONFIG
DATASET_PATH = "dataset"
IMG_SIZE     = 128 #resize all images to 128x128
CLASSES      = ["glioma", "meningioma", "pituitary", "notumor"]

def load_data(split="Training"):
    images = []
    labels = []

    for label_index, class_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, split, class_name)

        print(f"Loading {class_name}...")

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)

            #Read Grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            #Normalize pixel values (0-255->0-1)
            img = img / 255.0

            images.append(img)
            labels.append(label_index)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

X_train, y_train = load_data("Training")
X_test,  y_test  = load_data("Testing")

print(f"\nTraining images shape : {X_train.shape}") #(5600, 128, 128)
print(f"Training labels shape : {y_train.shape}") #(5600,)
print(f"Testing  images shape : {X_test.shape}") #(1600, 128, 128)