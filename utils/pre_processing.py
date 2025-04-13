import os
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

IMG_WIDTH = 250
IMG_HEIGHT = 100


def load_data(csv_path, img_folder, words_to_select, resize_to=(IMG_WIDTH, IMG_HEIGHT), normalize=True):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    # Take the first 'words_to_select' unique words
    unique_words = df['text'].unique()[:words_to_select]

    # Filter the dataframe to include only the selected words
    df_filtered = df[df['text'].isin(unique_words)]

    for _, row in df_filtered.iterrows():
        img_path = os.path.join(img_folder, row['image_path'])
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue

        # Read in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        if resize_to:
            img = cv2.resize(img, resize_to)

        img = img.astype(np.float32)
        if normalize:
            img /= 255.0

        # Add channel dimension
        img = np.expand_dims(img, axis=-1)

        images.append(img)
        labels.append(row['text'])

    return np.array(images), labels


# Set how many words you want to take from each set
words_to_select = 50  # Adjust this value as needed

# Load easy + hard sets
x_easy, y_easy = load_data("dataset/easy_set.csv",
                           "dataset/easy_set", words_to_select)
x_hard, y_hard = load_data("dataset/hard_set.csv",
                           "dataset/hard_set", words_to_select)

# Combine easy and hard sets
X = np.concatenate([x_easy, x_hard], axis=0)
y = y_easy + y_hard

# Label encode the selected words
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# We have total of 100 words
y_categorical = to_categorical(y_encoded, num_classes=words_to_select * 2)

print(f"Shape of images: {X.shape}")
print(f"Shape of labels: {y_categorical.shape}")
