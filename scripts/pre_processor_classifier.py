import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

IMG_WIDTH = 250
IMG_HEIGHT = 100

EASY_DATASET = "../dataset/easy_set"
HARD_DATASET = "../dataset/hard_set"


class TextImageDataset(Dataset):
    def __init__(self, csv_path, img_folder, words_to_select, resize_to=(IMG_WIDTH, IMG_HEIGHT), normalize=True, label_encoder=None):
        # Load CSV without header
        df = pd.read_csv(csv_path, header=None, names=['image_path', 'text'])

        # Select top N unique words
        unique_words = df['text'].unique()[:words_to_select]
        df_filtered = df[df['text'].isin(unique_words)]

        self.images = []
        self.labels = []
        self.img_folder = img_folder
        self.resize_to = resize_to
        self.normalize = normalize

        for _, row in df_filtered.iterrows():
            img_path = os.path.join(img_folder, row['image_path'])
            if not os.path.exists(img_path):
                print(f"Image path not found: {img_path}")
                exit()

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if self.resize_to:
                img = cv2.resize(img, self.resize_to)

            img = img.astype(np.float32)
            if self.normalize:
                img /= 255.0

            # Add channel dim
            img = np.expand_dims(img, axis=0)  # Shape becomes (1, H, W)
            self.images.append(img)
            self.labels.append(row['text'])

        # Label encoding
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels)
        else:
            self.label_encoder = label_encoder
            self.label_encoder.fit(self.labels)

        self.labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# Set how many words to select from each dataset
hard_words = 50
easy_words = 100 - hard_words

# First, load easy to get a common label encoder
easy_dataset = TextImageDataset(EASY_DATASET + ".csv", EASY_DATASET, easy_words)
label_encoder = easy_dataset.label_encoder

# Then use the same encoder for hard
hard_dataset = TextImageDataset(HARD_DATASET + ".csv", HARD_DATASET, hard_words, label_encoder=label_encoder)

# Combine datasets
from torch.utils.data import ConcatDataset
full_dataset = ConcatDataset([easy_dataset, hard_dataset])

# Optional: create DataLoader
dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True)

# Test one batch
images, labels = next(iter(dataloader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")