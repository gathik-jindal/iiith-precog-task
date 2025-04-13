import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

IMG_WIDTH = 250
IMG_HEIGHT = 100

EASY_DATASET = "dataset/easy_set"
HARD_DATASET = "dataset/hard_set"


class OCRDataset(Dataset):
    def __init__(self, csv_path, img_folder, words_to_select, resize_to=(IMG_WIDTH, IMG_HEIGHT), normalize=True, char_to_idx=None):
        df = pd.read_csv(csv_path, header=None, names=['image_path', 'text'])

        # Select top N unique words
        unique_words = df['text'].unique()[:words_to_select]
        df_filtered = df[df['text'].isin(unique_words)]

        self.images = []
        self.labels = []
        self.texts = []
        self.resize_to = resize_to
        self.normalize = normalize
        self.img_folder = img_folder

        for _, row in df_filtered.iterrows():
            img_path = os.path.join(img_folder, row['image_path'])
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                exit()

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if self.resize_to:
                img = cv2.resize(img, self.resize_to)
            img = img.astype(np.float32)
            if self.normalize:
                img /= 255.0
            img = np.expand_dims(img, axis=0)  # (1, H, W)

            self.images.append(img)
            self.texts.append(row['text'])

        self.char_to_idx = char_to_idx
        if self.char_to_idx is None:
            raise ValueError("char_to_idx must be provided from combined dataset texts.")

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Convert texts to sequences
        self.labels = [[self.char_to_idx[c] for c in word] for word in self.texts]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def ocr_collate_fn(batch):
    """
    Collate function to pack variable-length sequences for OCR with CTC
    """
    images, labels = zip(*batch)

    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)

    # Assuming CRNN downsamples width by factor of 4 (e.g. W=250 -> output=62)
    input_lengths = torch.full(size=(len(images),), fill_value=images.shape[-1] // 4, dtype=torch.long)

    return images, labels, input_lengths, label_lengths


# Set how many words to select from each dataset
hard_words = 50
easy_words = 100 - hard_words

# Load all texts from both CSVs to build the full character set
def load_texts_from_csv(csv_path, words_to_select):
    df = pd.read_csv(csv_path, header=None, names=['image_path', 'text'])
    unique_words = df['text'].unique()[:words_to_select]
    return list(unique_words)

easy_texts = load_texts_from_csv(EASY_DATASET + ".csv", easy_words)
hard_texts = load_texts_from_csv(HARD_DATASET + ".csv", hard_words)

# Combine texts and build char list
all_texts = easy_texts + hard_texts
unique_chars = sorted(set("".join(all_texts)))
char_list = list(unique_chars)
char_to_index = {c: i + 1 for i, c in enumerate(char_list)}  # 0 reserved for CTC blank

# Now initialize datasets using the shared char_to_index
easy_dataset = OCRDataset(EASY_DATASET + ".csv", EASY_DATASET, easy_words, char_to_idx=char_to_index)
hard_dataset = OCRDataset(HARD_DATASET + ".csv", HARD_DATASET, hard_words, char_to_idx=char_to_index)

# Combine both datasets
full_dataset = ConcatDataset([easy_dataset, hard_dataset])

# Create DataLoader
dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True, collate_fn=ocr_collate_fn)

# Test one batch
images, labels, input_lengths, label_lengths = next(iter(dataloader))
print("Image batch shape:", images.shape)
print("Labels shape:", labels.shape)
print("Input lengths:", input_lengths)
print("Label lengths:", label_lengths)

# Expose for import
__all__ = ["dataloader", "char_list", "char_to_index"]
