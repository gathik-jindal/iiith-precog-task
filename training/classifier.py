import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from pre_processor_classifier import dataloader

NUM_CLASSES = 100

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNNClassifier, self).__init__()
        
        self.conv_block = nn.Sequential(   # Dimensions    (B, C, H, W)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # (B, 64, 100, 250)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # (B, 64, 50, 125)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 50, 125)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # (B, 128, 25, 62)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (B, 256, 25, 62)
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (B, 256, 25, 62)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),              # (B, 256, 12, 62)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # (B, 512, 12, 62)
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (B, 512, 12, 62)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1)),              # (B, 512, 6, 62)

            nn.Conv2d(512, 512, kernel_size=2),             # (B, 512, 5, 61)
            nn.ReLU()
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape: (B, 512, 1, 1)

        # Fully connected output layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)         # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)       # Flatten to (B, 512)
        x = self.fc(x)                  # (B, NUM_CLASSES)
        return x


# Ensure using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier(num_classes=100).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

# Compute loss (example)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
