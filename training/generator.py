import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from pre_processor_generator import dataloader, char_list, char_to_index

NUM_CLASSES = 100

class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # (B, 64, 100, 250)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # (B, 64, 50, 125)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # (B, 128, 25, 62)

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),            # (B, 256, 12, 62)

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),            # (B, 512, 6, 62)

            nn.Conv2d(512, 512, 2),         # (B, 512, 5, 61)
            nn.ReLU()
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, 128, bidirectional=True, batch_first=True),
            nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        )

        self.classifier = nn.Linear(256, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        # Input: (B, 1, H, W)
        x = self.cnn(x)  # Output: (B, C, H', W') = (B, 512, 5, 61)

        B, C, H, W = x.size()
        x = x.permute(0, 3, 2, 1)  # (B, W, H, C)
        x = x.contiguous().view(B, W, H * C)  # (B, W, H*C)

        x, _ = self.rnn(x)  # (B, W, 256)
        x = self.classifier(x)  # (B, W, num_classes + 1)
        x = x.permute(1, 0, 2)   # (W, B, num_classes + 1) for CTC loss

        return x




# Ensure using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNNModel(num_classes=NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

ctc_loss = nn.CTCLoss(blank=len(char_list), zero_infinity=True)
model = CRNNModel(num_classes=len(char_list)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels, input_lengths, label_lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # Shape: (T, N, C)
        outputs = outputs.log_softmax(2)  # Required for CTC loss

        # Compute CTC loss
        loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
