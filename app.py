import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Define a utility class for Smoothed Values for metric tracking
class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.deque = []
        self.window_size = window_size
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"
        self.reset()

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    def median(self):
        d = torch.tensor(self.deque)
        return d.median().item()

    def avg(self):
        d = torch.tensor(self.deque)
        return d.mean().item()

    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median(), avg=self.avg(), global_avg=self.global_avg()
        )


# Utility to plot images with bounding boxes
def plot_image_with_boxes(image, boxes):
    image = image.permute(1, 2, 0).numpy()
    plt.imshow(image)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='r', facecolor='none'))
    plt.show()


# Create the model with Faster R-CNN and MobileNetV2 backbone (for speed)
def create_model(num_classes):
    # Load a pre-trained MobileNetV2 model and use it as a backbone for Faster R-CNN
    backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).features
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


# Training function
def train_model(model, data_loader, optimizer, device):
    model.train()
    loss_stats = defaultdict(SmoothedValue)
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        # Update losses and statistics
        for name, loss in loss_dict.items():
            loss_stats[name].update(loss.item(), images.size(0))

    return loss_stats


# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    loss_stats = defaultdict(SmoothedValue)
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Update losses and statistics
            for name, loss in loss_dict.items():
                loss_stats[name].update(loss.item(), images.size(0))

    return loss_stats


# Main function to run the complete pipeline
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define number of classes (for VOC, it's 21, including background)
    num_classes = 21

    # Load the dataset
    transform = torchvision.transforms.ToTensor()
    dataset = VOCDetection(root="path_to_VOC", year="2012", image_set="train", download=True,
                           transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # Create model
    model = create_model(num_classes)
    model.to(device)

    # Create optimizer (using Adam for simplicity)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss_stats = train_model(model, data_loader, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss_stats['loss'].avg:.4f}")

    # Evaluation loop
    eval_loss_stats = evaluate_model(model, data_loader, device)
    print(f"Evaluation Loss: {eval_loss_stats['loss'].avg:.4f}")

    # Save model
    torch.save(model.state_dict(), "fasterrcnn_model.pth")

    # Test: Plot an image with boxes (showing detection on a sample image)
    sample_image, _ = dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(sample_image)
    plot_image_with_boxes(sample_image[0], prediction[0]['boxes'])


if __name__ == "__main__":
    main()
