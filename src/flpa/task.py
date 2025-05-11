from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from torch.utils.data import Subset
import torchvision
import random


class CNN(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(partition_id: int, num_partitions: int):
    """Manually partition CIFAR-10 using torch Subset for consistent sample IDs"""

    data_root = "./data"
    cifar_folder = os.path.join(data_root, "cifar-10-batches-py")
    if not os.path.exists(cifar_folder):
        raise RuntimeError(
            f"CIFAR-10 dataset not found in '{cifar_folder}'. "
            f"Run `prepare_dataset.py` before starting the simulation."
        )

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )

    # Deterministic partitioning
    total_samples = len(trainset)
    indices = list(range(total_samples))
    random.seed(42)
    random.shuffle(indices)

    data_per_client = total_samples // num_partitions
    start_idx = partition_id * data_per_client
    end_idx = start_idx + data_per_client
    client_indices = indices[start_idx:end_idx]

    # 80-20 train/val split
    split = int(0.8 * len(client_indices))
    train_ids = client_indices[:split]
    test_ids = client_indices[split:]

    train_subset = Subset(trainset, train_ids)
    test_subset = Subset(trainset, test_ids)

    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0

    # Subset provides indices used for this client
    sample_ids = list(trainloader.dataset.indices)

    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    print(
        f"(1) Number of samples used for training this client: {len(sample_ids)} in task.py"
    )
    return avg_trainloss, sample_ids


def test(net, testloader, device):
    """Evaluate model and return loss + full metrics."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    loss = loss / len(testloader)

    return loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
