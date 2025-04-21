import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        for epoch in range(1):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        test_loader = DataLoader(self.test_data, batch_size=32)
        correct, total, loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return float(loss), total, {"accuracy": accuracy}
