import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


def load_partitioned_datasets(num_clients):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # TODO: Use ramdom sampling instead of range based sampling in the future

    client_data = []
    data_per_client = len(trainset) // num_clients
    for i in range(num_clients):
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_data.append(Subset(trainset, indices))

    return client_data, testset
