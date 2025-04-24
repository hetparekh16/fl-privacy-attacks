import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from loguru import logger
from pydantic import BaseModel, Field
from uuid import uuid4


class Client(BaseModel):
    """Client configuration."""

    client_id: str = Field(
        description="Unique identifier for the client.",
        default=str(uuid4()),
    )


def load_partitioned_datasets(
    num_clients,
) -> tuple[dict[str, Subset], torchvision.datasets.CIFAR10]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    logger.info("Loading CIFAR10 dataset for training and testing")

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # TODO: Use random sampling instead of range-based sampling in the future

    logger.info(f"Partitioning training dataset with {len(trainset)} samples")

    client_data = {}
    data_per_client = len(trainset) // num_clients
    for i in range(num_clients):
        new_client = Client()
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_data[new_client.client_id] = Subset(trainset, indices)

    logger.success(f"Successfully created {len(client_data)} client datasets")

    return client_data, testset
