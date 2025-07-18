import torchvision
from torchvision import transforms


def prepare_cifar10():
    print("⬇CIFAR-10 not found. Downloading...")

    transform = transforms.ToTensor()

    # Download both train and test splits
    torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    print("CIFAR-10 downloaded successfully.")


if __name__ == "__main__":
    prepare_cifar10()
