import torch

from flwr.client import ClientApp, NumPyClient, Client
from flwr.common import Context
from flpa.task import CNN, get_weights, load_data, set_weights, test, train
import json


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, cid):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.cid = cid

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss, sample_ids = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        sample_ids_str = ",".join(map(str, sample_ids))

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "sample_ids": sample_ids_str,
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, metrics = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context) -> Client:
    net = CNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)  # type: ignore
    local_epochs = context.run_config["local-epochs"]
    cid = context.node_id

    return FlowerClient(net, trainloader, valloader, local_epochs, cid).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
