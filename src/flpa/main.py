import flwr as fl
from flpa.client import FLClient
from flpa.model import CNN
from flpa.dataset import load_partitioned_datasets
from typing import Union

NUM_CLIENTS = 5


def client_fn(cid: Union[str, int]):
    cid = int(cid)
    train_datasets, test_dataset = load_partitioned_datasets(NUM_CLIENTS)
    model = CNN()
    train_data = train_datasets[cid]
    return FLClient(model, train_data, test_dataset).to_client()


if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
    )
