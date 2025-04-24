import flwr as fl
from flpa.client import FLClient
from flpa.model import CNN
from flpa.dataset import load_partitioned_datasets
from flpa.config import NUM_CLIENTS

client_datasets, test_dataset = load_partitioned_datasets(NUM_CLIENTS)
client_ids = list(client_datasets.keys())


def client_fn(cid: int):
    client_id = client_ids[cid]
    model = CNN()
    train_data = client_datasets[client_id]
    return FLClient(model, train_data, test_dataset).to_client()


if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
    )
