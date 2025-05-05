from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flpa.task import CNN, get_weights
from flwr.server.strategy import FedAvg


class LoggingFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        selected_clients = [client.cid for client, _ in results]
        print(f"\n🔁 [Round {server_round}] Selected clients: {selected_clients}")
        return super().aggregate_fit(server_round, results, failures)


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(CNN())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,  # type: ignore
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)  # type: ignore

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
