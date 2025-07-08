from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flpa.task import get_weights
from flpa.utils import clear_output_directory
from flpa.fed_strategies.logging_fedavg import LoggingFedAvg
from flpa.fed_strategies.logging_fedadam import LoggingFedAdam
from flpa.models import CNN, ResNet

def weighted_average(metrics_list):
    total = sum(num_examples for num_examples, _ in metrics_list)
    result = {}

    for key in ["accuracy", "precision", "recall", "f1"]:
        weighted_sum = sum(
            metrics.get(key, 0.0) * num_examples
            for num_examples, metrics in metrics_list
            if key in metrics
        )
        result[key] = weighted_sum / total if total > 0 else 0.0

    return result


def server_fn(context: Context):

    clear_output_directory()

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(ResNet())
    parameters = ndarrays_to_parameters(ndarrays)

    # strategy = LoggingFedAvg(
    #     num_rounds=num_rounds, # type: ignore = 14
    #     fraction_fit=fraction_fit,  # type: ignore
    #     fraction_evaluate=1.0,
    #     min_available_clients=2,
    #     initial_parameters=parameters,
    #     evaluate_metrics_aggregation_fn=weighted_average,  # type: ignore
    # )

    strategy = LoggingFedAdam(
        num_rounds=num_rounds, # type: ignore = 40
        eta=0.01,
        eta_l=0.1,
        beta_1=0.9,
        beta_2=0.999,
        tau=1e-9,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)  # type: ignore

    return ServerAppComponents(strategy=strategy, config=config) # type: ignore


app = ServerApp(server_fn=server_fn)
