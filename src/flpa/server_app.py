from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flpa.task import CNN, get_weights
from flwr.server.strategy import FedAvg
import hashlib


class LoggingFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        selected_clients = [client.cid for client, _ in results]
        # Log selected clients
        print(f"\n🔁 [Round {server_round}] Selected clients: {selected_clients}")
        print("Now clients will train their models and return the weights...")

        # Log weights hash for each client
        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat_weights = b"".join(w.tobytes() for w in weights)
            weight_hash = hashlib.sha256(flat_weights).hexdigest()[:8]

            print(f"  ↳ Client {client.cid} returned weights hash: {weight_hash}")

        # Aggregate as usual
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Log aggregated weights hash
        if aggregated_parameters is not None:
            agg_weights = parameters_to_ndarrays(aggregated_parameters)
            flat = b"".join(w.tobytes() for w in agg_weights)
            agg_hash = hashlib.sha256(flat).hexdigest()[:8]
            print("First round is completed! Now weights are aggregated...")
            print(
                f"✅ This is the Aggregated weights hash: {agg_hash} for round {server_round}"
            )

        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        print("Now we will send the aggregated model to clients for evaluation...")
        print(f"\n📊 [Round {server_round}] Evaluation results:")

        for client, evaluate_res in results:
            loss = evaluate_res.loss
            accuracy = evaluate_res.metrics.get("accuracy", None)
            print(f"  ↳ Client {client.cid} loss: {loss:.4f}, accuracy: {accuracy:.4f}")

        # Aggregate as usual
        agg_loss, agg_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Print server-aggregated evaluation result
        print(f"✅ [Round {server_round}] Aggregated eval loss: {agg_loss:.4f}")
        if "accuracy" in agg_metrics:
            print(
                f"✅ [Round {server_round}] Aggregated eval accuracy: {agg_metrics['accuracy']:.4f}"
            )

        return agg_loss, agg_metrics


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
