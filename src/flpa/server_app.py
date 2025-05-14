from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flpa.task import CNN, get_weights, set_weights
from flwr.server.strategy import FedAvg
import hashlib
from flpa.utils import save_eval_round, save_train_round
from datetime import datetime
import pathlib
import torch
from flpa.utils import clear_output_directory


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


class LoggingFedAvg(FedAvg):

    def __init__(self, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds


    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        updated_instructions = []

        for client, fit_ins in client_instructions:
            # Inject round number into config
            fit_ins.config["server_round"] = server_round
            updated_instructions.append((client, fit_ins))

        return updated_instructions

    def aggregate_fit(self, server_round, results, failures):
        selected_clients = [client.cid for client, _ in results]
        print(f"\n🔁 [Round {server_round}] Selected clients: {selected_clients}")
        print("Now clients will train their models and return the weights...")

        client_logs = []

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat_weights = b"".join(w.tobytes() for w in weights)
            weight_hash = hashlib.sha256(flat_weights).hexdigest()[:8]
            print(f"  ↳ Client {client.cid} returned weights hash: {weight_hash}")

            # sample_ids = fit_res.metrics.get("sample_ids")

            print(
                f"  ↳ Client {client.cid} used the sample_ids: {len(fit_res.metrics.get('sample_ids'))} for training before sending it to saving function in  server_app. py"  # type: ignore
            )
            save_train_round(
                round_id=server_round,
                client_id=client.cid,
                sample_ids=fit_res.metrics.get("sample_ids"),
            )

            log_entry = {
                "round_id": server_round,
                "client_id": str(client.cid),
                "train_loss": fit_res.metrics.get("train_loss"),
                "timestamp": datetime.now(),
            }

            client_logs.append(log_entry)

        # Aggregate as usual
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            agg_weights = parameters_to_ndarrays(aggregated_parameters)
            flat = b"".join(w.tobytes() for w in agg_weights)
            agg_hash = hashlib.sha256(flat).hexdigest()[:8]
            print("First round is completed! Now weights are aggregated...")
            print(
                f"✅ This is the Aggregated weights hash: {agg_hash} for round {server_round}"
            )

            # Remove the hardcoded server_round value
            # and replace it with the server_round variable
            if server_round == self.num_rounds:
                print("💾 Saving final global model...")
                model = CNN()
                set_weights(model, agg_weights)
                pathlib.Path("outputs").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "outputs/global_model/global_model.pt")
                print("✅ Global model saved to outputs/global_model/global_model.pt")

        return aggregated_parameters, metrics

    # inside LoggingFedAvg class
    def aggregate_evaluate(self, server_round, results, failures):
        print("Now we will send the aggregated model to clients for evaluation...")
        print(f"\n📊 [Round {server_round}] Evaluation results:")

        # Prepare data for logging
        client_logs = []
        for client, evaluate_res in results:
            metrics = evaluate_res.metrics
            metric_str = f"  ↳ Client {client.cid} loss: {evaluate_res.loss:.4f}"
            log_entry = {
                "round_id": server_round,
                "client_id": str(client.cid),
                "loss": evaluate_res.loss,
                "timestamp": datetime.now(),
            }
            for k, v in metrics.items():
                metric_str += f", {k}: {v:.4f}"
                log_entry[k] = v
            print(metric_str)
            client_logs.append(log_entry)

        # Aggregate as usual
        agg_loss, agg_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        print(f"✅ [Round {server_round}] Aggregated eval loss: {agg_loss:.4f}")
        agg_log = {
            "round_id": server_round,
            "loss": agg_loss,
            "timestamp": datetime.now(),
        }
        for k, v in agg_metrics.items():
            print(f"✅ [Round {server_round}] Aggregated eval {k}: {v:.4f}")
            agg_log[k] = v

        # Save metrics to parquet
        save_eval_round(server_round, client_logs, agg_log)

        return agg_loss, agg_metrics


def server_fn(context: Context):

    clear_output_directory()

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(CNN())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = LoggingFedAvg(
        num_rounds=num_rounds, # type: ignore
        fraction_fit=fraction_fit,  # type: ignore
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,  # type: ignore
    )
    config = ServerConfig(num_rounds=num_rounds)  # type: ignore

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
