import hashlib
from datetime import datetime
import pathlib
import torch
from flwr.common import parameters_to_ndarrays

from flpa.utils import save_train_round
from flpa.task import set_weights
from flpa.models import CNN, ResNet, ResNet18WithDropout, CNNWithDropout


class BaseLoggingStrategy:
    """Base class providing logging capabilities for federated learning strategies."""

    def __init__(self, num_rounds: int):
        self.num_rounds = num_rounds

    def configure_fit_with_logging(self, client_instructions):
        """Add logging data to fit instructions."""
        updated_instructions = []
        
        for client, fit_ins in client_instructions:
            # Get current server round from config
            server_round = fit_ins.config.get("server_round", 0)
            
            # Inject round number into config
            fit_ins.config["server_round"] = server_round
            updated_instructions.append((client, fit_ins))
            
        return updated_instructions

    def process_fit_results(self, server_round, results):
        """Process and log training results from clients."""
        selected_clients = [client.cid for client, _ in results]
        print(f"\n[Round {server_round}] Selected clients: {selected_clients}")
        print("Now clients will train their models and return the weights...")

        client_logs = []

        for client, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            flat_weights = b"".join(w.tobytes() for w in weights)
            weight_hash = hashlib.sha256(flat_weights).hexdigest()[:8]
            print(f"  ↳ Client {client.cid} returned weights hash: {weight_hash}")

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
            
        return client_logs

    def process_aggregated_parameters(self, server_round, aggregated_parameters):
        """Process and log aggregated model parameters."""
        if aggregated_parameters is not None:
            agg_weights = parameters_to_ndarrays(aggregated_parameters)
            flat = b"".join(w.tobytes() for w in agg_weights)
            agg_hash = hashlib.sha256(flat).hexdigest()[:8]
            print("First round is completed! Now weights are aggregated...")
            print(
                f"✅ This is the Aggregated weights hash: {agg_hash} for round {server_round}"
            )

            # Save final model
            if server_round == self.num_rounds:
                print("Saving final global model...")
                model = CNNWithDropout()
                set_weights(model, agg_weights)
                pathlib.Path("outputs/global_model").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), "outputs/global_model/global_model.pt")
                print("Global model saved to outputs/global_model/global_model.pt")

    def process_evaluate_results(self, server_round, results):
        """Process and log evaluation results from clients."""
        print("Now we will send the aggregated model to clients for evaluation...")
        print(f"\n[Round {server_round}] Evaluation results:")

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
            
        return client_logs

    def log_aggregated_eval(self, server_round, agg_loss, agg_metrics):
        """Log aggregated evaluation metrics."""
        print(f"[Round {server_round}] Aggregated eval loss: {agg_loss:.4f}")
        agg_log = {
            "round_id": server_round,
            "loss": agg_loss,
            "timestamp": datetime.now(),
        }
        for k, v in agg_metrics.items():
            print(f"[Round {server_round}] Aggregated eval {k}: {v:.4f}")
            agg_log[k] = v
            
        return agg_log