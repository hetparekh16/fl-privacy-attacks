from flwr.server.strategy import FedAdam
from flpa.fed_strategies.logging_base import BaseLoggingStrategy
from flpa.utils import save_eval_round


class LoggingFedAdam(BaseLoggingStrategy, FedAdam):
    """Federated Adam with logging capabilities for tracking training and evaluation."""

    def __init__(self, num_rounds: int, **kwargs):
        BaseLoggingStrategy.__init__(self, num_rounds)
        FedAdam.__init__(self, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )
        return self.configure_fit_with_logging(client_instructions)

    def aggregate_fit(self, server_round, results, failures):
        # Log client results
        self.process_fit_results(server_round, results)
        
        # Aggregate as usual
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Process aggregated parameters
        self.process_aggregated_parameters(server_round, aggregated_parameters)

        return aggregated_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # Process client evaluation results
        client_logs = self.process_evaluate_results(server_round, results)
        
        # Aggregate as usual
        agg_loss, agg_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log aggregated metrics
        agg_log = self.log_aggregated_eval(server_round, agg_loss, agg_metrics)
        
        # Save metrics to parquet
        save_eval_round(server_round, client_logs, agg_log)

        return agg_loss, agg_metrics
    


#     How to use this in server_app.py:
#     strategy = LoggingFedAdam(
#     Arguments for your LoggingMixin
#     num_rounds=num_rounds,

#     # Required arguments for the FedAdam optimizer
#     eta=0.1,
#     eta_l=0.1,
#     beta_1=0.9,
#     beta_2=0.999,
#     tau=1e-9,

#     # General strategy arguments
#     fraction_fit=fraction_fit,
#     fraction_evaluate=1.0,
#     min_available_clients=2,
#     initial_parameters=parameters,
#     evaluate_metrics_aggregation_fn=weighted_average,
# )