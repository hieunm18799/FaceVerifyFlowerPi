import flwr as fl
from flwr.common import FitRes, Parameters, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.history import History
from flwr.server.client_manager import ClientManager
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log

import torch
# from torchsummary import summary
# import numpy as np

from utils import EarlyStopping, save_model
from model_PTH import MobileFaceNet, INPUT_SHAPE
from loss import ArcMarginProduct

from pandas import DataFrame
import argparse
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union, OrderedDict
from collections import defaultdict
import datetime
import os
import timeit

#________________________ VARIABLES ___________________________
NUMBER_CLIENTS = 3
CLASSES_NUM = 36
THRESHOLD = 0.9
ROUNDS = 10
BEGIN_LR = 0.0005

global_accuracy = 0.0
devices_history = defaultdict(lambda: [])
best_model_saved = False
#________________________ CLASS ___________________________
class FlServer(fl.server.Server):
    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        early_stopper: EarlyStopping,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.early_stopper = early_stopper

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = super()._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = super().fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = super().evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                # Apply early stopping after the result of test on all clients
                if self.early_stopper.early_stop(loss_fed):
                    return history

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
    
class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        global cur_parameters

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        for _, r in results:
            devices_history[f'{r.metrics["id"]}-Val-Accuracy'].append(r.metrics["best_val_acc"])
            devices_history[f'{r.metrics["id"]}-Test-Accuracy'].append(r.metrics["test_acc"])
            devices_history[f'{r.metrics["id"]}-Test-TN'].append(r.metrics["tn"])
            devices_history[f'{r.metrics["id"]}-Test-FP'].append(r.metrics["fp"])
            devices_history[f'{r.metrics["id"]}-Test-FN'].append(r.metrics["fn"])
            devices_history[f'{r.metrics["id"]}-Test-TP'].append(r.metrics["tp"])
            devices_history[f'{r.metrics["id"]}-Epochs'].append(r.metrics["epochs"])
        cur_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        params_dict = zip(model.state_dict().keys(), cur_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        save_model(model, 'saved_models/{}.pth'.format(server_round))

        return aggregated_parameters, aggregated_metrics
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""
        global cur_parameters, global_accuracy, best_model_saved
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)
        devices_history['Avg-Aggregated-Test-Accuracy'].append(aggregated_accuracy)
        for _, r in results:
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-Accuracy'].append(r.metrics["accuracy"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-TN'].append(r.metrics["tn"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-FP'].append(r.metrics["fp"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-FN'].append(r.metrics["fn"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-TP'].append(r.metrics["tp"])
        temp_tp = sum([r.metrics['tp'] for _, r in results])
        devices_history[f'Avg-Aggregated-Test-Precision'].append(temp_tp / (temp_tp + sum([r.metrics['fp'] for _, r in results])))
        devices_history[f'Avg-Aggregated-Test-Recall'].append(temp_tp / (temp_tp + sum([r.metrics['fn'] for _, r in results])))

        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        if aggregated_accuracy > global_accuracy:
            global_accuracy = aggregated_accuracy
            # print(cur_parameters)
            params_dict = zip(model.state_dict().keys(), cur_parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            save_model(model, 'saved_models/best_model.pth')
            best_model_saved = True
        else:
            best_model_saved = False

        return aggregated_loss, {"accuracy": aggregated_accuracy}

#________________________ FUNCTION ___________________________
def fit_config(server_round: int):
    config = {
        # "local_epochs": 3 if global_accuracy > 0.8 else 5,
        'best_model_saved': best_model_saved,
        "local_epochs": 5,
        "lr": BEGIN_LR // 10 if global_accuracy > 0.8 else BEGIN_LR,
        "threshold": args.threshold,
        "server_round": server_round,
    }
    return config

def evaluate_config(server_round: int):
    config = {
        "threshold": args.threshold,
    }
    return config

#________________________ START ___________________________
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--rounds",
    type=int,
    default=ROUNDS,
    help=f"The number of rounds! (deafault 10)",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=THRESHOLD,
    help=f"Set the threshold! (deafault 0.9)",
)
args = parser.parse_args()

device = torch.device('cpu')
model = MobileFaceNet().to(device)
arc_loss = ArcMarginProduct(128, CLASSES_NUM, 0.5, 64).to(device)

# model_path = './model_PTH/best_model.pth'
# state_dict = torch.load(model_path, map_location='cpu')
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     new_state_dict[k.replace("module.", "")] = v
# model.load_state_dict(new_state_dict)
# summary(model, INPUT_SHAPE)

#________________________ FEDERATED LEARNING ___________________________
# Create strategy
strategy = CustomStrategy(
    fraction_fit = 1.0,
    fraction_evaluate = 1.0,
    min_fit_clients = NUMBER_CLIENTS,
    min_evaluate_clients = NUMBER_CLIENTS,
    min_available_clients = NUMBER_CLIENTS,
    on_fit_config_fn = fit_config,
    on_evaluate_config_fn = evaluate_config,
    initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()] + [val.cpu().numpy() for _, val in arc_loss.state_dict().items()]),
    # initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()]),
)

fl.server.start_server(
    server=FlServer(
        client_manager=SimpleClientManager(),
        early_stopper=EarlyStopping(3),
        strategy=strategy,
    ),
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds = args.rounds),
    strategy=strategy,
# Start Flower server (SSL-enabled) for four rounds of federated learning
    # certificates=(
    #     Path(".cache/certificates/ca.crt").read_bytes(),
    #     Path(".cache/certificates/server.pem").read_bytes(),
    #     Path(".cache/certificates/server.key").read_bytes(),
    # ),
)

# Save the DataFrame to a CSV file
# print(devices_history)
DataFrame(devices_history).to_csv(f'result/{datetime.datetime.now()}-server.csv', index=False)