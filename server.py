import flwr as fl
from flwr.common import FitRes, Parameters, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy

import torch
# from torchsummary import summary
import pandas as pd

from model_PTH import MobileFaceNet, INPUT_SHAPE
from loss import ArcMarginProduct

# import numpy as np
from typing import Dict, List, Optional, Tuple, Union, OrderedDict
from termcolor import colored
from collections import defaultdict
import datetime
import os

#________________________ VARIABLES ___________________________
NUMBER_CLIENTS = 3
CLASSES_NUM = 15
THRESHOLD = 0.9

global_accuracy = 0.0
devices_history = defaultdict(lambda: [])
#________________________ CLASS ___________________________
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
            devices_history[f'{r.metrics["id"]}-Train-Loss'].append(r.metrics["loss"])
            devices_history[f'{r.metrics["id"]}-Train-Accuracy'].append(r.metrics["train_acc"])
            devices_history[f'{r.metrics["id"]}-Test-Accuracy'].append(r.metrics["test_acc"])
            devices_history[f'{r.metrics["id"]}-Test-TPR'].append(r.metrics["tpr"])
            devices_history[f'{r.metrics["id"]}-Test-FPR'].append(r.metrics["fpr"])
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
        global cur_parameters, global_accuracy
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        for _, r in results:
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-Accuracy'].append(r.metrics["accuracy"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-TPR'].append(r.metrics["tpr"])
            devices_history[f'{r.metrics["id"]}-Aggregated-Test-FPR'].append(r.metrics["fpr"])
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]


        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        if aggregated_accuracy > global_accuracy:
            global_accuracy = aggregated_accuracy
            # print(cur_parameters)
            params_dict = zip(model.state_dict().keys(), cur_parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            save_model(model, 'saved_models/best_model.pth')

        return aggregated_loss, {"accuracy": aggregated_accuracy}

#________________________ FUNCTION ___________________________
def save_model(model, folder_path):
    torch.save(model.state_dict(), folder_path)
    print(colored("Model saved to %s" % folder_path, "red"))

#________________________ START ___________________________
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')    
#________________________ FEDERATED LEARNING ___________________________
def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        # "local_epochs": 3 if global_accuracy > 0.9 else 5,
        "local_epochs": 3,
        "threshold": THRESHOLD,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    """
    config = {
        # "test_steps": 5 if server_round < 4 else 10,
        "threshold": THRESHOLD,
        }
    return config

device = torch.device('cpu')
model = MobileFaceNet().to(device)
arc_loss = ArcMarginProduct(128, CLASSES_NUM + 1, 0.5, 64).to(device)

# model_path = './model_PTH/best_model.pth'
# state_dict = torch.load(model_path, map_location='cpu')
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     new_state_dict[k.replace("module.", "")] = v
# model.load_state_dict(new_state_dict)
# summary(model, INPUT_SHAPE)

# Create strategy
strategy = CustomStrategy(
    fraction_fit = 1.0,
    fraction_evaluate = 1.0,
    min_fit_clients = NUMBER_CLIENTS,
    min_evaluate_clients = NUMBER_CLIENTS,
    min_available_clients = NUMBER_CLIENTS,
    # eta=0.001,
    on_fit_config_fn = fit_config,
    on_evaluate_config_fn = evaluate_config,
    initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()] + [val.cpu().numpy() for _, val in arc_loss.state_dict().items()]),
    # initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()]),
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds = 10),
    strategy=strategy,
# Start Flower server (SSL-enabled) for four rounds of federated learning
    # certificates=(
    #     Path(".cache/certificates/ca.crt").read_bytes(),
    #     Path(".cache/certificates/server.pem").read_bytes(),
    #     Path(".cache/certificates/server.key").read_bytes(),
    # ),
)

print(devices_history)
df = pd.DataFrame(devices_history)

# Save the DataFrame to a CSV file
df.to_csv(f'result/{datetime.datetime.now()}.csv', index=False)