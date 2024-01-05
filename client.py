import flwr as fl

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_PTH import MobileFaceNet
from loss import ArcMarginProduct
from utils import cal_similarity, EarlyStopping, save_model

from pandas import DataFrame
import numpy as np
from typing import OrderedDict
import argparse
import datetime
import logging
import os
import time
from termcolor import colored
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from collections import defaultdict
# import matplotlib.pyplot as plt
# import math
# import tempfile

#________________________ CLASS ___________________________
# DATA_RATIO = {
#     'train': 0.8,
#     'val': 0,
#     'test': 0.2,
# }
DATA_RATIO = {
    'train': 0.6,
    'val': 0.2,
    'test': 0.2,
}
BATCH_SIZE = 8
CLASSES_NUM = 15
SAVED_CLIENT = './client_saved'

history = defaultdict(lambda: [])
global_parameters = []
#________________________ CLASS ___________________________
class PairImagesDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.num_samples = len(original_dataset)
        self.labels = [label for _, label in original_dataset]

    def __len__(self):
        return self.num_samples * 2  # Two images per pair

    def __getitem__(self, idx):
        idx1 = idx // 2
        label1 = self.labels[idx1]

        is_positive_pair = idx % 2 == 0

        positive_candidates = [i for i, label in enumerate(self.labels) if label == label1]
        negative_candidates = [i for i, label in enumerate(self.labels) if label != label1]

        if is_positive_pair and positive_candidates:
            idx2 = random.choice(positive_candidates)
        elif not is_positive_pair and negative_candidates:
            idx2 = random.choice(negative_candidates)
        else:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        img1, _ = self.original_dataset[idx1]
        img2, _ = self.original_dataset[idx2]

        return img1, img2, torch.tensor(int(label1 == self.labels[idx2]))
    
#________________________ FUNCTION ___________________________
def load_datas():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ])
    augmentation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,saturation=0.1, hue=0.1),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ])
    dataset = ConcatDataset([
        ImageFolder(root = args.face_dataset, transform = transform),
        ImageFolder(root = args.face_dataset, transform = augmentation_transform),
        ImageFolder(root = args.face_dataset, transform = augmentation_transform),
        ImageFolder(root = args.face_dataset, transform = augmentation_transform),
    ])
    # dataset = torch.utils.data.Subset(dataset, list(range(320)))
    
    total = len(dataset)
    num_examples = {"trainset": int(total * DATA_RATIO['train']), 'valset': int(total * DATA_RATIO['val']), "testset": int(total *DATA_RATIO['test'])}

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_examples["trainset"], num_examples['valset'], num_examples["testset"]], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PairImagesDataset(val_dataset), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairImagesDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False)
    print(colored("Train dataset: {}, Val dataset: {}, Test dataset: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)), 'red'))

    # Check image
    # sample_image, sample_label = dataset[0]
    # image_np = sample_image.permute(1, 2, 0).numpy()
    # plt.imshow(image_np)
    # plt.title(f"Class: {sample_label}")
    # plt.show()

    return train_loader, val_loader, test_loader, num_examples

def check_task_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = '_'.join(['Client', args.cid, timestamp])
    if not os.path.exists('log'):
        os.makedirs('log')

    log_path = os.path.join('log', file_name + '.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)

def log(msg):
    logging.info(msg)

def get_features(model, dataloader, device):
    pair_features_list = None
    res_list = None

    model.eval().to(device)

    for _, batch_datas in enumerate(dataloader):
        img1s, img2s, labels = batch_datas

        img1s = img1s.to(device)
        img2s = img2s.to(device)

        features1 = model(img1s).cpu().detach().numpy()
        features2 = model(img2s).cpu().detach().numpy()
        input_features = list(zip(features1, features2))

        pair_features_list = input_features if pair_features_list is None else np.vstack((pair_features_list, input_features))
        res_list = labels.cpu().numpy() if res_list is None else np.hstack((res_list, labels.cpu().numpy()))

    return pair_features_list, res_list

def cal_acc(sim, true, threshold):
    sim, true = np.asarray(sim), np.asarray(true)
    predictions = [1 if score >= threshold else 0 for score in sim]

    precision, recall, fscore, _ = precision_recall_fscore_support(true, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(true, predictions)

    return accuracy, precision, recall, fscore

def eval_performance(threshold, dataloader):
    start_time = time.time()

    model.eval()
    paired, result = get_features(model, dataloader, device)
    input_features_list, target_features_list = zip(*paired)
    # print(input_features_list, target_features_list)

    sim = [cal_similarity(input_features_list[i], target_features_list[i]) for i in range(len(result))]
    sim = np.array(sim)
    # print(sim)

    accuracy, precision, recall, fscore = cal_acc(sim, result, threshold)

    print(colored("Test Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}".format(accuracy, precision, recall, fscore, time.time() - start_time), "magenta"))
    print(colored("-----------------------------------------", "yellow"))

    log("Test Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}".format(accuracy, precision, recall, fscore, time.time() - start_time))
    log("-----------------------------------------")
    return accuracy, precision, recall

# def train_model(ray_config, threshold, epochs):
def train_model(lr, threshold, epochs):
    global global_parameters
    optimizer = SGD([
    {'params': prelu_params_list, 'weight_decay': 0.0},
    {'params': default_params_list, 'weight_decay': 4e-5},
    {'params': model.layers[-1].parameters(), 'weight_decay': 4e-4},
    {'params': model.layers[-2].parameters(), 'weight_decay': 4e-4},
    {'params': arc_loss.weight, 'weight_decay': 4e-4}],
    lr=lr, momentum=0.9, nesterov=True)
    # optimizer = Adam([
    #     {'params': model.parameters()},
    #     {'params': arc_loss.weight}],
    #     lr=0.01)
    
    best_val_acc = 0.0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1, verbose=True)
    early_stopping = EarlyStopping(epochs // 2)

    for epoch in range(epochs):
        print(colored("Epoch {}/{}".format(epoch + 1, epochs), "yellow"))
        log("Epoch {}/{}".format(epoch + 1, epochs))

        start_time = time.time()
        train_loss = 0.0
        train_acc = 0
        epoch_steps = 0
        model.train()
        for _, batch_datas in enumerate(trainloader):

            inputs, labels = batch_datas
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = arc_loss(outputs, labels)
            loss = CrossEntropyLoss()(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            train_acc += preds.eq(labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            epoch_steps += 1

        
        train_loss = train_loss / (epoch_steps)
        train_acc = train_acc / num_examples['trainset']
        print(colored("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss, train_acc, time.time() - start_time), "magenta"))
        print(colored("Iterations per epoch: {}".format(epoch_steps), "magenta"))
        log("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss, train_acc, time.time() - start_time))

        val_acc, precision, recall = eval_performance(threshold, valloader)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            global_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()] + [val.cpu().numpy() for _, val in arc_loss.state_dict().items()]
        history[f'Train_Loss'].append(train_loss)
        history[f'Train_Acc'].append(train_acc)
        history[f'Val_Acc'].append(val_acc)
        history[f'Val_Precision'].append(precision)
        history[f'Val_Recall'].append(recall)

        scheduler.step(1 - val_acc)
        if early_stopping.early_stop(1 - val_acc):
            break

    return train_loss, train_acc

#________________________ START ___________________________
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Embedded devices")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help=f"gRPC server address! (deafault '0.0.0.0:8080')",
    )
    parser.add_argument(
        "--face_dataset",
        type=str,
        default="/face_dataset",
        help=f"Path to data's directory!",
    )
    parser.add_argument(
        "--cid",
        type=str,
        required=True,
        help="Client id. Should be an integer!",
    )
    args = parser.parse_args()

    #________________________ INIT ___________________________
    check_task_folder()
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists(SAVED_CLIENT):
        os.makedirs(SAVED_CLIENT)
    torch.set_default_dtype(torch.float32)

    device = torch.device('cpu')
    model = MobileFaceNet().to(device)
    arc_loss = ArcMarginProduct(128, CLASSES_NUM, 0.5, 64).to(device)

    prelu_params_list = []
    default_params_list = []

    for name, param in model.layers[:-2].named_parameters():
        if 'prelu' in name:
            prelu_params_list.append(param)
        else:
            default_params_list.append(param)

    model_param_num = len(model.state_dict().keys())
    #________________________ DATASET ___________________________
    trainloader, valloader, testloader, num_examples = load_datas()

    #________________________ Federated Learning ____________________________
    class FLWRClient(fl.client.NumPyClient):
        def __init__(self):
            pass

        def get_properties(self, config):
            """Get properties of client."""
            raise Exception("Not implemented")

        def get_parameters(self, config):
            """Get parameters of the local model."""
            raise Exception("Not implemented (server-side parameter initialization)")

        def set_parameters(self, parameters):
            # params_dict = zip(model.state_dict().keys(), parameters)
            params_dict = zip(model.state_dict().keys(), parameters[:model_param_num])
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            params_dict = zip(arc_loss.state_dict().keys(), parameters[model_param_num:])
            state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
            arc_loss.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            print(colored("Round: {}".format(config['server_round']), "yellow"))
            print(colored("-----------------------------------------", "yellow"))

            log("Round: {}".format(config['server_round']))
            log("-----------------------------------------")
            self.set_parameters(parameters)
            if config['best_model_saved']:
                save_model(model, SAVED_CLIENT + '/best_model.pth')

            loss, train_accuracy = train_model(lr=config["lr"], epochs=config["local_epochs"], threshold=config["threshold"])
            test_accuracy, precision, recall = eval_performance(config["threshold"], testloader)
            results = {
                'id': args.cid,
                "loss": loss,
                "train_acc": train_accuracy,
                "test_acc": test_accuracy,
                "precision": precision,
                "recall": recall,
                "epochs": len(history['Train_Acc']),
            }
            # print([val.cpu().numpy() for _, val in model.state_dict().items()])

            return global_parameters, num_examples["trainset"], results
            # return [val.cpu().numpy() for _, val in model.state_dict().items()], num_examples["trainset"], results

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            threshold: float = config["threshold"]
            accuracy, precision, recall = eval_performance(threshold, testloader)
            return 1 - accuracy, num_examples["testset"], {'id': args.cid, "accuracy": float(accuracy), 'precision': float(precision),  'recall': float(recall)}
        
    fl.client.start_numpy_client(server_address=args.server_address, client=FLWRClient())

    # Save the DataFrame to a CSV file
    print(history)
    DataFrame(history).to_csv(f'{SAVED_CLIENT}/{datetime.datetime.now()}-client-{args.cid}.csv', index=False)