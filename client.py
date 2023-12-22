# pip install torch torchvision flwr==1.6.0
import flwr as fl
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

from model_PTH import MobileFaceNet, INPUT_SHAPE
from loss import ArcMarginProduct

from typing import OrderedDict
import numpy as np
import argparse
import datetime
import logging
import os
import time
# import random
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
# import math

# Define your data directory
BATCH_SIZE = 8
CLASSES_NUM = 15

#________________________ FUNCTION ___________________________
# Custom generator function
# def custom_generator(generator):
#     for (image_batch, label_batch) in generator:
#         yield ((image_batch, label_batch), label_batch)

# def collate_fn(batch):
#     datas, labels = zip(*batch)
#     list1 = []
#     list2 = []
#     c1 = []

#     for label, data in zip(labels, datas):
#         for paired_label, paired_data in zip(labels, datas):
#                 list1.append(data)
#                 list2.append(paired_data)
#                 c1.append(label == paired_label)
#     return torch.stack(list1), torch.stack(list2), torch.tensor(c1)

def load_datas():
    transform = transforms.Compose([
        transforms.CenterCrop((280, 240)),
        transforms.Resize(INPUT_SHAPE[1:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ])
    augmentation_transform = transforms.Compose([
        transform,
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])
    train_dataset = ConcatDataset([
        ImageFolder(root = args.train_dataset, transform = transform),
        # ImageFolder(root = args.dataset, transform = augmentation_transform),
    ])
    test_dataset = ConcatDataset([
        ImageFolder(root = args.test_dataset, transform = transform),
        # ImageFolder(root = args.dataset, transform = augmentation_transform),
    ])

    # dataset = torch.utils.data.Subset(dataset, list(range(320)))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(colored("Train dataset: {}, Test dataset: {}".format(len(train_dataset), len(test_dataset))), 'red')

    # Check image
    # index = 0
    # sample_image, sample_label = dataset[index]
    # print(sample_image.shape)

    # # Convert the PyTorch tensor to a NumPy array and transpose the channels
    # image_np = sample_image.permute(1, 2, 0).numpy()

    # # Display the image using Matplotlib
    # plt.imshow(image_np)
    # plt.title(f"Class: {sample_label}")
    # plt.show()

    num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset)}

    return train_loader, test_loader, num_examples

def check_task_folder():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = '_'.join(['Client', args.cid, timestamp])
    if not os.path.exists('log'):
        os.makedirs('log')

    log_path = os.path.join('log', file_name + '.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)

def log(msg):
    logging.info(msg)

# Cosine distance
def cal_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_features(model, dataloader, device):
    input_features_list = None
    labels_list = None

    model.eval().to(device)

    for _, batch_datas in enumerate(dataloader):
        input_datas, labels = batch_datas

        input_datas = input_datas.to(device)

        input_features = model(input_datas).cpu().detach().numpy()

        input_features_list = input_features if input_features_list is None else np.vstack((input_features_list, input_features))
        labels_list = labels.cpu().numpy() if labels_list is None else np.hstack((labels_list, labels.cpu().numpy()))

    return [(feature1, feature2) for idx, feature1 in enumerate(input_features_list) for feature2 in input_features_list[idx + 1:]], [1 if label1 == label2 else 0 for idx, label1 in enumerate(labels_list) for label2 in labels_list[idx + 1:]]

def cal_acc(sim, labels, threshold):
    sim, labels = np.asarray(sim), np.asarray(labels)
    predictions = [1 if score >= threshold else 0 for score in sim]

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = accuracy_score(labels, predictions)

    print("Confusion Matrix:")
    print(cm)

    return accuracy, tpr, fpr, cm

def eval_performance(threshold: float):
    start_time = time.time()

    model.eval()
    paired, result = get_features(model, testloader, device)
    input_features_list, target_features_list = zip(*paired)
    # print(input_features_list, target_features_list)

    sim = [cal_similarity(input_features_list[i], target_features_list[i]) for i in range(len(result))]
    sim = np.array(sim)
    # print(sim)

    accuracy, tpr, fpr, _ = cal_acc(sim, result, threshold)

    print(colored("Test Acc: {:.4f} TPR: {:.4f} FPR: {:.4f} Time: {:.4f}".format(accuracy, tpr, fpr, time.time() - start_time), "magenta"))
    print(colored("-----------------------------------------", "yellow"))

    log("Test Acc: {:.4f} TPR: {:.4f} FPR: {:.4f} Time: {:.4f}".format(accuracy, tpr, fpr, time.time() - start_time))
    log("-----------------------------------------")
    return accuracy, tpr, fpr

def train(epochs):
    # epochs_freeze= epochs // 3
    for epoch in range(epochs):
        print(colored("Epoch {}/{}".format(epoch + 1, epochs), "yellow"))
        log("Epoch {}/{}".format(epoch + 1, epochs))

        train_loss = 0.0
        train_correct = 0
        train_data_num = 0

        # if epoch < epochs_freeze:
        #     for param in model.parameters():
        #         param.requires_grad = False
        # else:
        #     for param in model.parameters():
        #         param.requires_grad = True
        start_time = time.time()

        model.train()
        for _, batch_datas in enumerate(trainloader):
            inputs, labels = batch_datas
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = arc_loss(outputs, labels)
            loss = CrossEntropyLoss()(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            train_correct += preds.eq(labels).sum().item()

            # for param in model.parameters():
            #     param.grad = None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_data_num += BATCH_SIZE

        
        train_loss = train_loss / (train_data_num / BATCH_SIZE)
        train_correct = train_correct / train_data_num
        print(colored("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss, train_correct, time.time() - start_time), "magenta"))
        print(colored("Iterations per epoch: {}".format(train_data_num // BATCH_SIZE), "magenta"))
        log("Train Loss: {:.4f} Acc: {:.4f} Time: {:.4f}".format(train_loss, train_correct, time.time() - start_time))

    return train_loss, train_correct, train_data_num

#________________________ START ___________________________
parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address! (deafault '0.0.0.0:8080')",
)
parser.add_argument(
    "--train_dataset",
    type=str,
    default="/dataset/train",
    help=f"Path to data's directory!",
)
parser.add_argument(
    "--test_dataset",
    type=str,
    default="/dataset/val",
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
torch.set_default_dtype(torch.float32)

device = torch.device('cpu')
model = MobileFaceNet().to(device)
arc_loss = ArcMarginProduct(128, CLASSES_NUM + 1, 0.5, 64).to(device)

# prelu_params_list = []
# default_params_list = []

# for name, param in model.layers[:-2].named_parameters():
#     if 'prelu' in name:
#         prelu_params_list.append(param)
#     else:
#         default_params_list.append(param)
# optimizer = SGD([
#     {'params': prelu_params_list, 'weight_decay': 0.0},
#     {'params': default_params_list, 'weight_decay': 4e-5},
#     {'params': model.layers[-1].parameters(), 'weight_decay': 4e-4},
#     {'params': model.layers[-2].parameters(), 'weight_decay': 4e-4},
#     {'params': arc_loss.weight, 'weight_decay': 4e-4}],
#     lr=0.001, momentum=0.9, nesterov=True)

optimizer = Adam([
    {'params': model.parameters()},
    {'params': arc_loss.weight}],
    lr=0.0001)

model_param_num = len(model.state_dict().keys())
#________________________ DATASET ___________________________
trainloader, testloader, num_examples = load_datas()

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
        self.set_parameters(parameters)
        epochs: int = config["local_epochs"]
        loss, train_accuracy, length = train(epochs=epochs)
        threshold: float = config["threshold"]
        test_accuracy, tpr, fpr = eval_performance(threshold)
        results = {
            'id': args.cid,
            "loss": loss,
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "tpr": tpr,
            "fpr": fpr,
        }
        # print([val.cpu().numpy() for _, val in model.state_dict().items()])
        return [val.cpu().numpy() for _, val in model.state_dict().items()] + [val.cpu().numpy() for _, val in arc_loss.state_dict().items()], length, results
        # return [val.cpu().numpy() for _, val in model.state_dict().items()], length, results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        threshold: float = config["threshold"]
        accuracy, tpr, fpr = eval_performance(threshold)
        return float(0), num_examples["testset"], {'id': args.cid, "accuracy": float(accuracy), 'tpr': float(tpr),  'fpr': float(fpr)}
    
fl.client.start_numpy_client(server_address=args.server_address, client=FLWRClient())