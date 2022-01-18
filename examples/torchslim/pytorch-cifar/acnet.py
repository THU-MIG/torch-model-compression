import torchslim
import torchslim.pruning
import torchslim.reparameter.reparam as reparam

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

reparam.ReparamSolver.print_config()
reparam.ReparamSolver.print_config(help=True)


def predict_function(model, data):
    X, _ = data
    prediction = model(X)
    return prediction.view(-1, prediction.size(1))


def calculate_loss(predict, data):
    _, Y = data
    loss = F.cross_entropy(predict, Y)
    return loss


def evaluate(predict, data):
    _, Y = data
    _, predicted = predict.max(1)
    correct = predicted.eq(Y).sum().item()
    return {"acc": correct / predict.size(0)}


def dataset_generator():
    return trainset, testset


config = {}
config["devices"] = [0]
config["epoch"] = 400
config["task_name"] = "resnet56_acnet"
config["lr"] = 0.2
config["num_workers"] = 2
config["save_keyword"] = "acc"
config["reparam_type"] = "acnet_cr"
config["reparam_args"] = { "with_bn": True }
config["predict_function"] = predict_function
config["calculate_loss_function"] = calculate_loss
config["evaluate_function"] = evaluate
config["dataset_generator"] = dataset_generator
config["save_deploy_format"] = False
config["batch_size"] = 128
config["log_interval"] = 50

import models

model = models.ResNet56()

solver = reparam.ReparamSolver(model, config)
solver.run()
