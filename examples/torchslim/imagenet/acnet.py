import sys

sys.path.append("../torch-compression")

import torchslim
import torchslim.pruning
import torchslim.reparameter.acnet as acnet

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from imagenet_train import *

acnet.ACNetSolver.print_config()
acnet.ACNetSolver.print_config(help=True)


config = {}
config["devices"] = [0, 1, 2, 3]
config["task_name"] = "resnet50_imagenet_acnet_cr"
config["lr"] = 0.1
config["epoch"] = 90
config["decay_step"] = [30, 60]
config["save_keyword"] = "top1"
config["acnet_type"] = "acnet_cr"
config["predict_function"] = predict_function
config["calculate_loss_function"] = calculate_loss
config["evaluate_function"] = evaluate
config["dataset_generator"] = generate_dataset_generator(
    "/dataset/ILSVRC/Data/CLS-LOC/"
)
config["batch_size"] = 256
config["num_workers"] = 32

import torchvision
import torchvision.models as models

model = models.resnet18(pretrained=False)

solver = acnet.ACNetSolver(model, config)
solver.run()
