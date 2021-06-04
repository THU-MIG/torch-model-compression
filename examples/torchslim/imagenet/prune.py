import sys

sys.path.append("../torch-compression")

import torchslim
import torchslim.pruning
import torchslim.pruning.resrep as resrep

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from imagenet_train import *


config = {}
config["devices"] = [0, 1, 2, 3, 4, 5, 6]
config["task_name"] = "resnet50_prune"
config["input_shapes"] = [[3, 224, 224]]
config["prune_rate"] = 0.55
config["prune_groups"] = 1
config["group_size"] = 8
config["prune_interval"] = 400
config["lasso_decay"] = 1e-4
config["warmup_epoch"] = 5
config["lr"] = 0.01
config["epoch"] = 180
# config['decay_step']=[30,60]
config["save_keyword"] = "top1"
config["predict_function"] = predict_function
config["calculate_loss_function"] = calculate_loss
config["evaluate_function"] = evaluate
config["dataset_generator"] = generate_dataset_generator(
    "/dataset/ILSVRC/Data/CLS-LOC/"
)
config["batch_size"] = 256
config["test_batch_size"] = 100
config["num_workers"] = 32

import torchvision
import torchvision.models as models

model = models.resnet50(pretrained=True)
print(model)

solver = resrep.ResRepSolver(model, config)
solver.run()
