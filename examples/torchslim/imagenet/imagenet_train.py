import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
import os


def generate_dataset_generator(file_path):
    def dataset_generator():
        traindir = os.path.join(file_path, "train")
        valdir = os.path.join(file_path, "val")
        # Normalize on RGB Value
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # dataset
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomSizedCrop(224),  # 224 , 299
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        return train_dataset, val_dataset

    return dataset_generator


def predict_function(model, data):
    X, _ = data
    prediction = model(X)
    return prediction.view(-1, prediction.size(1))


def calculate_loss(predict, data):
    _, Y = data
    loss = F.cross_entropy(predict, Y)
    return loss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1))

    res = []
    for k in topk:
        correct_k = correct[:, :k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def evaluate(predict, data):
    _, Y = data
    res = accuracy(predict, Y, (1, 5))
    return {"top1": res[0], "top5": res[1]}


def optimizer_generator(params, config):
    return torch.optim.SGD(params, lr=0.0)


def scheduler_generator(optimizer, config):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config["decay_steps"], gamma=0.1, last_epoch=-1
    )
