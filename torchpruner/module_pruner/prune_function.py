from collections import OrderedDict
import torch
import torch.nn as nn

import numpy as np

from typing import Dict, List


def intable(value):
    try:
        int(value)
        return True
    except:
        return False


def create_slice(index, dim):
    slice_list = []
    for i in range(0, dim):
        slice_list.append(slice(None, None, None))
    slice_list.append(index)
    return slice_list


def neg_index(index, size):
    n_index = np.arange(0, size)
    mask = np.ones(size) > 0
    mask[index] = False
    n_index = n_index[mask]
    return n_index


def set_cut_tensor(tensor, cut_dims):
    data = tensor
    param_list = []
    for i in range(0, len(cut_dims)):
        if len(cut_dims[i]) == 0:
            param_list.append(None)
            continue
        data_slice = create_slice(cut_dims[i], i)
        param_list.append(data[data_slice].cpu().detach().numpy())
        data_slice = create_slice(neg_index(cut_dims[i], data.size(i)), i)
        if data.requires_grad:
            with torch.no_grad():
                data.set_(data[data_slice])
        else:
            data = data[data_slice]
    data.grad = None
    if (not data.requires_grad) and isinstance(tensor, nn.Parameter):
        tensor.data = data
    return tensor, param_list


def recovery_cut_tensor(tensor, cut_dims, param_list):
    data = tensor
    tensor_size = list(data.size())
    current_params = data.detach().cpu().numpy()
    for i in range(len(cut_dims) - 1, -1, -1):
        if len(cut_dims[i]) == 0:
            continue
        tensor_size[i] += len(cut_dims[i])
        array = np.zeros(tensor_size).astype("float32")
        index = neg_index(cut_dims[i], tensor_size[i])
        array[create_slice(index, i)] = current_params
        if param_list is not None:
            array[create_slice(cut_dims[i], i)] = param_list[i]
        else:
            array[create_slice(cut_dims[i], i)] = 0
        current_params = array
    new_data = torch.tensor(current_params, device=data.device)
    with torch.no_grad():
        data.set_(new_data)
    data.grad = None
    return data


def set_zero_tensor(tensor, cut_dims):
    data = tensor
    if isinstance(tensor, nn.Parameter):
        data = tensor.data
    param_list = []
    for i in range(0, len(cut_dims)):
        if len(cut_dims[i]) == 0:
            param_list.append(None)
            continue
        data_slice = create_slice(cut_dims[i], i)
        param_list.append(data[data_slice].detach().cpu().numpy())
        with torch.no_grad():
            data[data_slice] = 0
    if isinstance(tensor, nn.Parameter):
        tensor.data = data
    return tensor, param_list


def recovery_zero_tensor(tensor, cut_dims, param_list):
    data = tensor
    if isinstance(tensor, nn.Parameter):
        data = tensor.data
    for i in range(len(cut_dims) - 1, -1, -1):
        if len(cut_dims[i]) == 0:
            continue
        data_slice = create_slice(cut_dims[i], i)
        if param_list is not None:
            with torch.no_grad():
                data[data_slice] = torch.tensor(param_list[i], device=data.device)
        else:
            with torch.no_grad():
                data[data_slice] = 0
    if isinstance(tensor, nn.Parameter):
        tensor.data = data
    return tensor
