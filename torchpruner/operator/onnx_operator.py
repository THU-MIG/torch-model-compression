import numpy as np
import torchpruner.mask_utils as mask_utils
from collections import OrderedDict
from . import operator

import copy


def mask_mapping(
    node, mask, operator, defined_dict={}, masks=None, return_origin=False
):
    in_or_out, rank = operator.rank(node)
    node_key = in_or_out + "_" + str(rank)
    if masks is None:
        masks = create_masks(operator)
    if node_key not in defined_dict:
        raise RuntimeError("The " + node_key + " cut is not defined")
    dim_dict = defined_dict[node_key]
    indexs, dims = mask.indexs(return_dims=True)
    shape_length = len(indexs)
    for dim in dims:
        mapping_list = None
        if dim in dim_dict.keys():
            mapping_list = dim_dict[dim]
        elif dim - shape_length in dim_dict.keys():
            mapping_list = dim_dict[dim - shape_length]
        else:
            if "any" in dim_dict.keys():
                mapping_list = dim_dict["any"]
        if mapping_list is None:
            continue
        for item in mapping_list:
            current_in_or_out = "in"
            if item[0].startswith("in"):
                nodes = operator.in_data
            else:
                nodes = operator.out_data
                current_in_or_out = "out"
            node_rank = int(item[0].split("_")[1])
            if node_rank >= len(nodes):
                continue
            current_node = nodes[node_rank]
            c_mask = mask_utils.Mask(current_node.size())
            if current_in_or_out in masks.keys():
                if node_rank < len(masks[current_in_or_out]):
                    c_mask = masks[current_in_or_out][node_rank]
            cut_dim = item[1]
            if cut_dim == "any":
                if len(item) == 2:
                    cut_dim = dim
                else:
                    cut_dim = dim + item[2]
            c_mask.set_mask(indexs=[indexs[dim]], dims=[cut_dim])
    if return_origin:
        masks[in_or_out][rank] = mask
    return masks


# just one to one
def set_reduce_masks(node, mask, masks, operator, dims=None, keepdims=False):
    in_or_out, rank = operator.rank(node)
    if dims is None and keepdims is False:
        raise RuntimeError("unsupport axis is None and keepdims is 0")
    if in_or_out == "in":
        masks["out"][0] = mask.reduce(dims, keepdims)
    if in_or_out == "out":
        masks["in"][0] = mask.copy()
        if not keepdims:
            dims = list(dims).sort()
            for dim in dims:
                masks["in"][0] = masks["in"][0].expand_dim(dim)
    return masks


def mask_list_to_name(operator, masks):
    return_dict = OrderedDict()
    items = ["in", "out"]
    for item in items:
        data_list = None
        if item == "in":
            data_list = operator.in_data
            mask_list = masks["in"]
        if item == "out":
            data_list = operator.out_data
            mask_list = masks["out"]
        for i in range(0, len(data_list)):
            if mask_list[i] is None or mask_list[i].no_cut():
                continue
            return_dict[data_list[i].name] = mask_list[i]
    return return_dict


def create_masks(operator):
    masks = OrderedDict()
    masks["in"] = []
    masks["out"] = []
    items = ["in", "out"]
    for item in items:
        data_list = None
        if item == "in":
            data_list = operator.in_data
            mask_list = masks["in"]
        if item == "out":
            data_list = operator.out_data
            mask_list = masks["out"]
        for i in range(0, len(data_list)):
            mask_list.append(mask_utils.Mask(data_list[i].size()))
    return masks


# apply the boardcase here
class onnx_Gemm(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Gemm, self).__init__(node)
        self.index_mapping = {
            "in_0": {0: [("in_2", 0), ("out_0", 0)], 1: [("in_1", 0)]},
            "in_1": {0: [("in_0", 1)], 1: [("in_2", 1), ("out_0", 1)]},
            "in_2": {0: [("in_0", 0), ("out_0", 0)], 1: [("in_1", 1), ("out_0", 1)]},
            "out_0": {0: [("in_0", 0), ("in_2", 0)], 1: [("in_1", 1), ("in_2", 1)]},
        }

    def analysis(self, node, mask):
        transA = 0
        if "transA" in self.params.keys():
            transA = self.params["transA"]
        transB = 0
        if "transB" in self.params.keys():
            transB = self.params["transB"]
        masks = create_masks(self)
        in_or_out, rank = self.rank(node)
        shink_size = None
        if transA == 1:
            masks["in"][0] = masks["in"][0].transpose([1, 0])
        if transB == 1:
            masks["in"][1] = masks["in"][1].transpose([1, 0])
        if len(masks["in"]) == 3:
            shink_size = masks["in"][2].shape
            masks["in"][2] = masks["in"][2].boardcast(masks["out"][0].shape)
        if in_or_out == "in" and rank == 0 and transA == 1:
            mask = mask.transpose([1, 0])
        if in_or_out == "in" and rank == 1 and transB == 1:
            mask = mask.transpose([1, 0])
        if in_or_out == "in" and rank == 2:
            mask = mask.boardcast(masks["out"][0].shape)
        masks = mask_mapping(node, mask, self, self.index_mapping, masks)
        if shink_size != None:
            masks["in"][2] = masks["in"][2].shrinkcast(shink_size)
        if transA == 1:
            masks["in"][0] = masks["in"][0].transpose([1, 0])
        if transB == 1:
            masks["in"][1] = masks["in"][1].transpose([1, 0])
        return mask_list_to_name(self, masks), None

    def flops(self):
        shape1 = self.in_data[0].size()
        shape2 = self.in_data[1].size()
        flops_matrix = 0
        if shape2[0] in shape1:
            flops_matrix = shape1[0] * shape1[1] * shape2[1]
        else:
            flops_matrix = shape1[0] * shape1[1] * shape2[0]
        if len(self.in_data) == 3:
            out_data_shape = self.out_data[0].size()
            flops_matrix += out_data_shape[0] * out_data_shape[1]
        return flops_matrix


# the last one should be changed to mask type
class onnx_Concat(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Concat, self).__init__(node)

    def analysis(self, node, mask):
        params = self.params
        axis = params["axis"]
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        # if out, don't have to do this
        if in_or_out == "in":
            indexs, dims = mask.indexs(return_dims=True)
            del indexs[axis]
            if axis in dims:
                dims.remove(axis)
            for i in range(0, len(masks["in"])):
                masks["in"][i].set_mask(indexs, dims)
            masks["in"][rank] = mask
            masks["out"][0] = mask_utils.concatenate(masks["in"], axis)
            masks["in"][rank] = None
            return mask_list_to_name(self, masks), None
        else:
            masks["out"][0] = mask
            begin = 0
            for i in range(0, len(masks["in"])):
                length = masks["in"][i].shape[axis]
                masks["in"][i] = masks["out"][0].slice(begin, begin + length, axis)
                begin += length
            masks["out"][0] = None
            return mask_list_to_name(self, masks), None


class onnx_Split(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Split, self).__init__(node)

    def analysis(self, node, mask):
        params = self.params
        axis = params["axis"]
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        # if out, don't have to do this
        if in_or_out == "out":
            indexs, dims = mask.indexs(return_dims=True)
            del indexs[axis]
            dims.remove(axis)
            for i in range(0, len(masks["out"])):
                masks["out"][i].set_mask(indexs, dims)
            masks["out"][rank] = mask
            masks["in"][0] = mask_utils.concatenate(masks["out"], axis)
            masks["out"][rank] = None
            return mask_list_to_name(self, masks), None
        else:
            masks["in"][0] = mask
            begin = 0
            for i in range(0, len(masks["out"])):
                length = masks["out"][i].shape[axis]
                masks["out"][i] = masks["in"][0].slice(begin, begin + length, axis)
                begin += length
            masks["in"][0] = None
            return mask_list_to_name(self, masks), None


## the following is the point wise one to one condition
# Abs Acos Acosh Asin Asinh Atan Atanh Cast Ceil Reciprocal
# Clip Cos Cosh DequantizeLinear Dropout DynamicQuantizeLinear Elu Erf Exp
# Floor HardSigmoid Hardmax Identity IsInf IsNaN LeakyRelu Log
# LogSoftmax Neg NonZero Not Relu Round Selu Shrink Sigmoid Sign Sin
# Sinh Softmax Softplus Softsign Sqrt Tan Tanh ThresholdedRelu
# Max Mean Min LpNormalization LRN  MeanVarianceNormalization
## the PRelu and the ThreasholdedRelu is special
#####################################################
class onnx_pw(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_pw, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        for i in range(0, len(masks["in"])):
            if in_or_out == "in" and rank == i:
                continue
            masks["in"][i] = mask.copy()
        for i in range(0, len(masks["out"])):
            if in_or_out == "out" and rank == i:
                continue
            masks["out"][i] = mask.copy()
        return mask_list_to_name(self, masks), None

    def flops(self):
        shape = self.in_data[0].size()
        flops = 1
        for size in shape:
            flops = flops * size
        return flops


## boardcast and pointwise
## just support n to 1
#  Add And BitShift Div Equal Greater Less Mod Mul Sub
#  Or Pow Sum Xor
# PRelu ThresholdedRelu
#######################################################
class onnx_bc_pw(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_bc_pw, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        # boardcast the mask
        boardcast_shape = self.out_data[0].size()
        mask = mask.boardcast(boardcast_shape)
        # pointwise assign
        for i in range(0, len(masks["in"])):
            if in_or_out == "in" and rank == i:
                continue
            masks["in"][i] = mask.copy()
        for i in range(0, len(masks["out"])):
            if in_or_out == "out" and rank == i:
                continue
            masks["out"][i] = mask.copy()
        # shrink the mask
        for i in range(0, len(masks["in"])):
            masks["in"][i] = masks["in"][i].shrinkcast(self.in_data[i].size())
        for i in range(0, len(masks["out"])):
            masks["out"][i] = masks["out"][i].shrinkcast(self.out_data[i].size())
        return mask_list_to_name(self, masks), None

    def flops(self):
        input_num = len(self.in_data)
        shape = self.out_data[0].size()
        flops = 1
        for size in shape:
            flops = flops * size
        return flops * (input_num - 1)


## reduce type operator, reduce the dims
# ArgMax ArgMin GlobalAveragePool GlobalLpPool OneHot Squeeze
# ReduceL1 ReduceL2 ReduceLogSum ReduceLogSumExp ReduceMax ReduceMean ReduceMin
# ReduceProd ReduceSum ReduceSUmSquare

# ordinary reduce
# ArgMax ArgMin ReduceL1 ReduceL2 ReduceLogSum ReduceLogSumExp
# ReduceMax ReduceMean ReduceMin ReduceProd ReduceSum ReduceSUmSquare
class onnx_reduce(operator.OperatorNode):
    def __init__(self, parameters={}):
        super(onnx_reduce, self).__init__(parameters)

    def get_axis(self):
        axis = None
        if "axes" in self.params:
            axis = self.params["axes"]
        else:
            axis = self.params["axis"]
        if not isinstance(axis, (list, tuple)):
            axis = [axis]
        keepdims = self.params["keepdims"]
        if keepdims == 1:
            keepdims = True
        else:
            keepdims = False
        return axis, keepdims

    def analysis(self, node, mask):
        axis, keepdims = self.get_axis()
        masks = create_masks(self)
        masks = set_reduce_masks(node, mask, masks, self, dims=axis, keepdims=keepdims)
        return mask_list_to_name(self, masks), None


# Global reduce
# GlobalLpPool GlobalAveragePool GlobalMaxPool
class onnx_GlobalPool(onnx_reduce):
    def __init__(self, node):
        super(onnx_GlobalPool, self).__init__(node)

    def get_axis(self):
        in_data = self.in_data[0]
        if len(in_data.size()) <= 2:
            raise RuntimeError("Wrong demision")
        keepdims = True
        axis = list(range(2, len(in_data.size())))
        return axis, keepdims


# Squeeze
class onnx_Squeeze(onnx_reduce):
    def __init__(self, node):
        super(onnx_Squeeze, self).__init__(node)

    def get_axis(self):
        axis = self.params["axes"]
        return axis, False


# UnSqueeze
class onnx_Unsqueeze(operator.OperatorNode):
    def __init__(self, parameters={}):
        super(onnx_Unsqueeze, self).__init__(parameters)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        axes = self.params["axes"]
        if in_or_out == "out":
            masks["in"][0] = mask.reduce(axes, False)
        if in_or_out == "in":
            dims = list(axes).sort()
            for dim in dims:
                masks["out"][0] = masks["out"][0].expand_dim(dim)
        return mask_list_to_name(self, masks), None


# no action operation
# Constant RandomNormal RandomNormalLike RandomUniform
# RandomUniformLike Range Shape Size MonMaxsuppression
# Gather GatherElements GatherND
class onnx_no_action(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_no_action, self).__init__(node)

    def analysis(self, node, mask):
        return OrderedDict(), None


# unsupport operation
# means the op doesn't support cutting
class onnx_unsupport(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_unsupport, self).__init__(node)

    def analysis(self, node, mask):
        raise RuntimeError("The opeartor doesn't support cutting")


# Conv like operation
# ConvInteger ConvTranspose Conv
# In_0: B G C
# In_1: G W C
# In_2: G W
# Out_0:B G W
class onnx_conv(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_conv, self).__init__(node)
        self.index_mapping = {
            # input
            "in_0": {
                0: [("out_0", 0)],
                1: [("in_1", 0), ("in_2", 0), ("out_0", 1)],
                2: [("in_1", 2)],
            },
            "in_1": {
                0: [("in_2", 0), ("out_0", 1), ("in_0", 1)],
                1: [("in_2", 1), ("out_0", 2)],
                2: [("in_0", 2)],
            },
            "in_2": {
                0: [("in_1", 0), ("out_0", 1), ("in_0", 1)],
                1: [("in_1", 1), ("out_0", 2)],
            },
            "out_0": {
                0: [("in_0", 0)],
                1: [("in_1", 0), ("in_2", 0), ("in_0", 1)],
                2: [("in_1", 1), ("in_2", 1)],
            },
        }

    def analysis(self, node, mask):
        params = self.params
        group = params["group"]
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)

        # divide in data
        for i in range(0, len(masks["in"])):
            if i == 0:
                masks["in"][0] = masks["in"][0].divide_dim(1, (group, -1))
            if i == 1 or i == 2:
                masks["in"][i] = masks["in"][i].divide_dim(0, (group, -1))
        # for ConvTranspose condition
        if self.type == "ConvTranspose":
            dims = list(range(0, len(masks["in"][1].shape)))
            dims[1] = 2
            dims[2] = 1
            masks["in"][1] = masks["in"][1].transpose(dims=dims)
        # divide out data
        for i in range(0, len(masks["out"])):
            if i == 0:
                masks["out"][0] = masks["out"][0].divide_dim(1, (group, -1))

        if in_or_out == "in":
            if rank == 0:
                mask = mask.divide_dim(1, (group, -1))
            if rank == 1 or rank == 2:
                mask = mask.divide_dim(0, (group, -1))
        if in_or_out == "out":
            mask = mask.divide_dim(1, (group, -1))

        if in_or_out == "in" and rank == 1 and self.type == "ConvTranspose":
            dims = list(range(0, len(mask.shape)))
            dims[1] = 2
            dims[2] = 1
            mask = mask.transpose(dims=dims)

        operator_dict = None
        group_cut = 0
        # sync the new_mask
        if in_or_out == "in":
            if rank == 0:
                mask = mask.trim(2)
                indexs = mask.indexs()
                group_cut = len(indexs[1])
            if rank == 1 or rank == 2:
                mask = mask.trim(1)
                indexs = mask.indexs()
                group_cut = len(indexs[0])
        if in_or_out == "out":
            mask = mask.trim(2)
            indexs = mask.indexs()
            group_cut = len(indexs[1])
        if group_cut >= 1:
            operator_dict = {"group": group_cut}
        # get result_dict
        masks = mask_mapping(node, mask, self, self.index_mapping, masks, True)
        # for ConvTranspose condition
        if self.type == "ConvTranspose":
            dims = list(range(0, len(masks["in"][1].shape)))
            dims[1] = 2
            dims[2] = 1
            masks["in"][1] = masks["in"][1].transpose(dims=dims)
        # recovery to normal shape
        for i in range(0, len(masks["in"])):
            if i == 0:
                masks["in"][0] = masks["in"][0].combine_dim([1, 2])
            if i == 1 or i == 2:
                masks["in"][i] = masks["in"][i].combine_dim([0, 1])
        # recovery out data
        for i in range(0, len(masks["out"])):
            if i == 0:
                masks["out"][0] = masks["out"][0].combine_dim([1, 2])
        # print(masks)
        return mask_list_to_name(self, masks), operator_dict

    def flops(self):
        kernal_flops = 1
        kernel_shape = self.params["kernel_shape"]
        for size in kernel_shape:
            kernal_flops = kernal_flops * size
        single_flops = (
            kernal_flops
            * self.in_data[0].size(1)
            * self.out_data[0].size(1)
            / self.params["group"]
        )
        element_size = 1
        if self.type == "Conv":
            shape = self.out_data[0].size()
            for i in range(2, len(shape)):
                element_size = element_size * shape[i]
        if self.type == "ConvTranspose":
            shape = self.in_data[0].size()
            for i in range(2, len(shape)):
                element_size = element_size * shape[i]
        element_count = self.in_data[0].size(0) * element_size
        conv_flops = single_flops * element_count
        # bias
        bias_size = 0
        if len(self.in_data) == 3:
            shape = self.in_data[2].size()
            bias_size = 1
            for i in range(0, len(shape)):
                bias_size = bias_size * shape[i]
        bias_flops = bias_size * element_count
        return conv_flops + bias_flops


# Expand
class onnx_Expand(operator.OperatorNode):
    def __init__(self, parameters={}):
        super(onnx_Expand, self).__init__(parameters)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        if in_or_out == "in" and rank == 1:
            raise RuntimeError("The shape is not cuttable")
        if in_or_out == "in" and rank == 0:
            masks["out"][0] = mask.boardcast(list(self.out_data[0].size()))
        if in_or_out == "out" and rank == 0:
            masks["in"][0] = mask.shrinkcast(list(self.in_data[0].size()))
        return mask_list_to_name(self, masks), None


##onnx mapping
# BatchNormalization GRU InstanceNormalization
# LRN LSTM MatMul MatMulInteger
# Matmul QuantizeLinear AveragePool MaxPool MaxUnPool
class onnx_mapping(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_mapping, self).__init__(node)
        if "strides" in self.params:
            if isinstance(self.params["strides"], list):
                if len(self.params["strides"]) == 0:
                    self.params["strides"] = copy.deepcopy(self.params["kernel_shape"])
        self.index_mapping = {}

    def analysis(self, node, mask):
        return (
            mask_list_to_name(self, mask_mapping(node, mask, self, self.index_mapping)),
            None,
        )


# GRU complex
# implement later
class onnx_GRU(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_GRU, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                2: [("in_1", 3)],
            },
            "in_1": {
                3: [("in_0", 2)],
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "in_2": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
                3: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "in_3": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ]
            },
            "in_5": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ]
            },
            "out_0": {
                3: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ]
            },
            "out_1": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ]
            },
        }

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)

        # divide in_5
        masks["in"][1] = masks["in"][1].divide_dim(1, (4, -1))
        masks["in"][2] = masks["in"][2].divide_dim(1, (4, -1))
        masks["in"][3] = masks["in"][3].divide_dim(1, (8, -1))

        # get result_dict
        masks = mask_mapping(node, mask, self, self.index_mapping, masks, True)
        # recovery out data

        operator_dict = {"hidden_size": len(masks["in"][1].indexs()[2])}

        masks["in"][1] = masks["in"][1].combine_dim([1, 2])
        masks["in"][2] = masks["in"][2].combine_dim([1, 2])
        masks["in"][3] = masks["in"][3].combine_dim([1, 2])
        return mask_list_to_name(self, masks), operator_dict


# LSTM complex
class onnx_LSTM(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_LSTM, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                2: [("in_1", 3)],
            },
            "in_1": {
                3: [("in_0", 2)],
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ],
            },
            "in_2": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ],
                3: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ],
            },
            "in_3": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "in_5": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "in_6": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "in_7": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "out_0": {
                3: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "out_1": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
            "out_2": {
                2: [
                    ("in_1", 2),
                    ("in_2", 2),
                    ("in_2", 3),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("in_6", 2),
                    ("in_7", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                    ("out_2", 2),
                ]
            },
        }

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)

        # divide in_5
        masks["in"][1] = masks["in"][1].divide_dim(1, (4, -1))
        masks["in"][2] = masks["in"][2].divide_dim(1, (4, -1))
        masks["in"][3] = masks["in"][3].divide_dim(1, (8, -1))
        masks["in"][7] = masks["in"][7].divide_dim(1, (3, -1))

        # get result_dict
        masks = mask_mapping(node, mask, self, self.index_mapping, masks, True)
        # recovery out data

        operator_dict = {"hidden_size": len(masks["in"][1].indexs()[2])}

        masks["in"][1] = masks["in"][1].combine_dim([1, 2])
        masks["in"][2] = masks["in"][2].combine_dim([1, 2])
        masks["in"][3] = masks["in"][3].combine_dim([1, 2])
        masks["in"][7] = masks["in"][7].combine_dim([1, 2])
        return mask_list_to_name(self, masks), operator_dict


# RNN complex
class onnx_RNN(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_RNN, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                2: [("in_1", 2)],
            },
            "in_1": {
                2: [("in_0", 2)],
                1: [
                    ("in_2", 1),
                    ("in_2", 2),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "in_2": {
                1: [
                    ("in_1", 1),
                    ("in_2", 2),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
                2: [
                    ("in_1", 1),
                    ("in_2", 1),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "in_3": {
                2: [
                    ("in_1", 1),
                    ("in_2", 1),
                    ("in_2", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "in_5": {
                2: [
                    ("in_1", 1),
                    ("in_2", 1),
                    ("in_2", 2),
                    ("in_3", 2),
                    ("out_0", 3),
                    ("out_1", 2),
                ],
            },
            "out_0": {
                3: [
                    ("in_1", 1),
                    ("in_2", 1),
                    ("in_2", 2),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_1", 2),
                ],
            },
            "out_1": {
                2: [
                    ("in_1", 1),
                    ("in_2", 1),
                    ("in_2", 2),
                    ("in_3", 2),
                    ("in_5", 2),
                    ("out_0", 3),
                ],
            },
        }

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)

        # divide in_5
        masks["in"][5] = masks["in"][5].divide_dim(2, (2, -1))

        # get result_dict
        masks = mask_mapping(node, mask, self, self.index_mapping, masks, True)
        # recovery out data
        operator_dict = {"hidden_size": len(masks["in"][1].indexs()[1])}

        masks["in"][5] = masks["out"][0].combine_dim([2, 3])
        return mask_list_to_name(self, masks), operator_dict


# InstanceNormalization
class onnx_InstanceNormalization(onnx_mapping):
    def __init__(self, node):
        super(onnx_InstanceNormalization, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                0: [("out_0", 0)],
                1: [("in_1", 0), ("in_2", 0), ("out_0", 1)],
                "any": [("out_0", "any")],
            },
            "in_1": {0: [("in_0", 1), ("in_2", 0), ("out_0", 1)]},
            "in_2": {0: [("in_0", 1), ("in_1", 0), ("out_0", 1)]},
            "out_0": {
                0: [("in_0", 0)],
                1: [("in_1", 0), ("in_2", 0), ("in_0", 1)],
                "any": [("in_0", "any")],
            },
        }

    def flops(self):
        shape = self.in_data[0].size()
        flops = 1
        for size in shape:
            flops = flops * size
        return 2 * flops


class onnx_BatchNormalization(onnx_mapping):
    def __init__(self, node):
        super(onnx_BatchNormalization, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                0: [("out_0", 0)],
                1: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ],
                "any": [("out_0", "any")],
            },
            "out_0": {
                0: [("in_0", 0)],
                1: [
                    ("in_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ],
                "any": [("in_0", "any")],
            },
            "in_1": {
                0: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_0", 1),
                    ("in_2", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ]
            },
            "in_2": {
                0: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_0", 1),
                    ("in_1", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ]
            },
            "in_3": {
                0: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_0", 1),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_4", 0),
                ]
            },
            "in_4": {
                0: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("out_2", 0),
                    ("in_0", 1),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_3", 0),
                ]
            },
            "out_1": {
                0: [
                    ("out_0", 1),
                    ("out_2", 0),
                    ("in_0", 1),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ]
            },
            "out_2": {
                0: [
                    ("out_0", 1),
                    ("out_1", 0),
                    ("in_0", 1),
                    ("in_1", 0),
                    ("in_2", 0),
                    ("in_3", 0),
                    ("in_4", 0),
                ]
            },
        }

    def flops(self):
        shape = self.in_data[0].size()
        flops = 1
        for size in shape:
            flops = flops * size
        return flops * 2


# MatMul
class onnx_MatMul(onnx_mapping):
    def __init__(self, node):
        super(onnx_MatMul, self).__init__(node)
        self.index_mapping = {
            "in_0": {
                -2: [("out_0", -2)],
                -1: [("in_1", -2)],
                "any": [("in_1", "any"), ("out_0", "any")],
            },
            "in_1": {
                -2: [("in_0", -1)],
                -1: [("out_0", -1)],
                "any": [("in_0", "any"), ("out_0", "any")],
            },
            "out_0": {
                -2: [("in_0", -2)],
                -1: [("in_1", -1)],
                "any": [("in_0", "any"), ("in_1", "any")],
            },
        }

    def flops(self):
        base = 1
        in_shape = self.in_data[0].size()
        out_shape = self.out_data[0].size()
        for i in range(0, len(in_shape) - 2):
            base = base * in_shape[i]
        single_flops = in_shape[-1] * out_shape[-2] * out_shape[-1]
        return base * single_flops


# MaxRoiPool
class onnx_MaxRoiPool(onnx_mapping):
    def __init__(self, node):
        super(onnx_MaxRoiPool, self).__init__(node)
        self.index_mapping = {"in_0": {0: [("out_0", 1)]}, "out_0": {1: [("in_0", 0)]}}


# RoiAlign
class onnx_RoiAlign(onnx_mapping):
    def __init__(self, node):
        super(onnx_RoiAlign, self).__init__(node)
        self.index_mapping = {"in_0": {0: [("out_0", 1)]}, "out_0": {1: [("in_0", 0)]}}


# AveragePool MaxPool MaxUnPool LpPool
class onnx_local_pool(onnx_mapping):
    def __init__(self, node):
        super(onnx_local_pool, self).__init__(node)
        self.index_mapping = {
            "in_0": {0: [("out_0", 0)], 1: [("out_0", 1)]},
            "out_0": {0: [("in_0", 0)], 1: [("in_0", 1)]},
        }

    def flops(self):
        element_shape = None
        if self.type == "MaxUnPool":
            element_shape = self.in_data[0].size()
        else:
            element_shape = self.out_data[0].size()
        element_size = 1
        for size in element_shape:
            element_size = element_size * size
        kernel_shape = self.params["kernel_shape"]
        k_size = 1
        for size in kernel_shape:
            k_size = k_size * size
        return element_size * k_size


# MatMulInteger
class onnx_MatMulInteger(onnx_mapping):
    def __init__(self, node):
        super(onnx_MatMulInteger, self).__init__(node)
        raise NotImplementedError()

    def flops(self):
        raise NotImplementedError("The MatMulInteger is not implemented")


# Flatten
class onnx_Flatten(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Flatten, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        axis = self.params["axis"]
        if in_or_out == "in":
            masks["out"][0] = mask.combine_dim(dims=list(range(axis, len(mask.shape))))
        if in_or_out == "out":
            masks["in"][0] = mask.divide_dim(
                dim=axis, size=self.in_data[0].size()[axis:]
            )
            masks["in"][0] = masks["in"][0].trim(axis)
            masks["out"][0] = masks["in"][0].combine_dim(
                dims=list(range(axis, len(mask.shape)))
            )
        return mask_list_to_name(self, masks), None


# Reshape
class onnx_Reshape(operator.OperatorNode):
    def __init__(self, parameters={}):
        super(onnx_Reshape, self).__init__(parameters)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        shape = self.in_data[1].data.tolist()
        out_shape = shape
        in_shape = self.in_data[0].size()
        if in_or_out == "in":
            masks["out"][0] = mask.combine_dim()
            masks["out"][0] = masks["out"][0].divide_dim(0, out_shape)
            max_value = max(masks["out"][0]._narray.shape)
            masks["out"][0] = masks["out"][0].trim(
                list(masks["out"][0]._narray.shape[:]).index(max_value)
            )
            masks["in"][0] = masks["out"][0].combine_dim()
            masks["in"][0] = masks["in"][0].divide_dim(0, in_shape)
        if in_or_out == "out":
            masks["in"][0] = mask.combine_dim()
            masks["in"][0] = masks["in"][0].divide_dim(0, in_shape)
            max_value = max(masks["in"][0]._narray.shape)
            masks["in"][0] = masks["in"][0].trim(
                list(masks["in"][0]._narray.shape[:]).index(max_value)
            )
            masks["out"][0] = masks["in"][0].combine_dim()
            masks["out"][0] = masks["out"][0].divide_dim(0, out_shape)
        return mask_list_to_name(self, masks), None


# Slice
class onnx_Slice(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Slice, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = node.rank(self)
        masks = create_masks(self)
        starts = list(self.in_data[1].data)
        ends = list(self.in_data[2].data)
        axes = None
        if len(self.in_data) >= 4:
            axes = list(self.in_data[3].data)
        else:
            axes = range(0, len(starts))
        steps = None
        if len(self.in_data) >= 5:
            steps = list(self.in_data[3].data)
        else:
            steps = list(np.ones(len(starts), dtype=int))
        if in_or_out == "in" and rank == 0:
            masks["out"][0] = mask.copy()
            for i in range(0, len(axes)):
                masks["out"][0] = mask.slice(starts[i], ends[i], axes[i], steps[i])
        if in_or_out == "out" and rank == 0:
            axes.sort()
            mask_slice = []
            for i in range(0, len(self.in_data[0].shape)):
                if i not in axes:
                    mask_slice.append(slice())
                else:
                    index = axes.index(i)
                    mask_slice = slice(starts[index], ends[index], steps[index])
            masks["in"][0][mask_slice] = mask
        return mask_list_to_name(self, masks), None


# Resize
class onnx_Resize(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Resize, self).__init__(node)
        self.index_mapping = {
            "in_0": {"any": [("out_0", "any")]},
            "out_0": {"any": [("in_0", "any")]},
        }

    def analysis(self, node, mask):
        masks = create_masks(self)
        indexs, dims = mask.indexs(return_dims=True)
        masks = mask_mapping(node, mask, self, self.index_mapping, masks)
        return mask_list_to_name(self, masks), None


# Pad
class onnx_Pad(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Pad, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        pads = list(self.in_data[1].data)
        if in_or_out == "in":
            if rank == 1 or rank == 2:
                raise RuntimeError("Unsupport cut the pads or value")
            indexs, dims = mask.indexs(return_dims=True)
            for dim in dims:
                if len(indexs[dim]) == 0:
                    continue
                indexs[dim] = list(np.array(indexs[dim]) + pads[dim * 2])
                masks["out"][0].set_mask([indexs[dim]], [dim])
            return mask_list_to_name(self, masks), None
        if in_or_out == "out":
            in_shape = self.in_data[0].size()
            indexs, dims = mask.indexs(return_dims=True)
            for dim in dims:
                if len(indexs[dim]) == 0:
                    continue
                indexs[dim] = list(np.array(indexs[dim]) - pads[dim * 2])
                indexs[dim] = list(
                    filter(lambda x: x >= 0 and x < in_shape[dim], indexs[dim])
                )
                masks["in"][0].set_mask([indexs[dim]], [dim])
            return mask_list_to_name(self, masks), None


# Transpose
class onnx_Transpose(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Transpose, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        perm = list(self.params["perm"])
        if in_or_out == "in":
            masks["out"][0] = mask.transpose(perm)
        else:
            masks["in"][0] = mask.transpose(perm)
        return mask_list_to_name(self, masks), None


# TopK
class onnx_TopK(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_TopK, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        indexs, dims = mask.indexs(return_dims=True)
        dim = self.params["axis"]
        k = self.in_data[1].data[0]
        if in_or_out == "out":
            raise RuntimeError("The topK output is not cuttable")
        if in_or_out == "in":
            if dim in dims:
                if len(indexs[dim]) < k:
                    raise RuntimeError("The input is less than k")
        return mask_list_to_name(self, masks), None


# Compress
class onnx_Compress(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_Compress, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        axis = None
        masks = create_masks(self)
        if "axis" in self.params:
            axis = self.params["axis"]
        condition = list(self.in_data[1].data)
        if in_or_out == "in":
            if axis is None:
                mask = mask.combine_dim()
                mask = mask[condition]
                masks["out"][0] = mask
                return mask_list_to_name(self, masks), None
            else:
                mask = mask[mask_utils.dim_slice(condition, axis)]
                masks["out"][0] = mask
                return mask_list_to_name(self, masks), None
        if in_or_out == "out":
            if axis is not None:
                masks["in"][0][mask_utils.dim_slice(condition, axis)] = mask
            else:
                narray = masks["in"][0].to_array(full=True)
                narray[condition] = mask.to_array(full=True)
                masks["in"][0] = mask_utils.from_array(narray, masks["in"][0].shape)
                return mask_list_to_name(self, masks), None


class onnx_SpaceToDepth(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_SpaceToDepth, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        blocksize = self.params["blocksize"]
        if in_or_out == "in":
            indexs, dims = mask.indexs(return_dims=True)
            if max(dims) >= 2:
                raise RuntimeError("Just support cut the batch and channels")
            for dim in dims:
                index = indexs[dim]
                index = np.expand_dims(index, 1)
                index = np.repeat(index, blocksize * blocksize, 1)
                index = index * blocksize * blocksize
                padding = np.arange(0, blocksize * blocksize)
                index = index + padding
                index = np.reshape(index, (-1))
                index = np.sort(index, 0)
                indexs[dim] = index
            masks["out"][0].set_mask(indexs, dims)
            return mask_list_to_name(self, masks), None
        if in_or_out == "out":
            indexs, dims = mask.indexs(return_dims=True)
            if max(dims) >= 2:
                raise RuntimeError("Just support cut the batch and channels")
            mask = mask.divide_dim(1, (-1, blocksize * blocksize))
            mask = mask.trim(1)
            indexs, dims = mask.indexs(return_dims=True)
            indexs = indexs[:2]
            dims = dims[:2]
            masks["in"][0] = masks["in"][0].set_mask(indexs, dims)
            mask = mask.combine_dim([1, 2])
            masks["out"][0] = mask
            return mask_list_to_name(self, masks), None


# opposite to the SpaceToDepth
class onnx_DepthToSpace(operator.OperatorNode):
    def __init__(self, node):
        super(onnx_DepthToSpace, self).__init__(node)

    def analysis(self, node, mask):
        in_or_out, rank = self.rank(node)
        masks = create_masks(self)
        blocksize = self.params["blocksize"]
        if in_or_out == "out":
            indexs, dims = mask.indexs(return_dims=True)
            if max(dims) >= 2:
                raise RuntimeError("Just support cut the batch and channels")
            for dim in dims:
                index = indexs[dim]
                index = np.expand_dims(index, 1)
                index = np.repeat(index, blocksize * blocksize, 1)
                index = index * blocksize * blocksize
                padding = np.arange(0, blocksize * blocksize)
                index = index + padding
                index = np.reshape(index, (-1))
                index = np.sort(index, 0)
                indexs[dim] = index
            masks["in"][0].set_mask(indexs, dims)
            return mask_list_to_name(self, masks), None
        if in_or_out == "in":
            indexs, dims = mask.indexs(return_dims=True)
            if max(dims) >= 2:
                raise RuntimeError("Just support cut the batch and channels")
            mask = mask.divide_dim(1, (-1, blocksize * blocksize))
            mask = mask.trim(1)
            indexs, dims = mask.indexs(return_dims=True)
            indexs = indexs[:2]
            dims = dims[:2]
            masks["out"][0] = masks["out"][0].set_mask(indexs, dims)
            mask = mask.combine_dim([1, 2])
            masks["in"][0] = mask
            return mask_list_to_name(self, masks), None


# onnx QDQ
# just for version 10
class onnx_QDQ(onnx_mapping):
    def __init__(self, node):
        super(onnx_QDQ, self).__init__(node)
        self.index_mapping = {
            "in_0": {"any": [("out_0", "any")]},
            "out_0": {"any": [("in_0", "any")]},
        }


class onnx_where(onnx_mapping):
    def __init__(self, node):
        super(onnx_where, self).__init__(node)
        self.index_mapping = {
            "in_0": {},
            "in_1": {"any": [("in_2", "any"), ("out_0", "any")]},
            "int_2": {"any": [("in_1", "any"), ("out_0", "any")]},
            "out_0": {"any": [("in_1", "any"), ("in_2", "any")]},
        }


class onnx_ScatterElements(onnx_mapping):
    def __init__(self, node):
        super(onnx_ScatterElements, self).__init__(node)
        self.index_mapping = {
            "in_0": {"any": [("out_0", "any")]},
            "out_0": {"any": [("in_0", "any")]},
        }
