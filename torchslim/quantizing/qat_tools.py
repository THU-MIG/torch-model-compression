import torch
import torch.nn as nn
import torch.nn.quantized.modules.functional_modules as qf
import torch.nn.intrinsic.qat as nniqat
import torch.nn.intrinsic as intrinsic
import torch.nn.qat as nnqat
import torch.nn.functional as F
import torch.quantization as q
import onnx
from onnx import onnx_pb as onnx_proto
import copy

import torchpruner
import torchpruner.model_tools as model_tools
import torchpruner.function_module as function_module

from torchslim.modules.rep_modules import merge_conv_bn

tensorrt_qconfig = q.QConfig(
    activation=q.default_weight_fake_quant, weight=q.default_weight_fake_quant
)


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output"""
    return self._post_observer(output)


def _observer_forward_pre_hook(self, input):
    """Forward pre hook that calls observer on the input (can be a tuple of values)"""
    return self._pre_observer(*input)


def _regist_post_observer(module, observer):
    module.add_module("_post_observer", observer)
    module.register_forward_hook(_observer_forward_hook)


def _regist_pre_observer(module, observer):
    module.add_module("_pre_observer", observer)
    module.register_forward_pre_hook(_observer_forward_pre_hook)


def _mark_end(model, graph, class_list):
    name_groups = model_tools.get_name_groups_by_classes(graph, class_list)
    for name_group in name_groups:
        # the last mark with
        for i in range(0, len(name_group) - 1):
            obj = model_tools.get_object(model, name_group[i])
            obj.require_observer = False
        obj = model_tools.get_object(model, name_group[len(name_group) - 1])
        if hasattr(obj, "require_observer") and obj.require_observer == False:
            continue
        obj.require_observer = True


def prepare_qat(model, inputs, qconfig=None):
    if not torchpruner.function_module_activate():
        raise RuntimeError("The function module mode must be enable")
    if qconfig is None:
        raise RuntimeError("The qconfig must be provided")

    # forward create the function_module
    torchpruner.init_function_module(model, inputs)

    # create graph
    graph = torchpruner.ONNXGraph(model)
    graph.build_graph(inputs)

    # mark the module that need observer
    # [conv]
    _mark_end(model, graph, [nn.Conv2d])
    # [conv bn]
    _mark_end(model, graph, [nn.Conv2d, nn.BatchNorm2d])
    # [conv bn relu]
    _mark_end(model, graph, [nn.Conv2d, nn.BatchNorm2d, function_module.Relu])
    # [add]
    _mark_end(model, graph, [function_module.Add])
    # [cat]
    _mark_end(model, graph, [function_module.Cat])
    # [relu]
    _mark_end(model, graph, [function_module.Relu])
    # [add relu]
    _mark_end(model, graph, [function_module.Add, function_module.Relu])
    # [cat relu]
    _mark_end(model, graph, [function_module.Cat, function_module.Relu])

    # convert conv bn to quantization format
    name_groups = torchpruner.model_tools.get_name_groups_by_classes(
        graph, [nn.Conv2d, nn.BatchNorm2d], include_super_class=False
    )
    for name_group in name_groups:
        conv_name, bn_name = name_group
        conv_obj = model_tools.get_object(model, conv_name)
        bn_obj = model_tools.get_object(model, bn_name)
        require_observer = bn_obj.require_observer
        conv_bn = intrinsic.ConvBn2d(conv_obj, bn_obj)
        conv_bn.qconfig = qconfig
        conv_bn_quant = nniqat.ConvBn2d.from_float(conv_bn)
        if require_observer:
            _regist_post_observer(conv_bn_quant, qconfig.activation())
        model = model_tools.set_object(model, conv_name, conv_bn_quant)
        model = model_tools.set_object(model, bn_name, nn.Identity())

    # convert conv
    # if insert_bn is true the bn layer will be inserted
    conv_names = torchpruner.model_tools.get_names_by_class(
        model, nn.Conv2d, include_super_class=False
    )
    for conv_name in conv_names:
        conv_obj = model_tools.get_object(model, conv_name)
        conv_obj.qconfig = qconfig
        conv_quant = nnqat.Conv2d.from_float(conv_obj)
        if conv_obj.require_observer:
            _regist_post_observer(conv_quant, qconfig.activation())
        model = model_tools.set_object(model, conv_name, conv_quant)

    # add observer for the other
    for module in model.modules():
        if hasattr(module, "require_observer") and module.require_observer == True:
            _regist_post_observer(module, qconfig.activation())

    # add observer for the input
    _regist_pre_observer(model, qconfig.activation())
    return model


def merge_convbn2d(model):
    module_names = model_tools.get_names_by_class(model, nniqat.ConvBn2d)
    for module_name in module_names:
        module = model_tools.get_object(model, module_name)
        if isinstance(module, nniqat.ConvBn2d):
            conv = merge_conv_bn(module, module.bn)
            conv.qconfig = module.qconfig
            weight_fake_quant = module.weight_fake_quant
            conv_quant = nnqat.Conv2d.from_float(conv)
            conv_quant.weight_fake_quant = weight_fake_quant
            conv_quant.to(conv.weight.device)
            if hasattr(module, "_post_observer"):
                _regist_post_observer(conv_quant, module._post_observer)
            model = model_tools.set_object(model, module_name, conv_quant)
    return model


def input_name_to_node(onnx_model):
    input_name_to_nodes = {}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name not in input_name_to_nodes:
                input_name_to_nodes[input_name] = [node]
            else:
                input_name_to_nodes[input_name].append(node)
    return input_name_to_nodes


def output_name_to_node(onnx_model):
    output_name_to_node = {}
    for node in onnx_model.graph.node:
        for output_name in node.output:
            output_name_to_node[output_name] = node
    return output_name_to_node


def get_children(onnx_model, node):
    input_name_to_node_dict = input_name_to_node(onnx_model)
    children = []
    for output in node.output:
        if output in input_name_to_node_dict:
            for node in input_name_to_node_dict[output]:
                children.append(node)
    return children


def get_parents(onnx_model, node):
    output_name_to_node_dict = output_name_to_node(onnx_model)
    parents = []
    for input in node.input:
        if input in output_name_to_node_dict:
            parents.append(output_name_to_node_dict[input])
    return parents


def get_attribute_value(node, name):
    for attr in node.attribute:
        if attr.name == name:
            value = onnx.helper.get_attribute_value(attr)
            return value
    raise RuntimeError("Can not find the attribute")


def intable(value, mins):
    try:
        number = int(value) - mins
        if number < 0:
            return value
        result = str(number)
        return result
    except Exception as e:
        return value


def onnx_post_process(onnx_model):
    # replace the global average pooling
    # replace the average when the final size is 1 1
    new_nodes = []
    remove_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "GlobalAveragePool":
            remove_nodes.append(node)
            name = node.name
            name = name.replace("GlobalAveragePool", "ReduceMean")
            new_node = onnx.helper.make_node(
                "ReduceMean",
                inputs=node.input,
                outputs=node.output,
                name=name,
                axes=[2, 3],
            )
            new_nodes.append(new_node)
        if node.op_type == "Conv" or node.op_type == "Add":
            children = get_children(onnx_model, node)
            if len(children) != 1:
                continue
            if children[0].op_type == "Relu":
                continue
            if children[0].op_type == "QuantizeLinear":
                quantize_node = children[0]
                children = get_children(onnx_model, quantize_node)
                if len(children) != 1 or children[0].op_type != "DequantizeLinear":
                    raise RuntimeError(
                        "the DequantizeLinear must be with QuantizeLinear"
                    )
                dequantize_node = children[0]
                children = get_children(onnx_model, dequantize_node)
                if len(children) == 1 and children[0].op_type == "Relu":
                    # remove the node after conv if has relu
                    relu_node = children[0]
                    children = get_children(onnx_model, relu_node)
                    if len(children) == 1 and children[0].op_type == "QuantizeLinear":
                        quantize_parents = get_parents(onnx_model, quantize_node)
                        remove_nodes.append(quantize_parents[1])
                        remove_nodes.append(quantize_parents[2])
                        dequantize_parents = get_parents(onnx_model, dequantize_node)
                        remove_nodes.append(dequantize_parents[1])
                        remove_nodes.append(dequantize_parents[2])
                        remove_nodes.append(quantize_node)
                        remove_nodes.append(dequantize_node)
                        relu_node.input[0] = node.output[0]
                    else:
                        raise RuntimeError("the QDQ must be added after Relu")
    # add the new node
    onnx_model.graph.node.extend(new_nodes)
    # remove the node
    for node in remove_nodes:
        if node in onnx_model.graph.node:
            onnx_model.graph.node.remove(node)

    # change the quantize and dequantize constant op to initilzer
    new_initizer = []
    remove_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear":
            parents = get_parents(onnx_model, node)
            indexing = len(parents) - 2
            scale_node = parents[indexing]
            scale_value = get_attribute_value(scale_node, "value")
            zp_node = parents[indexing + 1]
            zp_value = get_attribute_value(zp_node, "value")
            scale_initizer = onnx.helper.make_tensor(
                scale_node.output[0],
                onnx_proto.TensorProto.FLOAT,
                dims=[],
                vals=scale_value.raw_data,
                raw=True,
            )
            zp_initizer = onnx.helper.make_tensor(
                zp_node.output[0],
                onnx_proto.TensorProto.INT8,
                dims=[],
                vals=zp_value.raw_data,
                raw=True,
            )
            new_initizer.append(scale_initizer)
            new_initizer.append(zp_initizer)
            remove_nodes.append(scale_node)
            remove_nodes.append(zp_node)

    # add initizer
    for node in new_initizer:
        if node not in onnx_model.graph.initializer:
            onnx_model.graph.initializer.append(node)
    # remove the node
    for node in remove_nodes:
        if node in onnx_model.graph.node:
            onnx_model.graph.node.remove(node)


def export_onnx(model, inputs, onnx_file):
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, torch.quantization.FakeQuantize):
            module.calculate_qparams()
    model.apply(torch.quantization.disable_observer)
    torch.onnx.export(
        model,
        inputs,
        onnx_file,
        **model_tools.normalize_onnx_parameters(
            opset_version=10,
            verbose=False,
            enable_onnx_checker=False,
            _retain_param_name=True,
        )
    )
    onnx_model = onnx.load(onnx_file)
    onnx_post_process(onnx_model)
    onnx.save(onnx_model, onnx_file)


def export_trt(onnx_file):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    )

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        flags=network_flags
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file,")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
            # preprocess_network(network)

            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Fail to build the trt model")
            return engine
