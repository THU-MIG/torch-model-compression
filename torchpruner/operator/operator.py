from typing import List

import onnxruntime.backend
import onnx
import onnxruntime
import torch
from collections import OrderedDict
from onnx import AttributeProto, TensorProto, GraphProto, helper, shape_inference
import torchpruner.register as register
from onnx import numpy_helper
from onnx import version_converter
import numpy as np
import onnxruntime.backend

torch2onnx_type_mapping = {
    "Float": TensorProto.FLOAT,
    "UINT8": TensorProto.UINT8,
    "INT8": TensorProto.INT8,
    "UINT16": TensorProto.UINT16,
    "INT16": TensorProto.INT16,
    "INT32": TensorProto.INT32,
    "Long": TensorProto.INT64,
    "Bool": TensorProto.BOOL,
    "Double": TensorProto.DOUBLE,
}

num2type_mapping = {
    "1": "Float",
    "2": "UINT8",
    "3": "INT8",
    "4": "UINT16",
    "5": "INT16",
    "6": "INT32",
    "7": "Long",
    "9": "Bool",
    "11": "Double",
}


def _extract_shape(value_info):
    shape_dict = OrderedDict()
    for info in value_info:
        name = info.name
        shape_type = str(info.type.tensor_type.elem_type)
        shape = info.type.tensor_type.shape
        shape_list = []
        for d in shape.dim:
            shape_list.append(d.dim_value)
        shape_dict[name] = (shape_type, shape_list)
    return shape_dict


class OperatorNode(object):
    def __init__(self, node):
        import torchpruner.graph as g

        # define the OperatorNode data structure
        self.name = None
        op_kind = node.kind().split("::")
        self.type = op_kind[1]
        self.protocal = op_kind[0]
        self.obj_list = node.scopeName().split(".")
        if len(self.obj_list) == 1 and self.obj_list[0] == "":
            self.obj_list = ["self"]
        self.name = ".".join(self.obj_list)
        self.name += "."
        self.name += self.type
        self.params = OrderedDict()
        self.device = "CPU"
        attlist = list(node.attributeNames())
        for i in range(0, len(attlist)):
            if isinstance(node[attlist[i]], torch.Tensor):
                self.params[attlist[i]] = node[attlist[i]].numpy()
            else:
                self.params[attlist[i]] = node[attlist[i]]
        # the operator node will be filled at the build graph
        self.in_data: List[g.DataNode] = []
        self.out_data: List[g.DataNode] = []

    def set_device(self, device):
        self.device = device.upper()

    def __str__(self):
        return_string = ""
        for node in self.out_data:
            return_string += str(node)
            return_string += ", "
        if len(return_string) != 0:
            return_string = return_string[:-2]
            return_string += " = "
        return_string += self.protocal
        return_string += "::"
        return_string += self.type
        return_string += "["
        for key in self.params.keys():
            return_string += key
            return_string += "="
            return_string += str(self.params[key])
            return_string += ", "
        if return_string[-1] != "[":
            return_string = return_string[:-2]
        return_string += "]("
        for node in self.in_data:
            return_string += "%" + node.name
            return_string += ", "
        if return_string[-1] != "(":
            return_string = return_string[:-2]
        return_string += ")"
        return_string += ", scope: "
        return_string += ".".join(self.obj_list)
        return return_string

    def __repr__(self):
        return self.__str__()

    def rank(self, node):
        out_nodes = self.out_data
        for i in range(0, len(out_nodes)):
            out_node = out_nodes[i]
            if id(node) == id(out_node):
                return "out", i
        in_nodes = self.in_data
        for i in range(0, len(in_nodes)):
            in_node = in_nodes[i]
            if id(node) == id(in_node):
                return "in", i

    def flops(self):
        return 0

    def fill_shape(self):
        need_fill = False
        out_data_nodes = self.out_data
        for node in out_data_nodes:
            if node.kind != "NoneType" and node.type() is None:
                need_fill = True
                break
        if need_fill:
            out_data_node_names = []
            for node in out_data_nodes:
                out_data_node_names.append(node.name)
            in_data_nodes = self.in_data
            in_data_node_names = []
            in_data_nodes_protos = []
            for node in in_data_nodes:
                in_data_node_names.append(node.name)
                in_data_nodes_protos.append(
                    helper.make_tensor_value_info(
                        node.name,
                        torch2onnx_type_mapping[node.type()],
                        list(node.size()),
                    )
                )
            operator_params = {}
            for key in self.params:
                if isinstance(self.params[key], np.ndarray):
                    operator_params[key] = numpy_helper.from_array(self.params[key])
                else:
                    operator_params[key] = self.params[key]
            node_def = helper.make_node(
                self.type,  # node name
                in_data_node_names,  # inputs
                out_data_node_names,  # outputs
                **operator_params
            )
            graph_def = helper.make_graph(
                [node_def], "node-graph", in_data_nodes_protos, []
            )
            model = helper.make_model(
                graph_def,
                producer_name="node-model",
                opset_imports=[helper.make_opsetid("", 11)],
            )
            try:
                inferred_model = shape_inference.infer_shapes(model)
            except Exception as e:
                print(e)
                print(model)
            value_info = inferred_model.graph.value_info
            shape_dict = _extract_shape(value_info)
            shape_dict_keys = list(shape_dict.keys())
            for i in range(0, len(out_data_node_names)):
                name = out_data_node_names[i]
                node = out_data_nodes[i]
                if node.type() is None:
                    if name not in shape_dict_keys:
                        raise RuntimeError(
                            "Fail to predict the shape on operator name: '"
                            + self.name
                            + "', type: '"
                            + self.type
                            + "'"
                        )
                    else:
                        node._type = num2type_mapping[shape_dict[name][0]]
                        node._size = shape_dict[name][1]

    def fill_value(self):
        # fix the torch operator error
        if self.type == "Equal":
            if len(self.out_data) != 0:
                self.out_data[0]._type = "Bool"

        out_data_nodes = self.out_data
        out_data_node_names = []
        out_data_node_protos = []
        for node in out_data_nodes:
            if node.kind == "NoneType":
                return
            out_data_node_names.append(node.name)
            out_data_node_protos.append(
                helper.make_tensor_value_info(
                    node.name, torch2onnx_type_mapping[node.type()], list(node.size())
                )
            )
        in_data_nodes = self.in_data
        in_data_node_names = []
        in_data_node_protos = []
        feed_dict = {}
        inputs = []
        for node in in_data_nodes:
            if node.kind == "NoneType":
                continue
            in_data_node_names.append(node.name)
            in_data_node_protos.append(
                helper.make_tensor_value_info(
                    node.name, torch2onnx_type_mapping[node.type()], list(node.size())
                )
            )
            feed_dict[node.name] = node.data
            inputs.append(node.data)
        operator_params = {}
        for key in self.params:
            if isinstance(self.params[key], np.ndarray):
                operator_params[key] = numpy_helper.from_array(self.params[key])
            else:
                operator_params[key] = self.params[key]
        node_def = helper.make_node(
            self.type,  # node name
            in_data_node_names,  # inputs
            out_data_node_names,  # outputs
            **operator_params
        )

        graph_def = helper.make_graph(
            [node_def], "node-graph", in_data_node_protos, out_data_node_protos
        )

        model = helper.make_model(
            graph_def,
            producer_name="node-model",
            opset_imports=[helper.make_opsetid("", 11)],
        )
        onnx.checker.check_model(model)
        s = onnx._serialize(model)
        sess = onnxruntime.backend.prepare(s, device=self.device)
        results = onnxruntime.backend.run(sess, inputs)

        for i in range(0, len(out_data_node_names)):
            # name=out_data_node_names[i]
            out_data_nodes[i].data = results[i]
            if out_data_nodes[i]._size != list(results[i].shape):
                out_data_nodes[i]._size = list(results[i].shape)

    def analysis(self, node, mask):
        raise NotImplementedError("The analysis is not complete")
