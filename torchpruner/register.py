from collections import OrderedDict


class module_pruner_register(object):
    def __init__(self):
        self.regist_dict = OrderedDict()

    def regist(self, cls, module):
        self.regist_dict[cls] = module

    def get(self, cls):
        if cls in self.regist_dict.keys():
            return self.regist_dict[cls]
        return None


module_pruner_reg = module_pruner_register()

# operator register, onnx reegister will be replaced
class operator_register(object):
    def __init__(self):
        self.regist_dict = OrderedDict()
        self.version = 9
        self.regist_dict["all"] = OrderedDict()

    def regist(self, name, obj, version=[]):
        if len(version) == 0:
            if name not in self.regist_dict["all"].keys():
                self.regist_dict["all"][name] = obj
                return
            else:
                raise RuntimeError("The " + str(name) + " has already been registed")
        for v in version:
            if str(v) not in self.regist_dict.keys():
                self.regist_dict[str(v)] = OrderedDict()
            if name not in self.regist_dict[str(v)].keys():
                self.regist_dict[str(v)][name] = obj
            else:
                raise RuntimeError("The " + str(name) + " has already been registed")

    def get(self, name):
        if str(self.version) not in self.regist_dict.keys():
            if name in self.regist_dict["all"].keys():
                return self.regist_dict["all"][name]
        if str(self.version) not in self.regist_dict:
            return None
        if name in self.regist_dict[str(self.version)].keys():
            return self.regist_dict[str(self.version)][name]
        if name in self.regist_dict["all"].keys():
            return self.regist_dict["all"][name]
        return None

    def set_version(self, version):
        self.version = version


operator_reg = operator_register()
