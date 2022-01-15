import copy
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def get_time_string():
    dt = datetime.now()
    return dt.strftime("%Y%m%d%H%M")


class SlimSolver(object):
    # (name,type,default,optional,description)
    __config_setting__ = []

    @classmethod
    def print_config(cls, help=False):
        for setting in cls.__config_setting__:
            current_setting = []
            current_setting.append(setting[0])
            if "'" in str(setting[1]):
                current_setting.append(str(setting[1]).split("'")[1])
            else:
                current_setting.append(str(setting[1]))
            current_setting.append(setting[2])
            if setting[3]:
                current_setting.append("optional")
            else:
                current_setting.append("required")
            if not help:
                current_setting = tuple(current_setting)
                print("%s: %s default:%s %s " % current_setting)
            else:
                print("%s: %s" % (setting[0], setting[4]))

    def __init__(self, model, config):
        self.model = model
        config = self._set_defualt_config(copy.deepcopy(config))
        self._check_config(config)
        self.config = config
        self._print_config(config)

    def _set_defualt_config(self, config):
        for setting in self.__config_setting__:
            if setting[0] not in config:
                config[setting[0]] = setting[2]
        return config

    def _check_config(self, config):
        for setting in self.__config_setting__:
            if setting[3] != True and config[setting[0]] == None:
                raise RuntimeError("The config " + str(setting[0]) + " is required")

            if setting[1] == "function" and not callable(config[setting[0]]):
                raise RuntimeError(
                    "The type function is required, but got "
                    + str(type(config[setting[0]]) + "in config " + str(setting[0]))
                )
            else:
                continue
            if config[setting[0]] != None and not isinstance(
                config[setting[0]], setting[1]
            ):
                raise RuntimeError(
                    "The type "
                    + str(setting[1])
                    + " is required, but got "
                    + str(type(config[setting[0]]) + "in config " + str(setting[0]))
                )

    def _print_config(self, config):
        print("The config is:")
        for key in config.keys():
            print(key + ": " + str(config[key]))

    def run(self):
        raise NotImplementedError("The run function is not implemented")


class AvgMeter:
    def __init__(self):
        self.counter = 0
        self.avg_dict = {}

    def update(self, evalute_dict, count):
        if self.counter == 0:
            self.avg_dict = evalute_dict
            self.counter = count
            return
        for key in evalute_dict.keys():
            self.avg_dict[key] = (
                self.avg_dict[key] * self.counter + evalute_dict[key] * count
            )
            self.avg_dict[key] = self.avg_dict[key] / (self.counter + count)
        self.counter += count

    def get(self):
        return self.avg_dict


class CommonSlimSolver(SlimSolver):
    __config_setting__ = [
        ("task_name", str, "default", False, "The task name"),
        ("epoch", int, 360, False, "The total epoch to train the model"),
        ("batch_size", int, 128, False, "The batch size per step"),
        ("test_batch_size", int, 128, False, "The evaluation batch size per step"),
        ("num_workers", int, 0, False, "The number of workers to read data"),
        ("devices", list, None, False, "The device to be used in training"),
        ("log_interval", int, 100, False, "The interval to report the log"),
        ("save_keyword", str, "acc", False, "The keyword for save"),
        ("save_dir", str, "checkpoints", False, "The model save dir"),
        # generate the optimizer
        (
            "optimizer_generator",
            "function",
            None,
            False,
            "The optimizer generator (params)->optimizer",
        ),
        # generate the scheduler
        (
            "scheduler_generator",
            "function",
            None,
            True,
            "the scheduler generator for the task (optmizer,config)->scheduler",
        ),
        # predict the result
        (
            "predict_function",
            "function",
            None,
            False,
            "get the prediction of the data (model,batch_data)->predict",
        ),
        # calculate the loss for one iteration
        (
            "calculate_loss_function",
            "function",
            None,
            False,
            "(predict,batch_data)->loss",
        ),
        # get the evaluate result for one iteration
        (
            "evaluate_function",
            "function",
            None,
            True,
            "(predict,batch_data)->evaluate_dict",
        ),
        # get the dataset
        (
            "dataset_generator",
            "function",
            None,
            True,
            "()->dataset_train,dataset_validation",
        ),
    ]

    def __init__(self, model, config):
        super(CommonSlimSolver, self).__init__(model, config)
        self.variable_dict = {}
        self.variable_dict["epoch"] = 0
        self.variable_dict["iteration"] = 0
        self.variable_dict["step"] = 0
        self.variable_dict["allow_save"] = True

        self.init_hook = []
        self.end_hook = []
        self.epoch_begin_hook = []
        self.epoch_end_hook = []
        self.iteration_begin_hook = []
        self.iteration_end_hook = []

        self.writer = SummaryWriter(
            "runs/" + self.config["task_name"] + "-" + get_time_string()
        )

    def regist_init_hook(self, function):
        self.init_hook.append(function)

    def regist_end_hook(self, function):
        self.end_hook.append(function)

    def regist_epoch_begin_hook(self, function):
        self.epoch_begin_hook.append(function)

    def regist_epoch_end_hook(self, function):
        self.epoch_end_hook.append(function)

    def regist_iteration_begin_hook(self, function):
        self.iteration_begin_hook.append(function)

    def regist_iteration_end_hook(self, function):
        self.iteration_end_hook.append(function)

    def on_loss_backward(self):
        pass

    def after_loss_backward(self):
        pass

    def save_model(self):
        torch.save(
            {
                self.config["save_keyword"]: self.variable_dict["save_target"],
                "net": self.model.cpu(),
            },
            self.variable_dict["save_path"],
        )

    def run_hook(self, hook_list):
        for function in hook_list:
            function(self)

    def generate_params_setting(self):
        return self.model.parameters()

    def _sample_to_device(self, data, device):
        if isinstance(data, (tuple, list)):
            return_list = []
            for item in data:
                return_list.append(item.to(device))
            return tuple(return_list)
        if isinstance(data, dict):
            return_dict = {}
            for key in data.keys():
                return_dict[key] = data[key].to(device)
            return return_dict
        return data.to(device)

    def _get_sample_number(self, data):
        if isinstance(data, (tuple, list)):
            return data[0].size(0)
        if isinstance(data, dict):
            for key in data.keys():
                return data[key].size(0)
        return data.size(0)

    # the mode is iteration train or test
    def write_tensorboard(self, prefix, iteration, result_dict):
        for key in result_dict.keys():
            self.writer.add_scalar(
                "scalar/" + prefix + "_" + key, result_dict[key], iteration
            )

    def write_log(self, epoch, iteration, result_dict):
        content = "| "
        dt = datetime.now()
        content += dt.strftime("%H:%M:%S")
        content += " | "
        content += "epoch:"
        content += str(epoch)
        content += " | "
        content += "iteration:"
        content += str(iteration)
        content += " | "
        for key in result_dict.keys():
            add_value = "{:s}:{:.4f} | ".format(key, result_dict[key])
            content += add_value
        print(content)

    def run(self):
        # ensure model in eval mode
        self.model.eval()
        # prepare dataset
        print("preparing dataset...")
        dataset_train, dataset_validation = self.config["dataset_generator"]()
        self.trainloader = DataLoader(
            dataset_train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        self.valloader = DataLoader(
            dataset_validation,
            batch_size=self.config["test_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

        # model to device
        # check the model DataParallel
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module

        print("The device is :" + str(self.config["devices"]))
        # set the device
        if len(self.config["devices"]) == 0:
            raise RuntimeError("The device number must be set")
        self.variable_dict["base_device"] = "cuda:" + str(self.config["devices"][0]) \
            if int(self.config["devices"][0]) >= 0 else "cpu:0"

        # init the runner
        self.run_hook(self.init_hook)

        self.model = self.model.to(self.variable_dict["base_device"])
        if len(self.config["devices"]) >= 2:
            self.model = nn.DataParallel(self.model, device_ids=self.config["devices"])

        params = self.generate_params_setting()
        self.optimizer = self.config["optimizer_generator"](params, self.config)
        self.scheduler = self.config["scheduler_generator"](self.optimizer, self.config)

        # begin epoch
        print("training...")
        for epoch in range(0, self.config["epoch"]):
            self.variable_dict["epoch"] = epoch
            self.run_hook(self.epoch_begin_hook)
            self.variable_dict["avg_mentor"] = AvgMeter()
            self.model.train()
            for step, data in enumerate(self.trainloader):
                self.variable_dict["step"] = step + 1
                self.variable_dict["iteration"] += 1
                self.run_hook(self.iteration_begin_hook)
                data = self._sample_to_device(data, self.variable_dict["base_device"])
                c_sample_number = self._get_sample_number(data)
                predict = self.config["predict_function"](self.model, data)
                self.variable_dict["loss"] = self.config["calculate_loss_function"](
                    predict, data
                )
                self.on_loss_backward()
                self.variable_dict["loss"].backward()
                self.after_loss_backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                evaluate_result = self.config["evaluate_function"](predict, data)
                evaluate_result["loss"] = self.variable_dict["loss"].item()
                self.variable_dict["avg_mentor"].update(
                    evaluate_result, c_sample_number
                )
                if self.variable_dict["iteration"] % self.config["log_interval"] == 0:
                    self.write_log(
                        self.variable_dict["epoch"],
                        self.variable_dict["iteration"],
                        self.variable_dict["avg_mentor"].get(),
                    )
                    self.write_tensorboard(
                        "train_log",
                        self.variable_dict["iteration"],
                        self.variable_dict["avg_mentor"].get(),
                    )
                self.run_hook(self.iteration_end_hook)
                del self.variable_dict["loss"]
            self.scheduler.step()

            # evaluation step
            print("evaluating...")
            self.variable_dict["test_avg_mentor"] = AvgMeter()
            self.model.eval()
            for step, data in enumerate(self.valloader):
                self.variable_dict["step"] = step + 1
                data = self._sample_to_device(data, self.variable_dict["base_device"])
                c_sample_number = self._get_sample_number(data)
                predict = self.config["predict_function"](self.model, data)
                evaluate_result = self.config["evaluate_function"](predict, data)
                evaluate_result["loss"] = self.config["calculate_loss_function"](
                    predict, data
                ).item()
                self.variable_dict["test_avg_mentor"].update(
                    evaluate_result, c_sample_number
                )
                if self.variable_dict["step"] % self.config["log_interval"] == 0:
                    self.write_log(
                        self.variable_dict["epoch"],
                        self.variable_dict["step"],
                        self.variable_dict["test_avg_mentor"].get(),
                    )
            self.write_log(
                self.variable_dict["epoch"],
                "final",
                self.variable_dict["test_avg_mentor"].get(),
            )
            self.write_tensorboard(
                "test_log",
                self.variable_dict["epoch"],
                self.variable_dict["test_avg_mentor"].get(),
            )

            # save model is allow save is True
            if self.variable_dict["allow_save"]:
                # save the model
                if "save_target" not in self.variable_dict:
                    print("Saving model...")
                    self.variable_dict["save_target"] = self.variable_dict[
                        "test_avg_mentor"
                    ].get()[self.config["save_keyword"]]
                    if not os.path.exists(self.config["save_dir"]):
                        os.mkdir(self.config["save_dir"])
                    if not os.path.exists(
                        os.path.join(self.config["save_dir"], self.config["task_name"])
                    ):
                        os.mkdir(
                            os.path.join(
                                self.config["save_dir"], self.config["task_name"]
                            )
                        )
                    save_path = os.path.join(
                        self.config["save_dir"], self.config["task_name"]
                    )
                    save_path = os.path.join(save_path, "model.pth")
                    self.variable_dict["save_path"] = save_path
                    self.save_model()
                else:
                    current_target = self.variable_dict["save_target"]
                    result_target = self.variable_dict["test_avg_mentor"].get()[
                        self.config["save_keyword"]
                    ]
                    if current_target < result_target:
                        print("Saving model...")
                        self.variable_dict["save_target"] = result_target
                        self.save_model()
            else:
                print("Saving the model is not allowed currently")
            # end epoch hook
            self.run_hook(self.epoch_end_hook)

        # end the runner
        self.run_hook(self.end_hook)
