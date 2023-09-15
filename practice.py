import sys
import os

import numpy as np

# dir_name = os.path.dirname(__file__)
# print(dir_name)
# sys.path.insert(0, os.path.join(dir_name, ".."))
# print(sys.path)
# print(sys.argv)


def torch_tnt():
    import torch
    import torchnet as tnt
    import numpy as np
    class_error = tnt.meter.ClassErrorMeter(accuracy=False)
    prediction = torch.tensor([1, 2, 2, 4, 5])
    target = torch.tensor([1, 2, 3, 4, 6])
    # if np.ndim(target) == 1:
    #     target = target[np.newaxis]
    # print(target)
    class_error.add(prediction, target)
    accuracy = class_error.value()[0]
    return accuracy


def save_model():
    import torch
    torch.save({"batch": 0, "result": []}, os.path.join("./", "checkpoint.path.tar"))


def save_state():
    import torch
    from torchvision import models
    model = models.segmentation.FCN()
    state_dict = torch.load(".pth")
    model.load_state_dict(state_dict)   # 给model加载参数
    torch.save(model.state_dict(), ".pth")


def logger():
    import logging
    logging.getLogger().setLevel(logging.INFO)
    print(logging.INFO)
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        print(logging.getLogger().getEffectiveLevel())
        print(logging.DEBUG)
        print("yes")
    else:
        print("no")


def func_torch():
    import torch
    a = torch.ones((3, 8, 2))
    b = torch.ones((3, 2, 8))
    c = torch.bmm(b, a)
    print(c.shape)


# import shutil
# file_dir = "data/Stanford3dDataset_v1.2_Aligned_Version"
# file_list = os.listdir(file_dir)
# save_dir = os.path.join(file_dir, "stanford_indoor3d")
# for file in file_list:
#     file_path = os.path.join(file_dir, file)
#     if ".npy" in file:
#         shutil.move(file_path, save_dir)


def cal_grad():
    import torch
    data = torch.arange(1.0, 10.0, 1.0, requires_grad=True)
    data = torch.tensor(data, requires_grad=True)
    print(data)
    y = torch.sigmoid(data)
    y = data * data
    print(y)
    y.backward(torch.ones_like(data))
    print(data.grad)
cal_grad()


def cal_params():
    import torch
    w = torch.nn.Parameter(torch.randn((10, 10), requires_grad=True))
    x = torch.randn(10, 10)
    print(w@x)
    print(w)


def init_param():
    from torch import nn
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    def init_weight(m):
        if type == nn.Linear:
            nn.init.normal_(m.weight, std=0.1)
    net.apply(init_weight)

# import numpy as np
# a = np.power((1, 2, 3, 4), (2, 3))
# print(a)
# np.random.shuffle(a)