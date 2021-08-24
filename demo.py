import os
from numpy.lib.function_base import gradient
import torch
import torch.nn as nn
import torch.optim as optim

from utils.meminf import *
from demoloader.train import *
from utils.define_models import *
from demoloader.dataloader import *


def train_model(PATH, device, model, train_loader, test_loader):
    model = model_training(train_loader, test_loader, target_model, device, 0, 0, 0)
    acc_train = 0
    acc_test = 0

    for i in range(100):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")

        acc_train = model.train()
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

    filename = "UTKFace_target.pth"
    FILE_PATH = PATH + filename
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    name = "UTKFace"
    attr = "race"
    root = "../data"

    num_classes, target_trainloader, target_testloader, shadow_trainloader, shadow_testloader, target_model, shadow_model = prepare_dataset(name, attr, "../data")

    TARGET_PATH = "./demoloader/trained_model/"
    # acc_target_train, acc_target_test, overfitting_target = train_model(TARGET_PATH, device, target_model, target_trainloader, target_testloader, num_classes)
    # loss = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # batch_size = 64
    # shadow = shadow_model_training(shadow_trainloader, shadow_testloader, shadow_model, device, 0, 0, 0, batch_size, loss, optimizer)

    # acc_train = 0
    # acc_test = 0

    # for i in range(100):
    #     print("<======================= Epoch " + str(i+1) + " =======================>")
    #     print("shadow training")

    #     acc_train = shadow.train()
    #     print("shadow testing")
    #     acc_test = shadow.test()

    #     overfitting = round(acc_train - acc_test, 6)
    #     print('The overfitting rate is %s' % overfitting)

    # filename = name + "_shadow.pth"
    # FILE_PATH = TARGET_PATH + filename
    # shadow.saveModel(FILE_PATH)
    # print("Saved shadow model!!!")
    # print("Finished training!!!")

    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(target_trainloader, target_testloader, shadow_trainloader, shadow_testloader, batch_size=64)
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
    attack_model = WhiteBoxAttackModel(num_classes, total)

    attack_mode3(TARGET_PATH + name + "_target.pth", TARGET_PATH + name + "_shadow.pth", TARGET_PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 0, num_classes)
