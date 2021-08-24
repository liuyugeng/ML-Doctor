import os
from numpy.lib.function_base import gradient
import torch
import torch.nn as nn
import torch.optim as optim

from doctor.meminf import *
from doctor.attrinf import *
from demoloader.train import *
from utils.define_models import *
from demoloader.dataloader import *


def train_model(PATH, device, model, train_set, test_set, name):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)
    
    model = model_training(train_loader, test_loader, model, device, 0, 0, 0)
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

    filename = name + "_target.pth"
    FILE_PATH = PATH + filename
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

def test_meminf(num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model):
    # target_trainloader = torch.utils.data.DataLoader(
    #     target_train, batch_size=64, shuffle=True, num_workers=2)
    # target_testloader = torch.utils.data.DataLoader(
    #     target_test, batch_size=64, shuffle=True, num_workers=2)

    # shadow_trainloader = torch.utils.data.DataLoader(
    #     shadow_train, batch_size=64, shuffle=True, num_workers=2)
    # shadow_testloader = torch.utils.data.DataLoader(
    #     shadow_test, batch_size=64, shuffle=True, num_workers=2)
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

    attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size=64)
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
    attack_model = PartialAttackModel(num_classes)

    attack_mode1(TARGET_PATH + name + "_target.pth", TARGET_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes)

def test_attrinf(num_classes, target_trainloader, target_testloader, target_model):
    pass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    TARGET_PATH = "./demoloader/trained_model/"

    name = "UTKFace"
    attr = ["race", "gender"]
    root = "../data"

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root)

    # train_model(TARGET_PATH, device, target_model, target_train, target_test, name)
    # test_meminf(num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model)

    
    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - attack_length
    attack_model = attrinf_attack_model()

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(target_train[0][0].shape)
    train_attack_model(TARGET_PATH + name + "_target.pth", TARGET_PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, image_size, attack_model)

    
