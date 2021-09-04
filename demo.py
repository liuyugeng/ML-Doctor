import os
import torch
import torch.nn as nn

from doctor.meminf import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
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

def test_meminf(num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, device):
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
    # train_shadow_model(TARGET_PATH, device, num_classes, target_model, shadow_trainloader, shadow_testloader, 0, 0, 0)
    

    attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size=64)

    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    
    attack_model = PartialAttackModel(num_classes)

    attack_mode1(TARGET_PATH + name + "_target.pth", TARGET_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes)

def test_attrinf(num_classes, target_train, target_test, target_model, device):
    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - attack_length

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(target_train[0][0].shape)
    train_attack_model(TARGET_PATH + name + "_target.pth", TARGET_PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, image_size)

def test_modsteal(train, test, target_model, attack_model, device, PATH, name):
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=64, shuffle=True, num_workers=2)

    TARGET_PATH = PATH + name + "_target.pth"
    ATTACK_PATH = PATH + name + "_modsteal.pth"

    loss = nn.MSELoss()
    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9)

    attacking = train_steal_model(train_loader, test_loader, target_model, attack_model, TARGET_PATH, ATTACK_PATH, device, 64, loss, optimizer)

    for i in range(100):
        print("[Epoch %d/%d] attack training"%((i+1), 100))
        attacking.train_with_same_distribution()
    
    print("Finished training!!!")
    attacking.saveModel()
    acc_test, agreement_test = attacking.test()
    print("Saved Target Model!!!\nstolen test acc = %.3f, stolen test agreement = %.3f\n"%(acc_test, agreement_test))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    TARGET_PATH = "./demoloader/trained_model/"

    name = "UTKFace"
    attr = "race"
    root = "../data"

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(name, attr, root)

    # train_model(TARGET_PATH, device, target_model, target_train, target_test, name)
    # test_meminf(num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model)
    # test_attrinf(num_classes, target_train, target_test, target_model)
    # test_modsteal(shadow_train+shadow_test, target_test, target_model, shadow_model, device, TARGET_PATH, name)


    

    
