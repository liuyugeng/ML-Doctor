import os
import sys
import time
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from train import *
from dataloader import *

def train_model(PATH, device, model, train_loader, test_loader, num_classes):
    model = model_training(train_loader, test_loader, target_model, device, 0, num_classes, 0, 0)
    acc_train = 0
    acc_test = 0

    for i in range(300):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        print("target training")

        acc_train = model.train()
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

    filename = "target.pth"
    FILE_PATH = PATH + filename
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda:0")

    num_classes, target_trainloader, target_testloader, shadow_trainloader, shadow_testloader, target_model, shadow_model = prepare_dataset("UTKFace", "race")

    TARGET_PATH = "./trained_model/"
    acc_target_train, acc_target_test, overfitting_target = train_model(TARGET_PATH, device, target_model, target_trainloader, target_testloader, num_classes)
