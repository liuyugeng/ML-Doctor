import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.models as models

from doctor.meminf import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *


def train_model(PATH, device, train_set, test_set, model, use_DP, noise, norm, delta):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)
    
    model = model_training(train_loader, test_loader, model, device, use_DP, noise, norm, delta)
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

    FILE_PATH = PATH + "_target.pth"
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting

def train_DCGAN(PATH, device, train_set, name):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)

    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    print("Starting Training DCGAN...")
    GAN = GAN_training(train_loader, D, G, device)
    for i in range(200):
        print("<======================= Epoch " + str(i+1) + " =======================>")
        GAN.train()

    GAN.saveModel(PATH + "_discriminator.pth", PATH + "_generator.pth")

def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_shadow, use_DP, noise, norm, delta):
    batch_size = 64
    if train_shadow:
        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        train_shadow_model(PATH, device, shadow_model, shadow_trainloader, shadow_testloader, use_DP, noise, norm, loss, optimizer, delta)

    # attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, batch_size)

    #for white box
    gradient_size = get_gradient_size(target_model)
    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    # attack_model = ShadowAttackModel(num_classes)
    # attack_model = PartialAttackModel(num_classes)
    attack_model = WhiteBoxAttackModel(num_classes, total)
    
    # attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)
    # attack_mode1(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes)
    # attack_mode2(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model, attack_model, 1, num_classes)
    attack_mode3(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, 
        attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, 1, num_classes)

def test_modinv(PATH, device, num_classes, target_train, target_model, name):
    size = (1,) + tuple(target_train[0][0].shape)
    target_model, evaluation_model = load_data(PATH + "_target.pth", PATH + "_eval.pth", target_model, models.resnet18(num_classes=num_classes))

    # CCS 15
    modinv_ccs = ccs_inversion(target_model, size, num_classes, 1, 3000, 100, 0.001, 0.003, device)
    train_loader = torch.utils.data.DataLoader(target_train, batch_size=1, shuffle=False)
    ccs_result = modinv_ccs.reverse_mse(train_loader)

    # Secret Revealer

    if name.lower() == 'fmnist':
        D = FashionDiscriminator(ngpu=1).eval()
        G = FashionGenerator(ngpu=1).eval()
    else:
        D = Discriminator(ngpu=1).eval()
        G = Generator(ngpu=1).eval()

    PATH_D = PATH + "_discriminator.pth"
    PATH_G = PATH + "_generator.pth"
    
    D, G, iden = prepare_GAN(name, D, G, PATH_D, PATH_G)
    modinv_revealer = revealer_inversion(G, D, target_model, evaluation_model, iden, device)

def test_attrinf(PATH, device, num_classes, target_train, target_test, target_model):
    attack_length = int(0.5 * len(target_train))
    rest = len(target_train) - attack_length

    attack_train, _ = torch.utils.data.random_split(target_train, [attack_length, rest])
    attack_test = target_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=64, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=64, shuffle=True, num_workers=2)

    image_size = [1] + list(target_train[0][0].shape)
    train_attack_model(
        PATH + "_target.pth", PATH, num_classes[1], device, target_model, attack_trainloader, attack_testloader, image_size)

def test_modsteal(PATH, device, train_set, test_set, target_model, attack_model):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=2)

    loss = nn.MSELoss()
    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9)

    attacking = train_steal_model(
        train_loader, test_loader, target_model, attack_model, PATH + "_target.pth", PATH + "_modsteal.pth", device, 64, loss, optimizer)

    for i in range(100):
        print("[Epoch %d/%d] attack training"%((i+1), 100))
        attacking.train_with_same_distribution()
    
    print("Finished training!!!")
    attacking.saveModel()
    acc_test, agreement_test = attacking.test()
    print("Saved Target Model!!!\nstolen test acc = %.3f, stolen test agreement = %.3f\n"%(acc_test, agreement_test))

def str_to_bool(string):
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default="0")
    parser.add_argument('-a', '--attributes', type=str, default="race", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('-dn', '--dataset_name', type=str, default="UTKFace")
    parser.add_argument('-at', '--attack_type', type=int, default=0)
    parser.add_argument('-tm', '--train_model', action='store_true')
    parser.add_argument('-ts', '--train_shadow', action='store_true')
    parser.add_argument('-ud', '--use_DP', action='store_true',)
    parser.add_argument('-ne', '--noise', type=float, default=1.3)
    parser.add_argument('-nm', '--norm', type=float, default=1.5)
    parser.add_argument('-d', '--delta', type=float, default=1e-5)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    dataset_name = args.dataset_name
    attr = args.attributes
    if "_" in attr:
        attr = attr.split("_")
    root = "../data"
    use_DP = args.use_DP
    noise = args.noise
    norm = args.norm
    delta = args.delta
    train_shadow = args.train_shadow
    TARGET_PATH = "./demoloader/trained_model/" + dataset_name

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, attr, root)

    if args.train_model:
        train_model(TARGET_PATH, device, target_train, target_test, target_model, use_DP, noise, norm, delta)

    # membership inference
    if args.attack_type == 0:
        test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, train_shadow, use_DP, noise, norm, delta)

    # model inversion
    elif args.attack_type == 1:
        train_DCGAN(TARGET_PATH, device, shadow_test + shadow_train, dataset_name)
        test_modinv(TARGET_PATH, device, num_classes, target_train, target_model, dataset_name)

    # attribut inference
    elif args.attack_type == 2:
        test_attrinf(TARGET_PATH, device, num_classes, target_train, target_test, target_model)

    # model stealing
    elif args.attack_type == 3:
        test_modsteal(TARGET_PATH, device, shadow_train+shadow_test, target_test, target_model, shadow_model)

    else:
        sys.exit("we have not supported this mode yet! 0c0")


    # target_model = models.resnet18(num_classes=num_classes)
    # train_model(TARGET_PATH, device, target_train + shadow_train, target_test + shadow_test, target_model)
    
if __name__ == "__main__":
    main()
    
