import os
import sys
import torch
import argparse

from meminf import *
from attrinf import *
from utils.define_models import *



def meminf():
    pass

def modinv():
    pass

def attrinf():
    pass

def modsteal():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=str, default="1", 
                        help='Choose GPU')

    parser.add_argument('-a', '--attack', type=int, default="0", 
                        help='Choose attack and we have 4 attacks\
                        0: membership inference, 1: model inversion 2: attribute inference, 3: model stealing')

    parser.add_argument('-t', '--type', type=int, default="0", 
                        help='Choose attack type and we have different attack types\
                        for membership inference, we have 4 types\
                        for model inversion, we have 2 types\
                        for attribute inference, we have 2 types\
                        for model stealing, we have 2 types')


    args = parser.parse_args()

    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')

    attack = args.attack

    if not attack:
        meminf()
    
    elif attack == 1:
        modinv()

    elif attack == 2:
        attrinf()

    elif attack == 3:
        modsteal()

    else:
        sys.exit("we have not supported this attack yet! 0c0")

