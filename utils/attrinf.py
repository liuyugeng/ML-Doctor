import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import *
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score


class attack_training():
    def __init__(self, device, attack_trainloader, attack_testloader, target_model, TARGET_PATH, ATTACK_PATH):
        self.device = device
        self.TARGET_PATH = TARGET_PATH
        self.ATTACK_PATH = ATTACK_PATH

        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.target_model.eval()

        self.attack_model = None

        self.attack_trainloader = attack_trainloader
        self.attack_testloader = attack_testloader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    def _get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def init_attack_model(self, output_classes):
        x = torch.rand([1, 3, 64, 64]).to(self.device)
        input_classes = self.get_middle_output(x).flatten().shape[0]
        self.attack_model = get_attack_model(inputs_classes=input_classes, outputs_classes=output_classes)
        self.attack_model.to(self.device)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)

    def get_middle_output(self, x):
        temp = []
        for name, _ in self.target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if 1 > len(temp):
            raise IndexError('layer is out of range')

        name = temp[-1].split('.')
        var = eval('self.target_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation("attr", out))
        _ = self.target_model(x)
        
        return out["attr"]

    # Training
    def train(self):
        self.attack_model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, [_, targets]) in enumerate(self.attack_trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            oracles = self.get_middle_output(inputs)
            outputs = self.attack_model(oracles)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # self.scheduler.step()

        final_result = 1.*correct/total
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self):
        self.attack_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, [_, targets] in self.attack_testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                oracles = self.get_middle_output(inputs)
                outputs = self.attack_model(oracles)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        final_result = 1.*correct/total
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)