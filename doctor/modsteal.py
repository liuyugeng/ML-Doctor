import torch
import torch.nn.functional as F

from math import *
from tqdm import tqdm

class train_steal_model():
    def __init__(self, train_loader, test_loader, target_model, attack_model, TARGET_PATH, ATTACK_PATH, device, batch_size, loss, optimizer):
        self.device = device
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.TARGET_PATH = TARGET_PATH
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH, map_location=self.device))
        self.target_model.eval()
        
        self.ATTACK_PATH = ATTACK_PATH
        self.attack_model = attack_model.to(self.device)

        self.criterion = loss
        self.optimizer = optimizer

        self.index = 0

        self.count = [0 for i in range(10)]
        self.dataset = []

    def train(self, train_set, train_out):
        self.attack_model.train()

        for inputs, targets in tqdm(zip(train_set, train_out)):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.attack_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()


    def train_with_same_distribution(self):
        self.attack_model.train()

        train_loss = 0
        correct = 0
        total = 0
        
        correct_target = 0
        total_target = 0

        for inputs, targets in tqdm(self.train_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            target_model_logit = self.target_model(inputs)
            _,target_model_output = target_model_logit.max(1)
            target_model_posterior = F.softmax(target_model_logit, dim=1)
            # print(inputs, targets)
            self.optimizer.zero_grad()
            outputs = self.attack_model(inputs)
            # outputs = F.softmax(outputs, dim=1)
            # loss = self.criterion(outputs, targets)
            loss = self.criterion(outputs, target_model_posterior)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_target += targets.size(0)
            correct_target += predicted.eq(target_model_output).sum().item()


        print( 'Train Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
        print( 'Train Agreement: %.3f%% (%d/%d)' % (100.*correct_target/total_target, correct_target, total_target))

    def test(self):
        self.attack_model.eval()

        correct = 0
        target_correct = 0
        total = 0

        agreement_correct = 0
        agreement_total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.attack_model(inputs)
                _, predicted = outputs.max(1)

                target_model_logit = self.target_model(inputs)
                _,target_predicted = target_model_logit.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                target_correct += target_predicted.eq(targets).sum().item()


                output_target = self.target_model(inputs)
                _, predicted_target = output_target.max(1)
                agreement_total += targets.size(0)
                agreement_correct += predicted.eq(predicted_target).sum().item()



            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
            print( 'Target Test Acc: %.3f%% (%d/%d)' % (100.*target_correct/total, target_correct, total))
            print( 'Test Agreement: %.3f%% (%d/%d)' % (100.*agreement_correct/agreement_total, agreement_correct, agreement_total))

        acc_test = correct/total
        agreemenet_test = agreement_correct / agreement_total

        return acc_test, agreemenet_test

    def saveModel(self):
        torch.save(self.attack_model.state_dict(), self.ATTACK_PATH)


