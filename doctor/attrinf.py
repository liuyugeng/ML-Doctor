import torch
import pickle
import torch.nn as nn
import torch.optim as optim

from utils.define_models import *
from sklearn.metrics import f1_score

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
        self.dataset_type = None

    def _get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def init_attack_model(self, size, output_classes):
        x = torch.rand(size).to(self.device)
        input_classes = self.get_middle_output(x).flatten().shape[0]
        self.attack_model = attrinf_attack_model(inputs=input_classes, outputs=output_classes)
        self.attack_model.to(self.device)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)
        if output_classes == 2:
            self.dataset_type = "binary"
        else:
            self.dataset_type = "macro"

    def get_middle_output(self, x):
        temp = []
        for name, _ in self.target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if 1 > len(temp):
            raise IndexError('layer is out of range')

        name = temp[-2].split('.')
        var = eval('self.target_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation(name[1], out))
        _ = self.target_model(x)
        
        return out[name[1]]

    # Training
    def train(self, epoch):
        self.attack_model.train()
        
        train_loss = 0
        correct = 0
        total = 0

        final_result = []
        final_gndtrth = []
        final_predict = []
        final_probabe = []
        
        for batch_idx, (inputs, [_, targets]) in enumerate(self.attack_trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            oracles = self.get_middle_output(inputs)
            outputs = self.attack_model(oracles)
            outputs = F.softmax(outputs, dim=1)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if epoch:
                    final_gndtrth.append(targets)
                    final_predict.append(predicted)
                    final_probabe.append(outputs[:, 1])

        if epoch:
            final_gndtrth = torch.cat(final_gndtrth, dim=0).cpu().detach().numpy()
            final_predict = torch.cat(final_predict, dim=0).cpu().detach().numpy()
            final_probabe = torch.cat(final_probabe, dim=0).cpu().detach().numpy()

            test_f1_score = f1_score(final_gndtrth, final_predict, average=self.dataset_type)

            final_result.append(test_f1_score)

            with open(self.ATTACK_PATH + "_attrinf_train.p", "wb") as f:
                pickle.dump((final_gndtrth, final_predict, final_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f" % (test_f1_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def test(self, epoch):
        self.attack_model.eval()

        correct = 0
        total = 0
        final_result = []
        final_gndtrth = []
        final_predict = []
        final_probabe = []

        with torch.no_grad():
            for inputs, [_, targets] in self.attack_testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                oracles = self.get_middle_output(inputs)
                outputs = self.attack_model(oracles)
                outputs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if epoch:
                    final_gndtrth.append(targets)
                    final_predict.append(predicted)
                    final_probabe.append(outputs[:, 1])

        if epoch:
            final_gndtrth = torch.cat(final_gndtrth, dim=0).cpu().numpy()
            final_predict = torch.cat(final_predict, dim=0).cpu().numpy()
            final_probabe = torch.cat(final_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_gndtrth, final_predict, average=self.dataset_type)

            final_result.append(test_f1_score)

            with open(self.ATTACK_PATH + "_attrinf_test.p", "wb") as f:
                pickle.dump((final_gndtrth, final_predict, final_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f" % (test_f1_score))

        final_result.append(1.*correct/total)
        print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def saveModel(self):
        torch.save(self.attack_model.state_dict(), self.ATTACK_PATH + "_attrinf_attack_model.pth")

def train_attack_model(TARGET_PATH, ATTACK_PATH, output_classes, device, target_model, train_loader, test_loader, size):
    attack = attack_training(device, train_loader, test_loader, target_model, TARGET_PATH, ATTACK_PATH)
    attack.init_attack_model(size, output_classes)

    for epoch in range(100):
        flag = 1 if epoch==99 else 0
        print("<======================= Epoch " + str(epoch+1) + " =======================>")
        print("attack training")
        acc_train = attack.train(flag)
        print("attack testing")
        acc_test = attack.test(flag)

    attack.saveModel()
    print("Saved Attack Model")
    print("Finished!!!")


    return acc_train, acc_test