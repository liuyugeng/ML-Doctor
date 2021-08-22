import os
import sys
import time
import glob
import torch
import random
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)
import torchvision.transforms as transforms

from tqdm import tqdm
from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

class shadow_model_training():
    def __init__(self, trainloader, testloader, model, device, use_DP, num_classes, noise, norm, batch_size):
        self.use_DP = use_DP
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.net = module_modification.convert_batchnorm_modules(self.net)
            inspector = DPModelInspector()
            inspector.validate(self.net)
            privacy_engine = PrivacyEngine(
                self.net,
                batch_size=batch_size,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                secure_rng=False,
            )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.use_DP:
            epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B1: %.3f \u03B5: %.3f \u03B4: 1e-5" % (best_alpha, epsilon))
                
        self.scheduler.step()

        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total


    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total


class distillation_training():
    def __init__(self, PATH, trainloader, testloader, model, teacher, device):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        self.PATH = PATH
        self.teacher = teacher.to(self.device)
        self.teacher.load_state_dict(torch.load(self.PATH))
        self.teacher.eval()

        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    def distillation_loss(self, y, labels, teacher_scores, T, alpha):
        loss = self.criterion(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
        loss = loss * (T*T * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
        return loss

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, [targets, _]) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            teacher_output = self.teacher(inputs)
            teacher_output = teacher_output.detach()
    
            loss = self.distillation_loss(outputs, targets, teacher_output, T=2.0, alpha=0.95)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return 1.*correct/total

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, [targets, _] in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

        return 1.*correct/total

class attack_for_blackbox():
	def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device, r):
		self.device = device

		self.TARGET_PATH = TARGET_PATH
		self.SHADOW_PATH = SHADOW_PATH
		self.ATTACK_SETS = ATTACK_SETS

		self.target_model = target_model.to(self.device)
		self.shadow_model = shadow_model.to(self.device)

		self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
		self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))

		self.target_model.eval()
		self.shadow_model.eval()

		self.attack_train_loader = attack_train_loader
		self.attack_test_loader = attack_test_loader

		self.attack_model = attack_model.to(self.device)
		torch.manual_seed(r)
		self.attack_model.apply(weights_init)

		if self.device == 'cuda':
			self.net = torch.nn.DataParallel(self.net)
			cudnn.benchmark = True

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

	def get_data(self, model, inputs, targets):
		result = model(inputs)
		
		output, _ = torch.sort(result, descending=True)
		# results = F.softmax(results[:,:5], dim=1)
		_, predicts = result.max(1)
		
		prediction = []
		for predict in predicts:
			prediction.append([1,] if predict else [0,])

		prediction = torch.Tensor(prediction)

		# final_inputs = torch.cat((results, prediction), 1)
		# print(final_inputs.shape)

		return output, prediction

	def prepare_dataset(self):
		with open(self.ATTACK_SETS + "train.p", "wb") as f:
			for inputs, targets, members in self.attack_train_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				output, prediction = self.get_data(self.shadow_model, inputs, targets)
				# output = output.cpu().detach().numpy()
			
				pickle.dump((output, prediction, members), f)

		print("Finished Saving Train Dataset")

		with open(self.ATTACK_SETS + "test.p", "wb") as f:
			for inputs, targets, members in self.attack_test_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				output, prediction = self.get_data(self.target_model, inputs, targets)
				# output = output.cpu().detach().numpy()
			
				pickle.dump((output, prediction, members), f)

		print("Finished Saving Test Dataset")

	def train(self, epoch, result_path):
		self.attack_model.train()
		batch_idx = 1
		train_loss = 0
		correct = 0
		total = 0

		final_train_gndtrth = []
		final_train_predict = []
		final_train_probabe = []

		final_result = []

		with open(self.ATTACK_SETS + "train.p", "rb") as f:
			while(True):
				try:
					output, prediction, members = pickle.load(f)
					output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

					results = self.attack_model(output, prediction)

					# results = F.softmax(results, dim=1)
					losses = self.criterion(results, members)
					losses.backward()
					self.optimizer.step()

					train_loss += losses.item()
					_, predicted = results.max(1)
					total += members.size(0)
					correct += predicted.eq(members).sum().item()

					if epoch == 49:
						final_train_gndtrth.append(members)
						final_train_predict.append(predicted)
						final_train_probabe.append(results[:, 1])

					batch_idx += 1
				except EOFError:
					break

		if epoch == 49:
			final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
			final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
			final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

			train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
			train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

			final_result.append(train_f1_score)
			final_result.append(train_roc_auc_score)

			with open(result_path, "wb") as f:
				pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
			
			print("Saved Attack Train Ground Truth and Predict Sets")
			print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

		final_result.append(1.*correct/total)
		print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

		return final_result

	def test(self, epoch, result_path):
		self.attack_model.eval()
		batch_idx = 1
		correct = 0
		total = 0

		final_test_gndtrth = []
		final_test_predict = []
		final_test_probabe = []

		final_result = []

		with torch.no_grad():
			with open(self.ATTACK_SETS + "test.p", "rb") as f:
				while(True):
					try:
						output, prediction, members = pickle.load(f)
						output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

						results = self.attack_model(output, prediction)
						_, predicted = results.max(1)
						total += members.size(0)
						correct += predicted.eq(members).sum().item()

						results = F.softmax(results, dim=1)

						if epoch == 49:
							final_test_gndtrth.append(members)
							final_test_predict.append(predicted)
							final_test_probabe.append(results[:, 1])

						batch_idx += 1
					except EOFError:
						break

		if epoch == 49:
			final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
			final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
			final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

			test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
			test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

			final_result.append(test_f1_score)
			final_result.append(test_roc_auc_score)

			with open(result_path, "wb") as f:
				pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

			print("Saved Attack Test Ground Truth and Predict Sets")
			print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

		final_result.append(1.*correct/total)
		print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

		return final_result

	def delete_pickle(self):
		train_file = glob.glob(self.ATTACK_SETS +"train.p")
		for trf in train_file:
			os.remove(trf)

		test_file = glob.glob(self.ATTACK_SETS +"test.p")
		for tef in test_file:
			os.remove(tef)

	def saveModel(self, path):
		torch.save(self.attack_model.state_dict(), path)

class attack_for_whitebox():
	def __init__(self, TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device, class_num, r):
		self.device = device
		self.class_num = class_num

		self.ATTACK_SETS = ATTACK_SETS

		self.TARGET_PATH = TARGET_PATH
		self.target_model = target_model.to(self.device)
		self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
		self.target_model.eval()


		self.SHADOW_PATH = SHADOW_PATH
		self.shadow_model = shadow_model.to(self.device)
		self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
		self.shadow_model.eval()

		self.attack_train_loader = attack_train_loader
		self.attack_test_loader = attack_test_loader

		self.attack_model = attack_model.to(self.device)
		torch.manual_seed(r)
		self.attack_model.apply(weights_init)

		if self.device == 'cuda':
			self.net = torch.nn.DataParallel(self.net)
			cudnn.benchmark = True

		self.target_criterion = nn.CrossEntropyLoss(reduction='none')
		self.attack_criterion = nn.CrossEntropyLoss()
		#self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
		self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

		self.attack_train_data = None
		self.attack_test_data = None
		

	def get_data(self, model, inputs, targets):
		results = model(inputs)
		# outputs = F.softmax(outputs, dim=1)
		losses = self.target_criterion(results, targets)

		gradients = []
		
		for loss in losses:
			loss.backward(retain_graph=True)

			gradient_list = reversed(list(model.named_parameters()))

			for name, parameter in gradient_list:
				if 'weight' in name:
					gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
					gradient = gradient.unsqueeze_(0)
					gradients.append(gradient.unsqueeze_(0))
					break

		labels = []
		for num in targets:
			label = [0 for i in range(self.class_num)]
			label[num.item()] = 1
			labels.append(label)

		gradients = torch.cat(gradients, dim=0)
		losses = losses.unsqueeze_(1).detach()
		outputs, _ = torch.sort(results, descending=True)
		labels = torch.Tensor(labels)

		return outputs, losses, gradients, labels

	def prepare_dataset(self):
		with open(self.ATTACK_SETS + "train.p", "wb") as f:
			for inputs, targets, members in self.attack_train_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				output, loss, gradient, label = self.get_data(self.shadow_model, inputs, targets)

				pickle.dump((output, loss, gradient, label, members), f)

		print("Finished Saving Train Dataset")

		with open(self.ATTACK_SETS + "test.p", "wb") as f:
			for inputs, targets, members in self.attack_test_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				output, loss, gradient, label = self.get_data(self.target_model, inputs, targets)
			
				pickle.dump((output, loss, gradient, label, members), f)

			# pickle.dump((output, loss, gradient, label, members), open(self.ATTACK_PATH + "test.p", "wb"))

		print("Finished Saving Test Dataset")

	
	def train(self, epoch, result_path):
		self.attack_model.train()
		batch_idx = 1
		train_loss = 0
		correct = 0
		total = 0

		final_train_gndtrth = []
		final_train_predict = []
		final_train_probabe = []

		final_result = []

		with open(self.ATTACK_SETS + "train.p", "rb") as f:
			while(True):
				try:
					output, loss, gradient, label, members = pickle.load(f)
					output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

					results = self.attack_model(output, loss, gradient, label)
					# results = F.softmax(results, dim=1)
					losses = self.attack_criterion(results, members)
					
					losses.backward()
					self.optimizer.step()

					train_loss += losses.item()
					_, predicted = results.max(1)
					total += members.size(0)
					correct += predicted.eq(members).sum().item()

					if epoch == 49:
						final_train_gndtrth.append(members)
						final_train_predict.append(predicted)
						final_train_probabe.append(results[:, 1])

					batch_idx += 1
				except EOFError:
					break	

		if epoch == 49:
			final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
			final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
			final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

			train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
			train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

			final_result.append(train_f1_score)
			final_result.append(train_roc_auc_score)

			with open(result_path, "wb") as f:
				pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
			
			print("Saved Attack Train Ground Truth and Predict Sets")
			print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

		final_result.append(1.*correct/total)
		print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

		return final_result


	def test(self, epoch, result_path):
		self.attack_model.eval()
		batch_idx = 1
		correct = 0
		total = 0

		final_test_gndtrth = []
		final_test_predict = []
		final_test_probabe = []

		final_result = []

		with torch.no_grad():
			with open(self.ATTACK_SETS + "test.p", "rb") as f:
				while(True):
					try:
						output, loss, gradient, label, members = pickle.load(f)
						output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

						results = self.attack_model(output, loss, gradient, label)

						_, predicted = results.max(1)
						total += members.size(0)
						correct += predicted.eq(members).sum().item()

						results = F.softmax(results, dim=1)

						if epoch == 49:
							final_test_gndtrth.append(members)
							final_test_predict.append(predicted)
							final_test_probabe.append(results[:, 1])

						batch_idx += 1
					except EOFError:
						break

		if epoch == 49:
			final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
			final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
			final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

			test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
			test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

			final_result.append(test_f1_score)
			final_result.append(test_roc_auc_score)


			with open(result_path, "wb") as f:
				pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

			print("Saved Attack Test Ground Truth and Predict Sets")
			print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

		final_result.append(1.*correct/total)
		print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

		return final_result

	def delete_pickle(self):
		train_file = glob.glob(self.ATTACK_SETS +"train.p")
		for trf in train_file:
			os.remove(trf)

		test_file = glob.glob(self.ATTACK_SETS +"test.p")
		for tef in test_file:
			os.remove(tef)

	def saveModel(self, path):
		torch.save(self.attack_model.state_dict(), path)