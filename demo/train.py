import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine
from torch.optim import lr_scheduler
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector



def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.normal_(m.weight.data)
		m.bias.data.fill_(0)
	elif isinstance(m,nn.Linear):
		nn.init.xavier_normal_(m.weight)
		nn.init.constant_(m.bias, 0)


class model_training():
	def __init__(self, trainloader, testloader, model, device, use_DP, num_classes, noise, norm):
		self.use_DP = use_DP
		self.device = device
		self.net = model.to(self.device)
		self.trainloader = trainloader
		self.testloader = testloader

		self.num_classes = num_classes

		if self.device == 'cuda':
			self.net = torch.nn.DataParallel(self.net)
			cudnn.benchmark = True

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

		self.noise_multiplier, self.max_grad_norm = noise, norm
		
		if self.use_DP:
			self.net = module_modification.convert_batchnorm_modules(self.net)
			inspector = DPModelInspector()
			inspector.validate(self.net)
			privacy_engine = PrivacyEngine(
				self.net,
				batch_size=64,
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
		
		for batch_idx, (inputs, targets) in enumerate(self.trainloader):
			if str(self.criterion) != "CrossEntropyLoss()":
				targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

			inputs, targets = inputs.to(self.device), targets.to(self.device)
			self.optimizer.zero_grad()
			outputs = self.net(inputs)

			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			if str(self.criterion) != "CrossEntropyLoss()":
				_, targets= targets.max(1)

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
			for inputs, targets in self.testloader:
				if str(self.criterion) != "CrossEntropyLoss()":
					targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.net(inputs)

				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				if str(self.criterion) != "CrossEntropyLoss()":
					_, targets= targets.max(1)

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

		if self.device == 'cuda':
			self.model = torch.nn.DataParallel(self.model)
			cudnn.benchmark = True

		self.criterion = nn.KLDivLoss(reduction='batchmean')
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

		self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

	def distillation_loss(self, y, labels, teacher_scores, T, alpha):
		loss = self.criterion(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1))
		loss = loss * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
		return loss

	def train(self):
		self.model.train()
		self.teacher.eval()
		train_loss = 0
		correct = 0
		total = 0

		for batch_idx, (inputs, targets) in enumerate(self.trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			teacher_output = self.teacher(inputs)
			teacher_output = teacher_output.detach()
	
			loss = self.distillation_loss(outputs, targets, teacher_output, T=20.0, alpha=0.7)
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
			for inputs, targets in self.testloader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.model(inputs)

				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

			print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

		return 1.*correct/total

