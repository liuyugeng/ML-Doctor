import torch
import torch.nn as nn
import torch.nn.functional as F

class attrinf_attack_model(nn.Module):
    def __init__(self, inputs, outputs):
        super(attrinf_attack_model, self).__init__()
        self.classifier = nn.Linear(inputs, outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ShadowAttackModel(nn.Module):
	def __init__(self, class_num):
		super(ShadowAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result


class PartialAttackModel(nn.Module):
	def __init__(self, class_num):
		super(PartialAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result


class WhiteBoxAttackModel(nn.Module):
	def __init__(self, class_num, total):
		super(WhiteBoxAttackModel, self).__init__()

		self.Output_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Loss_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Gradient_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Conv2d(1, 1, kernel_size=5, padding=2),
			nn.BatchNorm2d(1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten(),
			nn.Dropout(p=0.2),
			nn.Linear(total, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Label_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, loss, gradient, label):
		Output_Component_result = self.Output_Component(output)
		Loss_Component_result = self.Loss_Component(loss)
		Gradient_Component_result = self.Gradient_Component(gradient)
		Label_Component_result = self.Label_Component(label)

		# Loss_Component_result = F.softmax(Loss_Component_result, dim=1)
		# Gradient_Component_result = F.softmax(Gradient_Component_result, dim=1)

		# final_inputs = Output_Component_result
		# final_inputs = Loss_Component_result
		# final_inputs = Gradient_Component_result
		# final_inputs = Label_Component_result
		
		final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result