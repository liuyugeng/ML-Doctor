import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine
from torch.optim import lr_scheduler

def GAN_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class model_training():
    def __init__(self, trainloader, testloader, model, device, use_DP, noise, norm, delta):
        self.use_DP = use_DP
        self.device = device
        self.delta = delta
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.noise_multiplier, self.max_grad_norm = noise, norm
        
        if self.use_DP:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
                module=model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            # self.net = module_modification.convert_batchnorm_modules(self.net)
            # inspector = DPModelInspector()
            # inspector.validate(self.net)
            # privacy_engine = PrivacyEngine(
            #     self.net,
            #     batch_size=64,
            #     sample_size=len(self.trainloader.dataset),
            #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            #     noise_multiplier=self.noise_multiplier,
            #     max_grad_norm=self.max_grad_norm,
            #     secure_rng=False,
            # )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            # privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

    # Training
    def train(self):
        self.net.train()
        
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

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
            epsilon, best_alpha = self.privacy_engine.accountant.get_privacy_spent(delta=self.delta)
            # epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
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
                if isinstance(targets, list):
                    targets = targets[0]
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

class GAN_training():
    def __init__(self, trainloader, model_discriminator, model_generator, device):
        self.device = device
        self.trainloader = trainloader

        self.model_discriminator = model_discriminator.to(self.device)
        self.model_generator = model_generator.to(self.device)

        self.model_discriminator.apply(GAN_init)
        self.model_generator.apply(GAN_init)

        
        self.criterion = nn.BCELoss()
        self.optimizer_discriminator = optim.Adam(model_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_generator = optim.Adam(model_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.real_label = 1.
        self.fake_label = 0.

    def train(self):
        # For each batch in the dataloader
        for i, data in enumerate(self.trainloader, 0):
            self.model_discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.model_discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.model_generator(noise)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.model_discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizer_discriminator.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.model_generator.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.model_discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizer_generator.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (i, len(self.trainloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            # G_losses.append(errG.item())
            # D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == 4) and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


    def saveModel(self, path_d, path_g):
        torch.save(self.model_discriminator.state_dict(), path_d)
        torch.save(self.model_generator.state_dict(), path_g)

