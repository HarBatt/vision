import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.cuda import amp
from utils import LambdaLR
from modules import Generator, Discriminator
from data_loading import ReplayBuffer
from utils import sample_images
from shared.component_logger import component_logger as logger


class CycleGAN:
    """
       Cycle-Consistent Adversarial Networks:
       -----------------------------------------------------------
       Loosely based on the open source implementation of the CycleGAN

       Reference:
       Zhu, Jun-Yan, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired image-to-image translation using cycle-consistent adversarial networks." 
       In Proceedings of the IEEE international conference on computer vision, pp. 2223-2232. 2017.
    """

    def __init__(self, params):
        self.params = params
        input_shape = (params.channels, params.img_height, params.img_width)
        self.G_AB = Generator(input_shape, params.n_residual_blocks)
        self.G_BA = Generator(input_shape, params.n_residual_blocks)
        self.D_A = Discriminator(input_shape)
        self.D_B = Discriminator(input_shape)


        # Losses
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()


        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr= params.learning_rate, betas = (params.beta1, params.beta2))
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr= params.learning_rate, betas = (params.beta1, params.beta2))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr= params.learning_rate, betas = (params.beta1, params.beta2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(params.n_epochs, params.epoch, params.decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(params.n_epochs, params.epoch, params.decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(params.n_epochs, params.epoch, params.decay_epoch).step
        )

        logger.log("Traininable Parameters in Generator (G_AB): {:,}".format(self.parameters_count(self.G_AB)))
        logger.log("Traininable Parameters in Generator (G_BA): {:,}".format(self.parameters_count(self.G_BA)))
        logger.log("Traininable Parameters in Discriminator (D_A): {:,}".format(self.parameters_count(self.D_A)))
        logger.log("Traininable Parameters in Discriminator (D_B): {:,}".format(self.parameters_count(self.D_A)))




    def parameters_count(self, model):
        """
        params:
            model: A neural network architecture to count trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def save_model(self, model, step, path):
        """
        params:
            model: model to save
            epoch: epoch of the training process
            path: path to save the model
        """

        if model == "d_b":
            torch.save({
                'step': step, 
                'model_state_dict': self.D_B.state_dict(),
                'optimizer_state_dict': self.optimizer_D_B.state_dict()
                }, path)
        
        elif model == "d_a":
            torch.save({
                'step': step, 
                'model_state_dict': self.D_A.state_dict(),
                'optimizer_state_dict': self.optimizer_D_A.state_dict()
                }, path)
        
        elif model == "g_ab":
            torch.save({
                'step': step, 
                'model_state_dict': self.G_AB.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict()
                }, path)

        elif model == "g_ba":
            torch.save({
                'step': step, 
                'model_state_dict': self.G_BA.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict()
                }, path)
        
        else:
            raise ValueError("Invalid model or path name")
    

    def load_model(self, model, path):
        """
        Load model from the path.

        params:
            model: model name
            path: path to the model
        """
        if model =="d_b":
            checkpoint = torch.load(path)
            self.D_B.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_D_B.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="d_a":
            checkpoint = torch.load(path)
            self.D_A.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_D_A.load_state_dict(checkpoint['optimizer_state_dict'])
        
        elif model=="g_ab":
            checkpoint = torch.load(path)
            self.G_AB.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

        elif model=="g_ba":
            checkpoint = torch.load(path)
            self.G_BA.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        
        else:
            raise ValueError("Invalid model or path name")
    

    def translate_image(self, image, domain_from, domain_to):
        """
        Domain tranlation from A to B or B to A.
        -----------------------------------------
        params:
            image: image to translate
            domain_from: domain from which the image is translated
            domain_to: domain to which the image is translated

        """
        if domain_from == "a" and domain_to == "b":
            return self.G_AB(image)
        elif domain_from == "b" and domain_to == "a":
            return self.G_BA(image)
        else:
            raise ValueError("Invalid domain")

    
    def train(self, train_dataloader, val_dataloader):
        """
        Train the model, with train_dataloader and sample images for translation using val_dataloader
        """
        params = self.params 
        # Initialize Generator and Discriminator to device
        self.G_AB.to(params.device)
        self.G_BA.to(params.device)
        self.D_A.to(params.device)
        self.D_B.to(params.device)

        params.learning_rate = 0.0001

        scaler = amp.GradScaler()

        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()


        for epoch in range(params.epoch, params.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # Set model input
                real_A = batch["A"].to(params.device)
                real_B = batch["B"].to(params.device)
                # Adversarial ground truths
                valid = torch.Tensor(torch.ones((real_A.size(0), *self.D_A.output_shape))).to(params.device)
                fake = torch.Tensor(torch.zeros((real_A.size(0), *self.D_A.output_shape))).to(params.device)

                #Train Generators#
                
                self.G_AB.train()
                self.G_BA.train()

                self.optimizer_G.zero_grad()
                
                with amp.autocast():
                    # Identity loss
                    loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
                    loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

                    loss_identity = (loss_id_A + loss_id_B) / 2

                    # GAN loss
                    fake_B = self.G_AB(real_A)
                    loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
                    fake_A = self.G_BA(real_B)
                    loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

                    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                    # Cycle loss
                    recov_A = self.G_BA(fake_B)
                    loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                    recov_B = self.G_AB(fake_A)
                    loss_cycle_B = self.criterion_cycle(recov_B, real_B)

                    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                    # Total loss
                    loss_G = loss_GAN + params.lambda_cycle * loss_cycle + params.lambda_identity * loss_identity

                scaler.scale(loss_G).backward()
                scaler.step(self.optimizer_G)
                scaler.update()


                """
                Train Discriminator A
                """

                self.optimizer_D_A.zero_grad()
                
                with amp.autocast():
                    # Real loss
                    loss_real = self.criterion_GAN(self.D_A(real_A), valid)
                    # Fake loss (on batch of previously generated samples)
                    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                    loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
                    # Total loss
                    loss_D_A = (loss_real + loss_fake) / 2
                
                scaler.scale(loss_D_A).backward()
                scaler.step(self.optimizer_D_A)
                scaler.update()

                """
                Train Discriminator B
                """

                self.optimizer_D_B.zero_grad()
                
                with amp.autocast():
                    # Real loss
                    loss_real = self.criterion_GAN(self.D_B(real_B), valid)
                    # Fake loss (on batch of previously generated samples)
                    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                    loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
                    # Total loss
                    loss_D_B = (loss_real + loss_fake) / 2
                
                scaler.scale(loss_D_B).backward()
                scaler.step(self.optimizer_D_B)
                scaler.update()

                loss_D = (loss_D_A + loss_D_B) / 2
                    
                batches_done = epoch * len(train_dataloader) + step
                
                if step % params.print_every == 0:
                    self.save_model("g_ab", step, "checkpoints/{}/G_AB_{}.pth".format(params.dataset_name, epoch))
                    self.save_model("g_ba", step, "checkpoints/{}/G_BA_{}.pth".format(params.dataset_name, epoch))
                    self.save_model("d_a", step, "checkpoints/{}/D_A_{}.pth".format(params.dataset_name, epoch))
                    self.save_model("d_b", step, "checkpoints/{}/D_B_{}.pth".format(params.dataset_name, epoch))
                    sample_images(self.G_AB, self.G_BA, val_dataloader, batches_done, params)

                    logger.log(
                        "[Epoch {}/{}] [Batch {}/{}] [L(D): {}] [L(G): {}, L(Adv): {}, L(Cyc): {}, L(I): {}]".format(
                        epoch, params.n_epochs, step, 
                        len(train_dataloader), np.round(loss_D.item(), 4), 
                        np.round(loss_G.item(), 4), np.round(loss_GAN.item(), 4), 
                        np.round(loss_cycle.item(), 4), np.round(loss_identity.item(), 4))
                    )
                
            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()