import utils, torch, time, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms
#from torch.distributions.distribution import Distribution

from utils import log_likelihood_samplesImean_sigma, prior_z, log_mean_exp

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

PI2 = torch.log(torch.FloatTensor(np.asarray([np.pi]))*2)

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, z_dim, batch_size, arch_type, gpu_mode):
        super(Discriminator, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.z_dim = z_dim
        self.output_dim=1
        self.arch_type=arch_type
        self.gpu_mode = gpu_mode
        self.batch_size = batch_size

        self.zlayer = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim, 1),
        )

        self.zflayer = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim*4, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.LeakyReLU(0.2),

        )

        self.xlayer = nn.Sequential(
            nn.Linear(self.input_height*self.input_width, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.ReLU(),
            nn.Linear(self.z_dim*4, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.ReLU(),
            nn.Linear(self.z_dim*4, 1),
        )

        self.xflayer = nn.Sequential(
            nn.Linear(self.input_height*self.input_width, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim*4, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim*4, self.z_dim*4),
            nn.BatchNorm1d(self.z_dim*4),
            nn.LeakyReLU(0.2),

        ) 
        utils.initialize_weights(self)

        if self.gpu_mode:
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()


    def forward(self, X, Z):

        H = self.logQZX_PZX(X, Z)
        return F.sigmoid(H)


    def logQZX_PZX(self, X, Z):

        X = X.view([-1, self.input_height * self.input_width * self.input_dim])
        Tx = self.xlayer(X)
        Tz = self.zlayer(Z)

        Hx = self.xflayer(X)
        Hz = self.zflayer(Z)
        H = torch.sum(Hx.mul_(Hz), dim=1) + torch.squeeze(Tx.add_(Tz))
    
        return H - 0.5 * torch.sum(Z**2, dim=1)


    def loss(self, x_, z_x_, z_, ftype='gan'):

        Y0 = self.forward(x_, z_x_)
        Y1 = self.forward(x_, z_)
        #J  = -torch.mean(torch.log(Y0) + torch.log(1 - Y1))
        y_real_ = Variable(torch.ones(self.batch_size, 1).cuda())
        y_fake_ = Variable(torch.zeros(self.batch_size, 1).cuda())
        D_real_loss = self.BCE_loss(Y0, y_real_)
        D_fake_loss = self.BCE_loss(Y1, y_fake_)
        return D_real_loss+D_fake_loss



class Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim, arch_type, acF=False, gpu_mode=True, dim_sam=10):
        super(Encoder, self).__init__()

        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.acF = acF
        self.z_dim = z_dim
        self.dim_sam = dim_sam
        self.gpu_mode = gpu_mode
        self.arch_type = arch_type

        if self.arch_type == 'conv':
            self.enc_layer1 = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), self.z_dim),
            )
        
        else:

            self.enc_layer1 = nn.Sequential(
                nn.Linear(self.input_height*self.input_width\
                                *self.input_dim, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2))
            self.enc_nlayer1 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim*4),
            )

            self.enc_layer2 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2))
            self.enc_nlayer2 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim*4),
            )

            self.enc_layer3 = nn.Sequential( 
                nn.Linear(self.z_dim*4, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2)
            )
            self.enc_nlayer3 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim*4),
            )

            self.fc = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim),
            )

        if self.acF: 
            self.mu = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim),
            )
    
            self.sigma = nn.Sequential(
                nn.Linear(self.z_dim*4, self.z_dim),
            )
    
        utils.initialize_weights(self)


    def forward(self, x):

        if self.arch_type == 'conv':
            x = self.enc_layer1(x)
            x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        else:
            x = x.view([-1, self.input_height * self.input_width * self.input_dim])

            x = self.enc_layer1(x)
            if self.gpu_mode :
                eps = torch.randn(x.size()).cuda() 
            else:
                eps = torch.randn(x.size())
            eps = Variable(eps, requires_grad=False) 
            x = x + self.enc_nlayer1(eps)

            if self.gpu_mode :
                eps = torch.randn(x.size()).cuda() 
            else:
                eps = torch.randn(x.size())
            eps = Variable(eps, requires_grad=False) 
            x = x + self.enc_nlayer2(eps)

            if self.gpu_mode :
                eps = torch.randn(x.size()).cuda() 
            else:
                eps = torch.randn(x.size())
            eps = Variable(eps, requires_grad=False) 
            x = self.enc_layer3(x + self.enc_nlayer3(eps))
        h = self.fc(x)

        return h


    def sample(self, mu, logvar):

        std = torch.exp(logvar)
        if self.gpu_mode: 
            eps = torch.randn(x.size()).cuda() * 0.05
        else:
            eps = torch.randn(x.size()) * 0.05

        eps = Variable(eps)

        return eps.mul(std).add_(mu)



class Decoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim, arch_type):
        super(Decoder, self).__init__()

        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        self.input_height = 28
        self.input_width = 28
        self.output_dim = 1
        self.z_dim = z_dim
        self.arch_type = arch_type

        if self.arch_type == 'conv':
            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
                nn.LeakyReLU(0.2),
            )
            self.dec_layer2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:

            self.dec_layer1 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.LeakyReLU(0.2),
                nn.Linear(self.z_dim*4, self.z_dim*4),
                nn.BatchNorm1d(self.z_dim*4),
                nn.Tanh(),
            )

            self.dec_layer2 = nn.Sequential(
                nn.Linear(self.z_dim*4, self.input_height * self.input_width),
                nn.Sigmoid(),
            )


        utils.initialize_weights(self)


    def forward(self, z):

        N,D = z.size()
        x = self.dec_layer1(z.view([-1,D]))
        if self.arch_type == 'conv':
            x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
            x = self.dec_layer2(x)
        else:
            x = self.dec_layer2(x)
            x = x.view(-1, 1, self.input_height, self.input_width)

        return x



class AVB():

    def __init__(self, args):
        super(AVB, self).__init__()

        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.model_type
        self.z_dim = args.z_dim
        self.dim_sam=args.dim_sam
        self.arch_type=args.arch_type
        self.acF=args.acF

        # Networks init
        self.disc_net = Discriminator(self.z_dim, self.batch_size, self.arch_type, self.gpu_mode)
        self.encoder  = Encoder(self.z_dim, self.arch_type, self.acF, self.gpu_mode, dim_sam=self.dim_sam)
        self.decoder  = Decoder(self.z_dim, self.arch_type)

        if self.gpu_mode:
            self.disc_net = self.disc_net.cuda()
            self.encoder  = self.encoder.cuda()
            self.decoder  = self.decoder.cuda()

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.randn((self.batch_size, self.z_dim)), volatile=True)

    
    def forward(self, x):

        if self.model_name == 'DAVB':
            if self.gpu_mode: 
                eps = torch.randn(x.size()).cuda() * 0.05
            else:
                eps = torch.randn(x.size()) * 0.05
            x = x.add_(Variable(eps))
        
        z   = self.encoder(x)
        res = self.decoder(z)
        return res, z


    def loss(self, x, beta):

        N, C, iw, ih = x.size()
        if self.acF:
            z, mu, logvar = encoder(x_real)
            mu, logvar = tf.stop_gradient(mean), tf.stop_gradient(logvar)
            logstd = tf.sqrt(logvar + 1e-4)
        else:
            z   = self.encoder(x)
        recon_x = self.decoder(z)


        bce = x * torch.log(recon_x) + (1. - x) * torch.log(1 - recon_x)
        bce = torch.sum(torch.sum(torch.sum(bce, dim=3), dim=2), dim=1)
        
        Td = self.disc_net.logQZX_PZX(x, z)
        #logz = prior_z(z, dim=-1)
        #logr = - 0.5 * torch.sum(z**2 + PI2[0], dim=-1) 
        #logr = - 0.5 * torch.sum(z**2 + logstd*2 + PI2[0], dim=-1) 
        KL = torch.squeeze(Td) #+ logr - logz
        J  = torch.mean(bce - beta * KL, dim=0)
        return -J / (iw * ih * C)


    def lle(self, x):

        N, C, iw, ih = x.size()
        if self.acF:
            z, mu, logvar = encoder(x_real)
            mu, logvar = tf.stop_gradient(mean), tf.stop_gradient(logvar)
            logstd = tf.sqrt(logvar + 1e-4)
        else:
            z   = self.encoder(x)

        recon_x = self.decoder(z)

        bce = x * torch.log(recon_x) + (1. - x) * torch.log(1 - recon_x)
        bce = torch.sum(torch.sum(torch.sum(bce, dim=3), dim=2), dim=1)

        Td = self.disc_net.logQZX_PZX(x, z)
        #Td = log_mean_exp(Td.view([self.dim_sam, -1]).permute(1,0), dim=1)

        #log_q_z_x = log_likelihood_samplesImean_sigma(Z, mu, logvar, dim=2)
        #log_p_z   = prior_z(Z, dim=2)
        #log_p_z_q_zx = torch.squeeze(Td) #+ logr - logz
        log_ws = bce - Td 
        log_ws = log_ws.view([self.dim_sam, -1]).permute(1,0)
        J = torch.mean(torch.squeeze(log_mean_exp(log_ws, dim=1)), dim=0)

        return -J/ (iw * iw * C)


