import torch
import itertools
import segmentation_models_pytorch as smp

from losses.DiceLoss import DiceLoss
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MutualModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_KD_1', type=float, default=0.5)
            parser.add_argument('--lambda_KD_2', type=float, default=1.0)
            parser.add_argument('--num_labels', type=int, default=8)

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_T', 'G_T', 'cycle_T', 'syn_sup', 'real_sup', 'kd_r_s', 'kd_s_r']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_T', 'rec_A']
        visual_names_T = ['real_T', 'fake_A', 'rec_T']


        self.visual_names = visual_names_A + visual_names_T
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_T', 'D_A', 'D_T', 'S_real', 'S_syn']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_T']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_T = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_T = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netS_real = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=opt.output_nc,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=opt.num_labels,                      # model output channels (number of classes in your dataset)
        )

        self.netS_real.to(self.gpu_ids[0])

        self.netS_syn = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=opt.output_nc,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=opt.num_labels,                      # model output channels (number of classes in your dataset)
        )

        self.netS_syn.to(self.gpu_ids[0])

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_T_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterion_dice = DiceLoss().to(self.device)
            self.softmax = torch.nn.Softmax(dim=1)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_T = torch.optim.Adam(self.netG_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_T = torch.optim.Adam(self.netD_T.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_S = torch.optim.Adam(itertools.chain(self.netS_real.parameters(), self.netS_syn.parameters()), lr=opt.lr, betas=(0.9, 0.999))

            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_T)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_T)
            self.optimizers.append(self.optimizer_S)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input["A_scan"].to(self.device)
        self.real_T = input["T_scan"].to(self.device)
        self.label_A = input["A_labels"].to(self.device)
        self.label_T = input["T_labels"].to(self.device)

        self.label_A_raw = input["A_labels_raw"]
        self.label_T_raw = input["T_labels_raw"]
        self.image_paths = input['A_paths']

    def forward(self):
        torch.autograd.set_detect_anomaly(True)#TODO remove
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_T = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_T(self.fake_T)   # G_B(G_A(A))
        self.fake_A = self.netG_T(self.real_T)  # G_B(B)
        self.rec_T = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.p_T_real = self.netS_real(self.real_T)
        self.p_A_T_real = self.netS_real(self.fake_T)

        self.p_T_syn = self.netS_syn(self.real_T)
        self.p_A_T_syn = self.netS_syn(self.fake_T)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        torch.set_printoptions(profile="full")
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # self.optimizer_D_A.zero_grad()
        fake_T = self.fake_T_pool.query(self.fake_T)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_T, fake_T)
        # self.optimizer_D_A.step()

    def backward_D_T(self):
        """Calculate GAN loss for discriminator D_B"""
        # self.optimizer_D_T.zero_grad()
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_T = self.backward_D_basic(self.netD_T, self.real_A, fake_A)
        # self.optimizer_D_T.step()

    def backward_G_A_T(self):
        lambda_A = self.opt.lambda_A
        # self.optimizer_G_A.zero_grad()
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_T), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_syn_sup = self.criterionCE(self.p_A_T_syn, self.label_A) + self.criterion_dice(self.p_A_T_syn, self.label_A)

        self.loss_G_A_T = self.loss_G_A + self.loss_cycle_A + self.loss_syn_sup
        self.loss_G_A_T.backward(retain_graph=True)
        # self.optimizer_G_A.step()
        return self.loss_G_A_T

    def backward_G_T_A(self):
        lambda_T = self.opt.lambda_T

        # self.optimizer_G_T.zero_grad()
        self.loss_G_T = self.criterionGAN(self.netD_T(self.fake_A), True)
        self.loss_cycle_T = self.criterionCycle(self.rec_T, self.real_T) * lambda_T

        self.loss_G_T_A = self.loss_G_T + self.loss_cycle_T
        self.loss_G_T_A.backward()
        # self.optimizer_G_A.step()
        # self.optimizer_G_T.step()
        return self.loss_G_T_A

    def backward_S(self):
        lambda_KD_1 = self.opt.lambda_KD_1
        lambda_KD_2 = self.opt.lambda_KD_2
        # self.optimizer_S.zero_grad()


        self.loss_real_sup = self.criterionCE(self.p_T_real, self.label_T) + self.criterion_dice(self.p_T_real, self.label_T)
        self.loss_kd_s_r = self.criterionCE(self.p_A_T_real, self.softmax(self.p_A_T_syn)) * lambda_KD_1
        self.loss_real_seg = self.loss_real_sup + self.loss_kd_s_r

        # self.loss_syn_sup = self.criterionCE(self.p_A_T_syn, self.label_A) + self.criterion_dice(self.p_A_T_syn, self.label_A)
        self.loss_kd_r_s = self.criterionCE(self.p_T_syn, self.softmax(self.p_T_real)) * lambda_KD_2
        self.loss_syn_seg = self.loss_syn_sup + self.loss_kd_r_s

        self.loss_real_seg.backward(retain_graph=True)
        self.loss_syn_seg.backward()
        # self.optimizer_S.step()
        return self.loss_syn_seg, self.loss_real_seg

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        for optimizer in self.optimizers:
            optimizer.zero_grad() #
        self.set_requires_grad([self.netD_A, self.netD_T], False)
        # self.set_requires_grad([self.netG_A], True)
        self.backward_G_A_T()
        # self.set_requires_grad([self.netG_A], False)
        self.set_requires_grad([self.netD_T], True)
        self.backward_D_T()
        self.set_requires_grad([self.netD_T], False)
        # self.set_requires_grad([self.netG_T], True)
        self.backward_G_T_A()
        # self.set_requires_grad([self.netG_T], False)
        self.set_requires_grad([self.netD_A], True)
        self.backward_D_A()
        self.set_requires_grad([self.netD_A], False)
        # self.set_requires_grad([self.netS_real, self.netS_syn], True)
        self.backward_S()
        for optimizer in self.optimizers:
            optimizer.step()
