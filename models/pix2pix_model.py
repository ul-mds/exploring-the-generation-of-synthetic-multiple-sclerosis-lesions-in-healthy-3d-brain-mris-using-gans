import torch
from .base_model import BaseModel
from . import networks
from util.util import zero_division


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_L2', type=float, default=100.0, help='weight for L2 truth loss in background area.')
        parser.add_argument('--k_fold', type=int, default=-1, help='fold which is not used for training, but saved for testing.')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt, threed=False):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.mask = None
        self.truth = None
        self.real_A = None
        self.real_B = None
        self.fp16 = opt.fp16
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_L2', 'D_real', 'D_fake']
        # specify the images you want to save/display.
        
        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            # The training scripts will call <BaseModel.get_current_visuals>
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'truth']
        else:  # during test time, only load G
            self.visual_names = ['real_A', 'fake_B', 'truth']
            self.model_names = ['G']

        self.nifti = True if opt.dataset_mode == "nifti" else False
        # define networks (both generator and discriminator)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.upsampling,
                                      threed)

        if self.isTrain:  # define a discriminator;
            # conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, threed)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            if self.opt.dataset_mode == "gan2": 
                # in gan2 we use L1 for the whole image, L2 for lesion areas only
                self.criterionL1 = torch.nn.L1Loss(reduction='mean')
                self.criterionL2 = torch.nn.MSELoss(reduction='sum')
            elif self.opt.dataset_mode == "gan1":
                # in gan1 we use L1 for lesion areas only, L2 for the whole image
                #TODO Set back? Experimenting with L1 reduction = 'mean' instead of sum -> no manual division.
                self.criterionL1 = torch.nn.L1Loss(reduction='sum')
                self.criterionL2 = torch.nn.MSELoss(reduction='mean')
            else:
                # standard: take mean.
                self.criterionL1 = torch.nn.L1Loss(reduction='mean')
                self.criterionL2 = torch.nn.MSELoss(reduction='mean')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if len(opt.gpu_ids) > 0 and self.fp16:
                from apex import amp
                [self.netD, self.netG], [self.optimizer_D, self.optimizer_G] = amp.initialize(
                    [self.netD, self.netG],
                    [self.optimizer_D, self.optimizer_G],
                    opt_level='O1',
                    num_losses=2
                )
            if len(opt.gpu_ids) > 0:
                self.netD = torch.nn.DataParallel(self.netD, opt.gpu_ids)  # multi-GPUs
        if len(opt.gpu_ids) > 0:
            self.netG = torch.nn.DataParallel(self.netG, opt.gpu_ids)  # multi-GPUs

    def set_input(self, input, L1_mask_only=False):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        if 'B' in input:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        else: 
            self.real_A = input['A'].to(self.device)
            self.real_B = None

        if self.nifti or L1_mask_only:
            self.mask = input['mask']
        else:
            # If we calculate L1 for all pixels, we don't need a mask
            self.mask = torch.ones(self.real_A.shape, dtype=torch.uint8)
        self.mask = self.mask.to(self.device)

        if self.nifti and 'truth' in input and input['truth'] is not None:
            self.truth = input['truth']
        else:
            # If we don't have the truth, nothing should matter
            self.truth = torch.zeros(self.real_A.shape, dtype=torch.bool)
        self.truth = self.truth.to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs;
        # we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if self.fp16:
            from apex import amp
            with amp.scale_loss(self.loss_D, self.optimizer_D, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # 1. GAN loss: G(A) should fool the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        #self.loss_G_L1 = zero_division(self.loss_G_L1, torch.sum(self.mask))
        
         # 3. L2 loss in background or lesion regions, depending on gan type
        if self.opt.dataset_mode == "gan2":
            # In gan2, L1 is computed for the whole image, L2 only for lesion areas
            # we have 3 channels, loss is only calculated on the first channel, the others are ignored.
            # L1 reduction method is set to 'mean' if dataset_mode == "gan2"
            self.loss_G_L1 = self.criterionL1(self.fake_B[:, 0:1, :,:],
                                          self.real_B[:, 0:1, :, :]) * self.opt.lambda_L1
            
            lsmsk_tensor = self.real_B[:, 1:2, :, :]
            self.loss_G_L2 = self.criterionL2(self.fake_B[:, 0:1, :,:] * lsmsk_tensor,
                                              self.real_B[:, 0:1, :, :] * lsmsk_tensor) * self.opt.lambda_L2
            
            # L2 reduction = 'sum' -> we divide by the sum of voxels manually
            self.loss_G_L2 = zero_division(self.loss_G_L2, torch.sum(lsmsk_tensor))
        else: 
            # in gan1 L1 is computed for the lesion areas only, L2 for the background 
            # we have one channel instead of three.

            # L1 reduction = 'mean' -> we divide by the sum of lesion voxels manually
            self.loss_G_L1 = self.criterionL1(self.fake_B * self.mask,
                                          self.real_B * self.mask) * self.opt.lambda_L1
            #TODO Set back? Experimenting with L1 reduction 'mean' instead of 'sum' 
            self.loss_G_L1 = zero_division(self.loss_G_L1, torch.sum(self.mask))

            # calculate L2 loss for whole img ( background + ls_regions)
            # L2 reduction = 'mean' (no further division)
            self.loss_G_L2 = self.criterionL2(self.fake_B * (1-self.mask), 
                                              self.real_B * (1-self.mask)) * self.opt.lambda_L2

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  + self.loss_G_L2
        if self.fp16:
            from apex import amp
            with amp.scale_loss(self.loss_G, self.optimizer_G, loss_id=1) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()

    def optimize_parameters(self, i):
        self.forward()  # compute fake images: G(A)
        # update D
        #if i % 5 == 0:  TODO set as hyperparameter to slow down training of D 
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
