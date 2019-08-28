import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from model.netUtil import *
#Conv2d, Conv2d2x2, lReLU, BN, Conv1x1, Unet, Unet31, Unet_shallow, SRnet, tmpnet, SRnet31, SRnet2, UnetL3, StarG, UnetIN, UnetINShallow, UnetINMultiDiv8, UnetINDiv8

dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

REAL_LABEL = 1#0.9
eps        = 1e-12
class Colla8:
    def __init__(self, opt):
        self.nB    = opt.nB
        self.nCh_in = opt.nCh_in
        self.nCh_out = opt.nCh_out
        self.nY    = opt.nY
        self.nX    = opt.nX
        self.lr    = opt.lr
        self.lr_D  = opt.lr_D
        self.lr_C  = opt.lr_C
        self.nCh   = opt.ngf
        self.nCh_D = opt.nCh_D
        self.nCh_C = opt.nCh_C
        self.use_lsgan = opt.use_lsgan
        self.class_N = 8
        self.lambda_l1_cyc = opt.lambda_l1_cyc
        self.lambda_l2_cyc = opt.lambda_l2_cyc
        self.lambda_l1 = opt.lambda_l1
        self.lambda_l2 = opt.lambda_l2
        self.lambda_GAN = opt.lambda_GAN
        self.lambda_G_clsf = opt.lambda_G_clsf
        self.lambda_D_clsf = opt.lambda_D_clsf
        self.lambda_ssim = opt.lambda_ssim
        self.lambda_ssim_cyc = opt.lambda_ssim_cyc
        self.scale = 45.0 #  tmporally for visualization

        self.G = Generator('G', opt.G, self.nCh_out,nCh=opt.ngf,use_1x1Conv=opt.use_1x1Conv, w_decay=opt.w_decay,resid=opt.use_resid)
        self.D = Discriminator('D', nCh=self.nCh_D, w_decay_D=opt.w_decay_D,class_N=self.class_N, DR_ratio=opt.dropout)

        # placeholders 
        self.targets = tf.placeholder(dtype, [self.nB, self.nCh_out, self.nY, self.nX])
        self.tar_class_idx = tf.placeholder(tf.uint8)
        self.is_Training= tf.placeholder(tf.bool)
        
        self.a_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.b_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.c_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.d_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.e_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])     
        self.f_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.g_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.n_img = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])     

        self.a_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.b_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.c_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.d_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.e_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.f_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.g_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
        self.n_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])

        self.bool1 = tf.placeholder(tf.bool)
        self.bool2 = tf.placeholder(tf.bool)
        self.bool3 = tf.placeholder(tf.bool)
        self.bool4 = tf.placeholder(tf.bool)
        self.bool5 = tf.placeholder(tf.bool)
        self.bool6 = tf.placeholder(tf.bool)
        self.bool7 = tf.placeholder(tf.bool)
        self.bool8 = tf.placeholder(tf.bool)
        
        ''' generate inputs ( imag + mask ) '''
        tmp_zeros = tf.zeros([self.nB,self.nCh_out,self.nY,self.nX],dtype)
        inp1 = tf.cond(self.bool1, lambda:tmp_zeros, lambda:self.a_img)
        inp2 = tf.cond(self.bool2, lambda:tmp_zeros, lambda:self.b_img)
        inp3 = tf.cond(self.bool3, lambda:tmp_zeros, lambda:self.c_img)
        inp4 = tf.cond(self.bool4, lambda:tmp_zeros, lambda:self.d_img)
        inp5 = tf.cond(self.bool5, lambda:tmp_zeros, lambda:self.e_img)      
        inp6 = tf.cond(self.bool6, lambda:tmp_zeros, lambda:self.f_img)
        inp7 = tf.cond(self.bool7, lambda:tmp_zeros, lambda:self.g_img)
        inp8 = tf.cond(self.bool8, lambda:tmp_zeros, lambda:self.n_img)      

        input_contrasts = tf.concat([inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8],axis=ch_dim) 
        self.inputs = tf.concat([input_contrasts, self.a_mask, self.b_mask,self.c_mask,self.d_mask, self.e_mask,self.f_mask,self.g_mask,self.n_mask],axis=ch_dim)

        ''' inference G, D for 1st input (not cyc) '''
        self.recon = self.G(self.inputs,self.is_Training)

        ## D(recon)
        RealFake_rec, self.type_rec = self.D(self.recon, self.is_Training)
        ## D(target)
        RealFake_tar, self.type_tar = self.D(self.targets, self.is_Training)

        ''' generate inputs for cyc '''
        # for cyc
        cyc1_ = tf.cond(self.bool1, lambda:self.recon, lambda:self.a_img)
        cyc2_ = tf.cond(self.bool2, lambda:self.recon, lambda:self.b_img)
        cyc3_ = tf.cond(self.bool3, lambda:self.recon, lambda:self.c_img)
        cyc4_ = tf.cond(self.bool4, lambda:self.recon, lambda:self.d_img)
        cyc5_ = tf.cond(self.bool5, lambda:self.recon, lambda:self.e_img)
        cyc6_ = tf.cond(self.bool6, lambda:self.recon, lambda:self.f_img)
        cyc7_ = tf.cond(self.bool7, lambda:self.recon, lambda:self.g_img)
        cyc8_ = tf.cond(self.bool8, lambda:self.recon, lambda:self.n_img)

        cyc_inp1_ = tf.concat([tmp_zeros,cyc2_,cyc3_,cyc4_,cyc5_,cyc6_,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp2_ = tf.concat([cyc1_,tmp_zeros,cyc3_,cyc4_,cyc5_,cyc6_,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp3_ = tf.concat([cyc1_,cyc2_,tmp_zeros,cyc4_,cyc5_,cyc6_,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp4_ = tf.concat([cyc1_,cyc2_,cyc3_,tmp_zeros,cyc5_,cyc6_,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp5_ = tf.concat([cyc1_,cyc2_,cyc3_,cyc4_,tmp_zeros,cyc6_,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp6_ = tf.concat([cyc1_,cyc2_,cyc3_,cyc4_,cyc5_,tmp_zeros,cyc7_,cyc8_],axis=ch_dim)
        cyc_inp7_ = tf.concat([cyc1_,cyc2_,cyc3_,cyc4_,cyc5_,cyc6_,tmp_zeros,cyc8_],axis=ch_dim)
        cyc_inp8_ = tf.concat([cyc1_,cyc2_,cyc3_,cyc4_,cyc5_,cyc6_,cyc7_,tmp_zeros],axis=ch_dim)

        atmp_zeros = tf.zeros([self.nB,1,self.nY,self.nX],dtype)
        atmp_ones  = tf.ones([self.nB,1,self.nY,self.nX],dtype)
        cyc_inp1 = tf.concat([cyc_inp1_,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp2 = tf.concat([cyc_inp2_,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp3 = tf.concat([cyc_inp3_,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp4 = tf.concat([cyc_inp4_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp5 = tf.concat([cyc_inp5_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp6 = tf.concat([cyc_inp6_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros,atmp_zeros],axis=ch_dim)
        cyc_inp7 = tf.concat([cyc_inp7_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones,atmp_zeros],axis=ch_dim)
        cyc_inp8 = tf.concat([cyc_inp8_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,atmp_ones],axis=ch_dim)

        ''' inference G, D for cyc inputs'''
        self.cyc1 = self.G(cyc_inp1, self.is_Training)
        self.cyc2 = self.G(cyc_inp2, self.is_Training)
        self.cyc3 = self.G(cyc_inp3, self.is_Training)
        self.cyc4 = self.G(cyc_inp4, self.is_Training)
        self.cyc5 = self.G(cyc_inp5, self.is_Training)
        self.cyc6 = self.G(cyc_inp6, self.is_Training)
        self.cyc7 = self.G(cyc_inp7, self.is_Training)
        self.cyc8 = self.G(cyc_inp8, self.is_Training)

        ## D(cyc), C(cyc)
        RealFake_cyc1,type_cyc1 = self.D(self.cyc1, self.is_Training)
        RealFake_cyc2,type_cyc2 = self.D(self.cyc2, self.is_Training)
        RealFake_cyc3,type_cyc3 = self.D(self.cyc3, self.is_Training)
        RealFake_cyc4,type_cyc4 = self.D(self.cyc4, self.is_Training)
        RealFake_cyc5,type_cyc5 = self.D(self.cyc5, self.is_Training)
        RealFake_cyc6,type_cyc6 = self.D(self.cyc6, self.is_Training)
        RealFake_cyc7,type_cyc7 = self.D(self.cyc7, self.is_Training)
        RealFake_cyc8,type_cyc8 = self.D(self.cyc8, self.is_Training)
        
        ## D(tar), C(tar)
        RealFake_tar1, type_tar1 = self.D(self.a_img, self.is_Training)
        RealFake_tar2, type_tar2 = self.D(self.b_img, self.is_Training)
        RealFake_tar3, type_tar3 = self.D(self.c_img, self.is_Training)
        RealFake_tar4, type_tar4 = self.D(self.d_img, self.is_Training)
        RealFake_tar5, type_tar5 = self.D(self.e_img, self.is_Training)
        RealFake_tar6, type_tar6 = self.D(self.f_img, self.is_Training)
        RealFake_tar7, type_tar7 = self.D(self.g_img, self.is_Training)
        RealFake_tar8, type_tar8 = self.D(self.n_img, self.is_Training)


        ''' Here, loss def starts here'''
        # gen loss for generator
        G_gan_loss_cyc1   = tf.reduce_mean(tf.squared_difference(RealFake_cyc1, REAL_LABEL))
        G_gan_loss_cyc2   = tf.reduce_mean(tf.squared_difference(RealFake_cyc2, REAL_LABEL))
        G_gan_loss_cyc3   = tf.reduce_mean(tf.squared_difference(RealFake_cyc3, REAL_LABEL))
        G_gan_loss_cyc4   = tf.reduce_mean(tf.squared_difference(RealFake_cyc4, REAL_LABEL))
        G_gan_loss_cyc5   = tf.reduce_mean(tf.squared_difference(RealFake_cyc5, REAL_LABEL))
        G_gan_loss_cyc6   = tf.reduce_mean(tf.squared_difference(RealFake_cyc6, REAL_LABEL))
        G_gan_loss_cyc7   = tf.reduce_mean(tf.squared_difference(RealFake_cyc7, REAL_LABEL))
        G_gan_loss_cyc8   = tf.reduce_mean(tf.squared_difference(RealFake_cyc8, REAL_LABEL))

        G_gan_loss_cyc  = G_gan_loss_cyc1 + G_gan_loss_cyc2 + G_gan_loss_cyc3 + G_gan_loss_cyc4 + G_gan_loss_cyc5 + G_gan_loss_cyc6 + G_gan_loss_cyc7 + G_gan_loss_cyc8

        G_gan_loss_orig   = tf.reduce_mean(tf.squared_difference(RealFake_rec, REAL_LABEL))
        G_gan_loss = (G_gan_loss_orig + G_gan_loss_cyc)/9.

        ## l2 loss for generator ( not use this l2 loss, just for monitoring )
        cyc_l2_loss1 = tf.reduce_mean(tf.squared_difference(self.cyc1, self.a_img))
        cyc_l2_loss2 = tf.reduce_mean(tf.squared_difference(self.cyc2, self.b_img))
        cyc_l2_loss3 = tf.reduce_mean(tf.squared_difference(self.cyc3, self.c_img))
        cyc_l2_loss4 = tf.reduce_mean(tf.squared_difference(self.cyc4, self.d_img))
        cyc_l2_loss5 = tf.reduce_mean(tf.squared_difference(self.cyc5, self.e_img))
        cyc_l2_loss6 = tf.reduce_mean(tf.squared_difference(self.cyc6, self.f_img))
        cyc_l2_loss7 = tf.reduce_mean(tf.squared_difference(self.cyc7, self.g_img))
        cyc_l2_loss8 = tf.reduce_mean(tf.squared_difference(self.cyc8, self.n_img))
        l2_cyc_loss = cyc_l2_loss1 + cyc_l2_loss2 + cyc_l2_loss3 + cyc_l2_loss4 + cyc_l2_loss5 + cyc_l2_loss6 + cyc_l2_loss7 + cyc_l2_loss8

        l2_loss_orig = tf.reduce_mean(tf.squared_difference(self.recon,self.targets))
        l2_loss = (l2_loss_orig+l2_cyc_loss)/9.


        ## l1 loss for generator
        cyc_l1_loss1 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc1, self.a_img))
        cyc_l1_loss2 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc2, self.b_img))
        cyc_l1_loss3 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc3, self.c_img))
        cyc_l1_loss4 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc4, self.d_img))
        cyc_l1_loss5 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc5, self.e_img))
        cyc_l1_loss6 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc6, self.f_img))
        cyc_l1_loss7 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc7, self.g_img))
        cyc_l1_loss8 = tf.reduce_mean(tf.losses.absolute_difference(self.cyc8, self.n_img))
        # for cyc
        cyc_l1_loss1 = tf.cond(self.bool1, lambda:0., lambda:cyc_l1_loss1)
        cyc_l1_loss2 = tf.cond(self.bool2, lambda:0., lambda:cyc_l1_loss2)
        cyc_l1_loss3 = tf.cond(self.bool3, lambda:0., lambda:cyc_l1_loss3)
        cyc_l1_loss4 = tf.cond(self.bool4, lambda:0., lambda:cyc_l1_loss4)
        cyc_l1_loss5 = tf.cond(self.bool5, lambda:0., lambda:cyc_l1_loss5)
        cyc_l1_loss6 = tf.cond(self.bool6, lambda:0., lambda:cyc_l1_loss6)
        cyc_l1_loss7 = tf.cond(self.bool7, lambda:0., lambda:cyc_l1_loss7)
        cyc_l1_loss8 = tf.cond(self.bool8, lambda:0., lambda:cyc_l1_loss8)
      
        l1_cyc_loss = cyc_l1_loss1 + cyc_l1_loss2 + cyc_l1_loss3 + cyc_l1_loss4 + cyc_l1_loss5  + cyc_l1_loss6 + cyc_l1_loss7 + cyc_l1_loss8

        l1_loss_orig = tf.reduce_mean(tf.losses.absolute_difference(self.recon,self.targets))
        l1_loss = (l1_loss_orig+l1_cyc_loss)/9.

        ## SSIM loss for generator ( temporally, working just nB=1 )
        ssim1R= tf.image.ssim(self.cyc1[0,0,:,:,tf.newaxis], self.a_img[0,0,:,:,tf.newaxis], 5)
        ssim1G= tf.image.ssim(self.cyc1[0,1,:,:,tf.newaxis], self.a_img[0,1,:,:,tf.newaxis], 5)
        ssim1B= tf.image.ssim(self.cyc1[0,2,:,:,tf.newaxis], self.a_img[0,2,:,:,tf.newaxis], 5)
        ssim2R= tf.image.ssim(self.cyc2[0,0,:,:,tf.newaxis], self.b_img[0,0,:,:,tf.newaxis], 5)
        ssim2G= tf.image.ssim(self.cyc2[0,1,:,:,tf.newaxis], self.b_img[0,1,:,:,tf.newaxis], 5)
        ssim2B= tf.image.ssim(self.cyc2[0,2,:,:,tf.newaxis], self.b_img[0,2,:,:,tf.newaxis], 5)
        ssim3R= tf.image.ssim(self.cyc3[0,0,:,:,tf.newaxis], self.c_img[0,0,:,:,tf.newaxis], 5)
        ssim3G= tf.image.ssim(self.cyc3[0,1,:,:,tf.newaxis], self.c_img[0,1,:,:,tf.newaxis], 5)
        ssim3B= tf.image.ssim(self.cyc3[0,2,:,:,tf.newaxis], self.c_img[0,2,:,:,tf.newaxis], 5)
        ssim4R= tf.image.ssim(self.cyc4[0,0,:,:,tf.newaxis], self.d_img[0,0,:,:,tf.newaxis], 5)
        ssim4G= tf.image.ssim(self.cyc4[0,1,:,:,tf.newaxis], self.d_img[0,1,:,:,tf.newaxis], 5)
        ssim4B= tf.image.ssim(self.cyc4[0,2,:,:,tf.newaxis], self.d_img[0,2,:,:,tf.newaxis], 5)
        ssim5R= tf.image.ssim(self.cyc5[0,0,:,:,tf.newaxis], self.e_img[0,0,:,:,tf.newaxis], 5)
        ssim5G= tf.image.ssim(self.cyc5[0,1,:,:,tf.newaxis], self.e_img[0,1,:,:,tf.newaxis], 5)
        ssim5B= tf.image.ssim(self.cyc5[0,2,:,:,tf.newaxis], self.e_img[0,2,:,:,tf.newaxis], 5)
        ssim6R= tf.image.ssim(self.cyc6[0,0,:,:,tf.newaxis], self.f_img[0,0,:,:,tf.newaxis], 5)
        ssim6G= tf.image.ssim(self.cyc6[0,1,:,:,tf.newaxis], self.f_img[0,1,:,:,tf.newaxis], 5)
        ssim6B= tf.image.ssim(self.cyc6[0,2,:,:,tf.newaxis], self.f_img[0,2,:,:,tf.newaxis], 5)
        ssim7R= tf.image.ssim(self.cyc7[0,0,:,:,tf.newaxis], self.g_img[0,0,:,:,tf.newaxis], 5)
        ssim7G= tf.image.ssim(self.cyc7[0,1,:,:,tf.newaxis], self.g_img[0,1,:,:,tf.newaxis], 5)
        ssim7B= tf.image.ssim(self.cyc7[0,2,:,:,tf.newaxis], self.g_img[0,2,:,:,tf.newaxis], 5)
        ssim8R= tf.image.ssim(self.cyc8[0,0,:,:,tf.newaxis], self.n_img[0,0,:,:,tf.newaxis], 5)
        ssim8G= tf.image.ssim(self.cyc8[0,1,:,:,tf.newaxis], self.n_img[0,1,:,:,tf.newaxis], 5)
        ssim8B= tf.image.ssim(self.cyc8[0,2,:,:,tf.newaxis], self.n_img[0,2,:,:,tf.newaxis], 5)

        ssimrR= tf.image.ssim(self.recon[0,0,:,:,tf.newaxis], self.targets[0,0,:,:,tf.newaxis], 5)
        ssimrG= tf.image.ssim(self.recon[0,1,:,:,tf.newaxis], self.targets[0,1,:,:,tf.newaxis], 5)
        ssimrB= tf.image.ssim(self.recon[0,2,:,:,tf.newaxis], self.targets[0,2,:,:,tf.newaxis], 5)
 
        cyc_ssim_loss1 = -tf.log( (1.0+ssim1R)/2.0) -tf.log( (1.0+ssim1G)/2.0) -tf.log( (1.0+ssim1B)/2.0) 
        cyc_ssim_loss2 = -tf.log( (1.0+ssim2R)/2.0) -tf.log( (1.0+ssim2G)/2.0) -tf.log( (1.0+ssim2B)/2.0)       
        cyc_ssim_loss3 = -tf.log( (1.0+ssim3R)/2.0) -tf.log( (1.0+ssim3G)/2.0) -tf.log( (1.0+ssim3B)/2.0) 
        cyc_ssim_loss4 = -tf.log( (1.0+ssim4R)/2.0) -tf.log( (1.0+ssim4G)/2.0) -tf.log( (1.0+ssim4B)/2.0) 
        cyc_ssim_loss5 = -tf.log( (1.0+ssim5R)/2.0) -tf.log( (1.0+ssim5G)/2.0) -tf.log( (1.0+ssim5B)/2.0)
        cyc_ssim_loss6 = -tf.log( (1.0+ssim6R)/2.0) -tf.log( (1.0+ssim6G)/2.0) -tf.log( (1.0+ssim6B)/2.0) 
        cyc_ssim_loss7 = -tf.log( (1.0+ssim7R)/2.0) -tf.log( (1.0+ssim7G)/2.0) -tf.log( (1.0+ssim7B)/2.0) 
        cyc_ssim_loss8 = -tf.log( (1.0+ssim8R)/2.0) -tf.log( (1.0+ssim8G)/2.0) -tf.log( (1.0+ssim8B)/2.0)

        cyc_ssim_loss1 = tf.cond(self.bool1, lambda:0., lambda:cyc_ssim_loss1)
        cyc_ssim_loss2 = tf.cond(self.bool2, lambda:0., lambda:cyc_ssim_loss2)
        cyc_ssim_loss3 = tf.cond(self.bool3, lambda:0., lambda:cyc_ssim_loss3)
        cyc_ssim_loss4 = tf.cond(self.bool4, lambda:0., lambda:cyc_ssim_loss4)
        cyc_ssim_loss5 = tf.cond(self.bool5, lambda:0., lambda:cyc_ssim_loss5)
        cyc_ssim_loss6 = tf.cond(self.bool6, lambda:0., lambda:cyc_ssim_loss6)
        cyc_ssim_loss7 = tf.cond(self.bool7, lambda:0., lambda:cyc_ssim_loss7)
        cyc_ssim_loss8 = tf.cond(self.bool8, lambda:0., lambda:cyc_ssim_loss8)
        ssim_cyc_loss  = cyc_ssim_loss1 + cyc_ssim_loss2 + cyc_ssim_loss3 + cyc_ssim_loss4 + cyc_ssim_loss5 + cyc_ssim_loss6 + cyc_ssim_loss7 + cyc_ssim_loss8
#
        ssim_loss_orig = -tf.log( (1.0+ssimrR)/2.0) -tf.log( (1.0+ssimrG)/2.0) -tf.log( (1.0+ssimrB)/2.0)  
        ssim_loss = (ssim_loss_orig + ssim_cyc_loss)/9.


        # some constants OH labels define here
        OH_label1 = tf.tile(tf.reshape(tf.one_hot(tf.cast(0,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label2 = tf.tile(tf.reshape(tf.one_hot(tf.cast(1,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label3 = tf.tile(tf.reshape(tf.one_hot(tf.cast(2,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label4 = tf.tile(tf.reshape(tf.one_hot(tf.cast(3,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label5 = tf.tile(tf.reshape(tf.one_hot(tf.cast(4,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label6 = tf.tile(tf.reshape(tf.one_hot(tf.cast(5,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label7 = tf.tile(tf.reshape(tf.one_hot(tf.cast(6,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_label8 = tf.tile(tf.reshape(tf.one_hot(tf.cast(7,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
        OH_labelT = tf.tile(tf.reshape(tf.one_hot(tf.cast(self.tar_class_idx,tf.uint8),self.class_N),[-1,1,self.class_N]),[1,1,1])
         
        '''classification loss for generator'''
        G_clsf_cyc_loss1 = tf.losses.softmax_cross_entropy(OH_label1, type_cyc1)
        G_clsf_cyc_loss2 = tf.losses.softmax_cross_entropy(OH_label2, type_cyc2)
        G_clsf_cyc_loss3 = tf.losses.softmax_cross_entropy(OH_label3, type_cyc3)
        G_clsf_cyc_loss4 = tf.losses.softmax_cross_entropy(OH_label4, type_cyc4)
        G_clsf_cyc_loss5 = tf.losses.softmax_cross_entropy(OH_label5, type_cyc5)
        G_clsf_cyc_loss6 = tf.losses.softmax_cross_entropy(OH_label6, type_cyc6)
        G_clsf_cyc_loss7 = tf.losses.softmax_cross_entropy(OH_label7, type_cyc7)
        G_clsf_cyc_loss8 = tf.losses.softmax_cross_entropy(OH_label8, type_cyc8)

        G_clsf_cyc_loss  = G_clsf_cyc_loss1 + G_clsf_cyc_loss2 + G_clsf_cyc_loss3 + G_clsf_cyc_loss4 + G_clsf_cyc_loss5  + G_clsf_cyc_loss6 + G_clsf_cyc_loss7 + G_clsf_cyc_loss8
        G_clsf_orig_loss = tf.losses.softmax_cross_entropy(OH_labelT, self.type_rec)
        G_clsf_loss = (G_clsf_orig_loss + G_clsf_cyc_loss)/9.
        
        ''' total generator loss '''
        self.G_loss       = (self.lambda_GAN*(G_gan_loss_orig + G_gan_loss_cyc) +
                 self.lambda_l1_cyc* ( l1_cyc_loss  ) + self.lambda_l1* (l1_loss_orig) +
                 self.lambda_l2_cyc* ( l2_cyc_loss  ) + self.lambda_l2* (l2_loss_orig) +
                 self.lambda_G_clsf* (G_clsf_orig_loss + G_clsf_cyc_loss) +
                 self.lambda_ssim_cyc* (ssim_cyc_loss)# +
                 )
        # not use l2 loss --> lambda_l2_cyc=0 

        # discriminator loss
        C_loss1 = tf.losses.softmax_cross_entropy(OH_label1,type_tar1)
        C_loss2 = tf.losses.softmax_cross_entropy(OH_label2,type_tar2)
        C_loss3 = tf.losses.softmax_cross_entropy(OH_label3,type_tar3)
        C_loss4 = tf.losses.softmax_cross_entropy(OH_label4,type_tar4)
        C_loss5 = tf.losses.softmax_cross_entropy(OH_label5,type_tar5)
        C_loss6 = tf.losses.softmax_cross_entropy(OH_label6,type_tar6)
        C_loss7 = tf.losses.softmax_cross_entropy(OH_label7,type_tar7)
        C_loss8 = tf.losses.softmax_cross_entropy(OH_label8,type_tar8)

        self.C_loss       = C_loss1 + C_loss2 + C_loss3 + C_loss4 + C_loss5 + C_loss6 + C_loss7 + C_loss8 
        
        if self.use_lsgan:
            err_real = tf.reduce_mean(tf.squared_difference(RealFake_tar, REAL_LABEL))
            err_fake = tf.reduce_mean(tf.square(RealFake_rec))
            D_err = err_real + err_fake

            cyc_real1 = tf.reduce_mean(tf.squared_difference(RealFake_tar1, REAL_LABEL))
            cyc_fake1 = tf.reduce_mean(tf.square(RealFake_cyc1))
            cyc_err1 = cyc_real1 + cyc_fake1 
            cyc_real2 = tf.reduce_mean(tf.squared_difference(RealFake_tar2, REAL_LABEL))
            cyc_fake2 = tf.reduce_mean(tf.square(RealFake_cyc2))
            cyc_err2 = cyc_real2 + cyc_fake2 
            cyc_real3 = tf.reduce_mean(tf.squared_difference(RealFake_tar3, REAL_LABEL))
            cyc_fake3 = tf.reduce_mean(tf.square(RealFake_cyc3))
            cyc_err3 = cyc_real3 + cyc_fake3 
            cyc_real4 = tf.reduce_mean(tf.squared_difference(RealFake_tar4, REAL_LABEL))
            cyc_fake4 = tf.reduce_mean(tf.square(RealFake_cyc4))
            cyc_err4 = cyc_real4 + cyc_fake4 
            cyc_real5 = tf.reduce_mean(tf.squared_difference(RealFake_tar5, REAL_LABEL))
            cyc_fake5 = tf.reduce_mean(tf.square(RealFake_cyc5))
            cyc_err5 = cyc_real5 + cyc_fake5
            cyc_real6 = tf.reduce_mean(tf.squared_difference(RealFake_tar6, REAL_LABEL))
            cyc_fake6 = tf.reduce_mean(tf.square(RealFake_cyc6))
            cyc_err6 = cyc_real6 + cyc_fake6 
            cyc_real7 = tf.reduce_mean(tf.squared_difference(RealFake_tar7, REAL_LABEL))
            cyc_fake7 = tf.reduce_mean(tf.square(RealFake_cyc7))
            cyc_err7 = cyc_real7 + cyc_fake7 
            cyc_real8 = tf.reduce_mean(tf.squared_difference(RealFake_tar8, REAL_LABEL))
            cyc_fake8 = tf.reduce_mean(tf.square(RealFake_cyc8))
            cyc_err8 = cyc_real8 + cyc_fake8

        else:
            # case for not using lsgan 
            st()

        D_gan_cyc  = cyc_err1 + cyc_err2 + cyc_err3 + cyc_err4 + cyc_err5  + cyc_err6 + cyc_err7 + cyc_err8 
        D_gan_loss  = (D_err + D_gan_cyc)/9.
        ##
        self.D_loss = (D_err + D_gan_cyc)/9. + (self.C_loss)/8.
        
        # Display
        tf.summary.scalar('0loss/G(ganfake + l2 +clsf)--0.16', self.G_loss)
        tf.summary.scalar('0loss/D(realfake+clsf)--0.41', self.D_loss)

        tf.summary.scalar('1G/G_gan', G_gan_loss)
        tf.summary.scalar('1G/L2', l2_loss)
        tf.summary.scalar('1G/SSIM', ssim_loss)
        tf.summary.scalar('1G/clsf', G_clsf_loss)
 
        tf.summary.scalar('2D/D_gan_loss(REAL/FAKE)', D_gan_loss)
        tf.summary.scalar('2D/C_loss(REAL)--1.386-->0', self.C_loss)

        tf.summary.scalar('G_gan(>0.16)/rec ', G_gan_loss_orig)
        tf.summary.scalar('G_gan(>0.16)/cyc1', G_gan_loss_cyc1)
        tf.summary.scalar('G_gan(>0.16)/cyc2', G_gan_loss_cyc2)
        tf.summary.scalar('G_gan(>0.16)/cyc3', G_gan_loss_cyc3)
        tf.summary.scalar('G_gan(>0.16)/cyc4', G_gan_loss_cyc4)
        tf.summary.scalar('G_gan(>0.16)/cyc5', G_gan_loss_cyc5)
        tf.summary.scalar('G_gan(>0.16)/cyc6', G_gan_loss_cyc6)
        tf.summary.scalar('G_gan(>0.16)/cyc7', G_gan_loss_cyc7)
        tf.summary.scalar('G_gan(>0.16)/cyc8', G_gan_loss_cyc8)

        tf.summary.scalar('G_l2/rec ', l2_loss_orig)
        tf.summary.scalar('G_l2/cyc1', cyc_l2_loss1)
        tf.summary.scalar('G_l2/cyc2', cyc_l2_loss2)
        tf.summary.scalar('G_l2/cyc3', cyc_l2_loss3)
        tf.summary.scalar('G_l2/cyc4', cyc_l2_loss4)
        tf.summary.scalar('G_l2/cyc5', cyc_l2_loss5)
        tf.summary.scalar('G_l2/cyc6', cyc_l2_loss6)
        tf.summary.scalar('G_l2/cyc7', cyc_l2_loss7)
        tf.summary.scalar('G_l2/cyc8', cyc_l2_loss8)

        tf.summary.scalar('G_ssim/rec ', ssimrR)
        tf.summary.scalar('G_ssim/cyc1', ssim1R)
        tf.summary.scalar('G_ssim/cyc2', ssim2R)
        tf.summary.scalar('G_ssim/cyc3', ssim3R)
        tf.summary.scalar('G_ssim/cyc4', ssim4R)
        tf.summary.scalar('G_ssim/cyc5', ssim5R)
        tf.summary.scalar('G_ssim/cyc6', ssim6R)
        tf.summary.scalar('G_ssim/cyc7', ssim7R)
        tf.summary.scalar('G_ssim/cyc8', ssim8R)

        tf.summary.scalar('G_clsf/rec_', G_clsf_orig_loss)      
        tf.summary.scalar('G_clsf/cyc_rec_a', G_clsf_cyc_loss1)      
        tf.summary.scalar('G_clsf/cyc_rec_b', G_clsf_cyc_loss2)     
        tf.summary.scalar('G_clsf/cyc_rec_c', G_clsf_cyc_loss3)      
        tf.summary.scalar('G_clsf/cyc_rec_d', G_clsf_cyc_loss4)     
        tf.summary.scalar('G_clsf/cyc_rec_e', G_clsf_cyc_loss5)    
        tf.summary.scalar('G_clsf/cyc_rec_f', G_clsf_cyc_loss6)      
        tf.summary.scalar('G_clsf/cyc_rec_g', G_clsf_cyc_loss7)     
        tf.summary.scalar('G_clsf/cyc_rec_n', G_clsf_cyc_loss8)    

        tf.summary.scalar('D_gan_loss(0.26)/Rec_err', D_err)
        tf.summary.scalar('D_gan_loss(0.26)/cyc1_err', cyc_err1)
        tf.summary.scalar('D_gan_loss(0.26)/cyc2_err', cyc_err2)
        tf.summary.scalar('D_gan_loss(0.26)/cyc3_err', cyc_err3)
        tf.summary.scalar('D_gan_loss(0.26)/cyc4_err', cyc_err4)
        tf.summary.scalar('D_gan_loss(0.26)/cyc5_err', cyc_err5)
        tf.summary.scalar('D_gan_loss(0.26)/cyc6_err', cyc_err6)
        tf.summary.scalar('D_gan_loss(0.26)/cyc7_err', cyc_err7)
        tf.summary.scalar('D_gan_loss(0.26)/cyc8_err', cyc_err8)

        tf.summary.scalar('C/a_img', C_loss1)
        tf.summary.scalar('C/b_img', C_loss2)
        tf.summary.scalar('C/c_img', C_loss3)
        tf.summary.scalar('C/d_img', C_loss4)
        tf.summary.scalar('C/e_img', C_loss5)
        tf.summary.scalar('C/f_img', C_loss6)
        tf.summary.scalar('C/g_img', C_loss7)
        tf.summary.scalar('C/n_img', C_loss8)
        # display an image

        tf.summary.image('1inputs/1Angry', self.tf_vis( self.inputs[:,0:3,:,:] ) )
        tf.summary.image('1inputs/2Contemptuous', self.tf_vis( self.inputs[:,3:6,:,:] ) )
        tf.summary.image('1inputs/3Disgusted', self.tf_vis( self.inputs[:,6:9,:,:] ) )
        tf.summary.image('1inputs/4Fearful', self.tf_vis( self.inputs[:,9:12,:,:] ) )
        tf.summary.image('1inputs/5Happy', self.tf_vis( self.inputs[:,12:15,:,:] ) )
        tf.summary.image('1inputs/6Sad', self.tf_vis( self.inputs[:,15:18,:,:] ) )
        tf.summary.image('1inputs/7Surprised', self.tf_vis( self.inputs[:,18:21,:,:] ) )
        tf.summary.image('1inputs/8Neutral', self.tf_vis( self.inputs[:,21:24,:,:] ) )
        tf.summary.image('4outputs/1Target', self.tf_vis( self.targets) )
       
        self.cyc1_rgbv = self.tf_vis(self.cyc1)
        self.cyc2_rgbv = self.tf_vis(self.cyc2)
        self.cyc3_rgbv = self.tf_vis(self.cyc3)
        self.cyc4_rgbv = self.tf_vis(self.cyc4)
        self.cyc5_rgbv = self.tf_vis(self.cyc5)
        self.cyc6_rgbv = self.tf_vis(self.cyc6)
        self.cyc7_rgbv = self.tf_vis(self.cyc7)
        self.cyc8_rgbv = self.tf_vis(self.cyc8)
        self.cyc1_rgb = self.tf_visout(self.cyc1)
        self.cyc2_rgb = self.tf_visout(self.cyc2)
        self.cyc3_rgb = self.tf_visout(self.cyc3)
        self.cyc4_rgb = self.tf_visout(self.cyc4)
        self.cyc5_rgb = self.tf_visout(self.cyc5)
        self.cyc6_rgb = self.tf_visout(self.cyc6)
        self.cyc7_rgb = self.tf_visout(self.cyc7)
        self.cyc8_rgb = self.tf_visout(self.cyc8)


        tf.summary.image('2cycle/1Angry', self.cyc1_rgbv)
        tf.summary.image('2cycle/2Contemptuous', self.cyc2_rgbv)
        tf.summary.image('2cycle/3Disgusted', self.cyc3_rgbv)
        tf.summary.image('2cycle/4Fearful', self.cyc4_rgbv)
        tf.summary.image('2cycle/5Happy', self.cyc5_rgbv)
        tf.summary.image('2cycle/6Sad', self.cyc6_rgbv)
        tf.summary.image('2cycle/7Surprised', self.cyc7_rgbv)
        tf.summary.image('2cycle/8Neutral', self.cyc8_rgbv)

        self.a_img_rgb = self.tf_visout( self.a_img )
        self.b_img_rgb = self.tf_visout( self.b_img )
        self.c_img_rgb = self.tf_visout( self.c_img )
        self.d_img_rgb = self.tf_visout( self.d_img )
        self.e_img_rgb = self.tf_visout( self.e_img )
        self.f_img_rgb = self.tf_visout( self.f_img )
        self.g_img_rgb = self.tf_visout( self.g_img )
        self.n_img_rgb = self.tf_visout( self.n_img )

        self.recon_rgb = self.tf_vis( self.recon )
        tf.summary.image('4outputs/2Recon', self.recon_rgb) 
        tf.summary.image('4outputs/3errx3', self.tf_vis_abs( 3*(self.recon-self.targets)))
    
        # display an image
        self.summary_op = tf.summary.merge_all()

        self.optimize(self.G_loss, self.D_loss, self.C_loss)

    def tf_visout(self, inp, order=[0,2,3,1]):
        return tf.transpose(inp,order)

    def tf_vis(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose(inp,order)*self.scale,tf.uint8)

    def tf_vis_abs(self, inp, order=[0,2,3,1]):
        return tf.cast( tf.transpose( tf.abs(inp),order)*self.scale,tf.uint8)

    def optimize(self, G_loss, D_loss, C_loss):
        def make_optimizer(loss, variables, lr,  name='Adam'):
            global_step = tf.Variable(0,trainable=False)
            decay_step  = 400
            lr_         = tf.train.exponential_decay(lr, global_step, decay_step,0.99,staircase=True)
            tf.summary.scalar('learning_rate/{}'.format(name), lr_)
            return tf.train.AdamOptimizer( lr_, beta1=0.5 , name=name).minimize(loss,global_step=global_step,var_list=variables)
        
        self.G_optm  = make_optimizer(G_loss, self.G.variables, self.lr,   name='Adam_G')
        self.D_optm  = make_optimizer(D_loss, self.D.variables, self.lr_D, name='Adam_D')
        self.C_optm  = make_optimizer(C_loss, self.D.variables, self.lr_C, name='Adam_C')

class Generator:
    def __init__(self,name,G, nCh_out,nCh=16, use_1x1Conv=False, w_decay=0, resid=False):
        if G=='UnetINMultiDiv8':
            self.net = UnetINMultiDiv8
        else:
            st()
        self.name = name
        self.nCh  = nCh
        self.nCh_out = nCh_out
        self.reuse = False
        self.use_1x1Conv=use_1x1Conv
        self.w_decay = w_decay 
        self.resid = resid
    def __call__(self, image, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay) if self.w_decay>0 else None 
            out = self.net(image, self.nCh_out, is_Training, reg_, nCh=self.nCh, _1x1Conv=self.use_1x1Conv)        
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return out#, logits

class Discriminator:
    def __init__(self, name='D', nCh=4, w_decay_D=0,DR_ratio=0,class_N=8):
        self.name   = name
        self.nCh    = [nCh, int(nCh*2), int(nCh*4),int(nCh*8), int(nCh*16), int(nCh*32)]
        self.reuse  = False
        self.k = 4
        self.kernel = 128/(2**(len(self.nCh)))
        self.w_decay_D = w_decay_D
        self.dropout_ratio = DR_ratio
        self.class_N=class_N

    def __call__(self, input, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay_D) if self.w_decay_D>0 else None

            ## img size =128 
            str_= self.name+'1D'
            h0  = lReLU( Conv2d2x2(input, kernel_size=self.k, ch_out=self.nCh[0], reg=reg_, name=str_), name=str_)
            ##img size = 64
            str_= self.name+'2D'
            h1  = lReLU( Conv2d2x2(   h0, kernel_size=self.k, ch_out=self.nCh[1], reg=reg_, name=str_), name=str_)
            ##img size = 32
            str_= self.name+'3D'
            h2  = lReLU( Conv2d2x2(   h1, kernel_size=self.k, ch_out=self.nCh[2], reg=reg_, name=str_), name=str_)
            ###img size = 16
            str_= self.name+'4D'
            h3 = lReLU( Conv2d2x2( h2, kernel_size=self.k, ch_out=self.nCh[3], reg=reg_, name=str_), name=str_)
            ##img size = 8
            str_= self.name+'5D'
            h4 = lReLU( Conv2d2x2( h3, kernel_size=self.k, ch_out=self.nCh[4], reg=reg_, name=str_), name=str_)
            ##img size = 4
            str_= self.name+'6D'
            hLast = lReLU( Conv2d2x2( h4, kernel_size=self.k, ch_out=self.nCh[5], reg=reg_, name=str_), name=str_)
            ##img size = 2
            
            hLast = tf.layers.dropout(hLast, rate=self.dropout_ratio,training=is_Training)
            RF_out = tf.layers.conv2d(hLast, filters=1,kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), name='RF_conv')
            logits = tf.layers.conv2d(hLast, filters=self.class_N,kernel_size=(self.kernel,self.kernel), strides=(1,1), padding="VALID", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), name='LastD_class')
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return RF_out, logits[:,tf.newaxis,:,0,0]


