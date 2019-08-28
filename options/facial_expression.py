import argparse
import os
from util import util
from ipdb import set_trace as st
# for gray scal : input_nc, output_nc, ngf, ndf, gpu_ids, batchSize, norm

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--G', type=str, default='UnetINMultiDiv8', help='choice of network for Generator')
        self.parser.add_argument('--dataroot', type=str, default='./../../Hdd/multipie/onlyFace', help='data root')
        self.parser.add_argument('--savepath', type=str, default='./result', help='savepath')
        self.parser.add_argument('--nEpoch', type=int, default=1000, help='number of Epoch iteration')
        self.parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
        self.parser.add_argument('--lr_D', type=float, default=0.00001, help='learning rate for D')
        self.parser.add_argument('--lr_C', type=float, default=0.00001, help='learning rate for C')
        self.parser.add_argument('--disp_div_N', type=int, default=7, help=' display N per epoch')
        self.parser.add_argument('--nB', type=int, default=1, help='input batch size')
        self.parser.add_argument('--DB_small', action='store_true', help='use small DB')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--w_decay', type=float, default=0.01, help='weight decay for generator')
        self.parser.add_argument('--w_decay_D', type=float, default=0., help='weight decay for discriminator')
        self.parser.add_argument('--lambda_l1_cyc', type=float, default=1., help='lambda_L1_cyc, StarGAN cyc loss rec')
        self.parser.add_argument('--lambda_l2_cyc', type=float, default=0., help='lambda_L2_cyc, StarGAN cyc loss rec')
        self.parser.add_argument('--lambda_ssim_cyc', type=float, default=10., help='lambda_ssim')
        self.parser.add_argument('--lambda_l2', type=float, default=0., help='lambda_L2')
        self.parser.add_argument('--lambda_l1', type=float, default=10., help='lambda_L1')
        self.parser.add_argument('--lambda_ssim', type=float, default=0., help='lambda_ssim')
        self.parser.add_argument('--lambda_GAN', type=float, default=1., help='lambda GAN')
        self.parser.add_argument('--lambda_G_clsf', type=float, default=1., help='generator classification loss. fake to be well classified')
        self.parser.add_argument('--lambda_D_clsf', type=float, default=1., help='discriminator classification loss. fake to be well classified')
        self.parser.add_argument('--lambda_cyc', type=float, default=1, help='lambda_cyc')
        self.parser.add_argument('--nEpochDclsf', type=int, default=30, help='# of nEpoch for Discriminator pretrain')
        self.parser.add_argument('--nCh_D', type=int, default=64, help='# of ngf for Discriminator')
        self.parser.add_argument('--nCh_C', type=int, default=16, help='# of ngf for Classifier')
        self.parser.add_argument('--use_lsgan', action='store_true', help='use lsgan, if not defualt GAN')
        self.parser.add_argument('--use_resid', action='store_true', help='use resid')
        self.parser.add_argument('--use_1x1Conv', action='store_true', help='use 1x1Conv, if not defualt 3x3conv')
        self.parser.add_argument('--wo_norm_std', action='store_true', help='NOT use std normalization')
        self.parser.add_argument('--ngf', type=int, default=64, help=' ngf')
        self.parser.add_argument('--dropout_G', type=float, default=0., help='droptout ') 
        self.parser.add_argument('--dropout', type=float, default=0.5, help='droptout ')
        self.parser.add_argument('--test_mode', action='store_true', help='not train. just test')
        self.parser.add_argument('--AUG', action='store_true', help='use augmentation')
        self.parser.add_argument('--N_null', type=int, default=4, help = '1-7')
        self.parser.add_argument('--nEpochD', type=int, default=1, help = 'nEpochD update while 1 G update')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.savepath, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt





