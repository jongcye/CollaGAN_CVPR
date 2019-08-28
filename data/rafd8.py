from os import listdir
from os.path import join, isfile
from scipy import misc
import numpy as np
import random
from ipdb import set_trace as st
from math import ceil
import copy
import time

class facialExp():
    def __init__(self):
        super(facialExp, self).__init__()
    
    def name(self):
        return 'rafd-facialExp8'

    def initialize(self, opt, phase):
        t_start = time.time()
        random.seed(0)
        self.root   = opt.dataroot
        self.flist = np.load(join(opt.dataroot, phase+'_flist_main2.npy'))     

        self.N = 8
        self.nCh_out = 3#opt.nCh_out
        self.nCh_in  = self.N*self.nCh_out + self.N #opt.nCh_in 
        self.nY      = 128 
        self.nX      = 128
        self.len     = len(self.flist) 
        self.fExp = ['A','B','C','D','E','F','G','N']
        self.use_aug = (phase=='train') and opt.AUG
        self.use_norm_std = (not opt.wo_norm_std)
        self.N_null  = opt.N_null

        # Here, for dropout input
        self.null_N_set = [x+1 for x in range(opt.N_null)] #[1,2,3,4]
        self.list_for_null = []
        
        for i in range(self.N):
            self.list_for_null.append( self.get_null_list_for_idx(i) )

    ''' Here, initialize the null vectors for random input dropout selection''' 
    def get_null_list_for_idx(self, idx):
        a_list = []
        for i_null in self.null_N_set:
            tmp_a = []
            if i_null == 1:
                tmp = [ bX==idx for bX in range(self.N) ]
                tmp_a.append(tmp)

            elif i_null ==2:
                for i_in in range(self.N):
                    if not i_in==idx:
                        tmp = [ bX in [i_in, idx] for bX in range(self.N) ]
                        tmp_a.append(tmp)
            
            elif i_null ==3:
                for i_in in range(self.N):
                    for ii_in in range(self.N):
                        if not (i_in==ii_in or (i_in==idx or ii_in==idx)):
                            tmp = [ ( bX in [i_in, ii_in, idx]) for bX in range(self.N) ]
                            tmp_a.append(tmp)
            
            elif i_null ==4:
                for i_in in range(self.N):
                    for ii_in in range(self.N):
                        for iii_in in range(self.N):
                            if not ( (i_in==ii_in or i_in==iii_in or ii_in==iii_in)  or (i_in==idx or ii_in==idx or iii_in==idx)):
                                tmp = [ (bX in [i_in, ii_in, iii_in, idx]) for bX in range(self.N) ]
                                tmp_a.append(tmp)
            elif i_null ==5:
                for i4_in in range(self.N):
                    for i5_in in range(self.N):
                        for i6_in in range(self.N):
                            if not ( (idx in [i4_in, i5_in, i6_in]) or (i4_in==i5_in or i4_in==i6_in or i5_in==i6_in) ):
                                tmp = [ (bX==idx) or not ( bX in [i4_in, i5_in, i6_in]) for bX in range(self.N) ]
                                tmp_a.append(tmp)
            elif i_null ==6:
                for i5_in in range(self.N):
                    for i6_in in range(self.N):
                        if not (idx==i5_in or idx==i6_in or i5_in==i6_in):
                            tmp = [ (bX==idx or not ( bX in [i5_in,i6_in] )) for bX in range(self.N) ]
                            tmp_a.append(tmp)
            elif i_null ==7:
                for i6_in in range(self.N):
                    if not (i6_in==idx):
                        tmp = [ (bX==idx or (not bX==i6_in)) for bX in range(self.N) ]
                        tmp_a.append(tmp)
            else:
                st()
            
            a_list.append(tmp_a)

        return a_list 


    def get_info(self,opt):
        opt.nCh_in = self.nCh_in
        opt.nCh_out= self.nCh_out
        opt.nY     = self.nY
        opt.nX     = self.nX
        return opt

    def getBatch_RGB(self, start, end):
        end         = min([end,self.len])
        batch       = self.flist[start:end]
        # channel First :
        sz_a   = [end-start,self.nCh_out,  self.nY, self.nX] 
        sz_M   = [end-start,  1,  self.nY, self.nX] 

        target_class_idx = np.empty([end-start,1],dtype=np.uint8)
        a_img = np.empty(sz_a, dtype=np.float32)
        b_img = np.empty(sz_a, dtype=np.float32)
        c_img = np.empty(sz_a, dtype=np.float32)
        d_img = np.empty(sz_a, dtype=np.float32)
        e_img = np.empty(sz_a, dtype=np.float32)
        f_img = np.empty(sz_a, dtype=np.float32)
        g_img = np.empty(sz_a, dtype=np.float32)
        n_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)
   
        a_mask = np.zeros(sz_M, dtype=np.float32)
        b_mask = np.zeros(sz_M, dtype=np.float32)
        c_mask = np.zeros(sz_M, dtype=np.float32)
        d_mask = np.zeros(sz_M, dtype=np.float32)
        e_mask = np.zeros(sz_M, dtype=np.float32)
        f_mask = np.zeros(sz_M, dtype=np.float32)
        g_mask = np.zeros(sz_M, dtype=np.float32)
        n_mask = np.zeros(sz_M, dtype=np.float32)
        targ_idx = random.randint(0,self.N-1)
        tar_class_bools = [ x==targ_idx for x in range(self.N) ]
        
        for iB, aFname in enumerate(batch):
            aug_idx = random.randint(0,1)
            a_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[0]+'.png')) ,dtype=np.float32) 
            b_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[1]+'.png')) ,dtype=np.float32)
            c_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[2]+'.png')) ,dtype=np.float32) 
            d_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[3]+'.png')) ,dtype=np.float32) 
            e_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[4]+'.png')) ,dtype=np.float32) 
            f_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[5]+'.png')) ,dtype=np.float32) 
            g_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[6]+'.png')) ,dtype=np.float32) 
            n_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[7]+'.png')) ,dtype=np.float32) 

            if self.use_aug:
                if aug_idx==1:
                    a_tmp = np.flip(a_tmp,axis=3)
                    b_tmp = np.flip(b_tmp,axis=3)
                    c_tmp = np.flip(c_tmp,axis=3)
                    d_tmp = np.flip(d_tmp,axis=3)
                    e_tmp = np.flip(e_tmp,axis=3)
                    f_tmp = np.flip(f_tmp,axis=3)
                    g_tmp = np.flip(g_tmp,axis=3)
                    n_tmp = np.flip(n_tmp,axis=3)
            if self.use_norm_std:
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/np.std(a_tmp)
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/np.std(b_tmp)
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/np.std(c_tmp)
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/np.std(d_tmp)
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/np.std(e_tmp)
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/np.std(f_tmp)
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/np.std(g_tmp)
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/np.std(n_tmp)
            else:
                scale=255.0
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/scale
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/scale
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/scale
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/scale
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/scale
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/scale
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/scale
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/scale
                         
            if targ_idx ==0:
                target_img[iB,:,:,:] = a_img[iB,:,:,:]
                a_mask[iB,0,:,:] = 1.
            elif targ_idx ==1:
                target_img[iB,:,:,:] = b_img[iB,:,:,:]
                b_mask[iB,0,:,:] = 1.
            elif targ_idx ==2:
                target_img[iB,:,:,:] = c_img[iB,:,:,:]
                c_mask[iB,0,:,:] = 1.
            elif targ_idx ==3:
                target_img[iB,:,:,:] = d_img[iB,:,:,:]
                d_mask[iB,0,:,:] = 1.
            elif targ_idx ==4:
                target_img[iB,:,:,:] = e_img[iB,:,:,:]
                e_mask[iB,0,:,:] = 1.
            elif targ_idx ==5:
                target_img[iB,:,:,:] = f_img[iB,:,:,:]
                f_mask[iB,0,:,:] = 1.
            elif targ_idx ==6:
                target_img[iB,:,:,:] = g_img[iB,:,:,:]
                g_mask[iB,0,:,:] = 1.
            elif targ_idx ==7:
                target_img[iB,:,:,:] = n_img[iB,:,:,:]
                n_mask[iB,0,:,:] = 1.
            else:
                st()
            target_class_idx[iB] = targ_idx
        return target_class_idx, a_img, b_img, c_img, d_img,e_img,  f_img, g_img,n_img, a_mask, b_mask, c_mask, d_mask, e_mask, f_mask, g_mask, n_mask, tar_class_bools, target_img 

    def getBatch_RGB_varInp(self, start, end):
        nB = end-start
        end         = min([end,self.len])
        start = end-nB
        batch       = self.flist[start:end]
        # channel First :
        sz_a   = [nB,self.nCh_out,  self.nY, self.nX] 
        sz_M   = [nB,  1,  self.nY, self.nX] 

        target_class_idx = np.empty([nB,1],dtype=np.uint8)
        a_img = np.empty(sz_a, dtype=np.float32)
        b_img = np.empty(sz_a, dtype=np.float32)
        c_img = np.empty(sz_a, dtype=np.float32)
        d_img = np.empty(sz_a, dtype=np.float32)
        e_img = np.empty(sz_a, dtype=np.float32)
        f_img = np.empty(sz_a, dtype=np.float32)
        g_img = np.empty(sz_a, dtype=np.float32)
        n_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)
   
        a_mask = np.zeros(sz_M, dtype=np.float32)
        b_mask = np.zeros(sz_M, dtype=np.float32)
        c_mask = np.zeros(sz_M, dtype=np.float32)
        d_mask = np.zeros(sz_M, dtype=np.float32)
        e_mask = np.zeros(sz_M, dtype=np.float32)
        f_mask = np.zeros(sz_M, dtype=np.float32)
        g_mask = np.zeros(sz_M, dtype=np.float32)
        n_mask = np.zeros(sz_M, dtype=np.float32)

        targ_idx = random.randint(0,self.N-1)
        # Here, choose the random null idx in the set, and which is not in the target
        N_for_null = random.randint(0,len(self.null_N_set)-1)
        # 0: 7-->1 map
        # 6: 1-->1 map
        cur_list = self.list_for_null[targ_idx][N_for_null]
        if len(cur_list)==1:
            tar_class_bools = cur_list[0]
        else:
            tar_class_bools = cur_list[random.randint(0,len(cur_list)-1)]
        
        for iB, aFname in enumerate(batch):
            aug_idx = random.randint(0,1)
            a_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[0]+'.png')) ,dtype=np.float32) 
            b_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[1]+'.png')) ,dtype=np.float32)
            c_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[2]+'.png')) ,dtype=np.float32) 
            d_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[3]+'.png')) ,dtype=np.float32) 
            e_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[4]+'.png')) ,dtype=np.float32) 
            f_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[5]+'.png')) ,dtype=np.float32) 
            g_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[6]+'.png')) ,dtype=np.float32) 
            n_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[7]+'.png')) ,dtype=np.float32) 

            if self.use_aug:
                if aug_idx==1:
                    a_tmp = np.flip(a_tmp,axis=3)
                    b_tmp = np.flip(b_tmp,axis=3)
                    c_tmp = np.flip(c_tmp,axis=3)
                    d_tmp = np.flip(d_tmp,axis=3)
                    e_tmp = np.flip(e_tmp,axis=3)
                    f_tmp = np.flip(f_tmp,axis=3)
                    g_tmp = np.flip(g_tmp,axis=3)
                    n_tmp = np.flip(n_tmp,axis=3)
            if self.use_norm_std:
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/np.std(a_tmp)
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/np.std(b_tmp)
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/np.std(c_tmp)
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/np.std(d_tmp)
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/np.std(e_tmp)
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/np.std(f_tmp)
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/np.std(g_tmp)
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/np.std(n_tmp)
            else:
                scale=255.0
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/scale
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/scale
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/scale
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/scale
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/scale
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/scale
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/scale
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/scale
                         
            if targ_idx ==0:
                target_img[iB,:,:,:] = a_img[iB,:,:,:]
                a_mask[iB,0,:,:] = 1.
            elif targ_idx ==1:
                target_img[iB,:,:,:] = b_img[iB,:,:,:]
                b_mask[iB,0,:,:] = 1.
            elif targ_idx ==2:
                target_img[iB,:,:,:] = c_img[iB,:,:,:]
                c_mask[iB,0,:,:] = 1.
            elif targ_idx ==3:
                target_img[iB,:,:,:] = d_img[iB,:,:,:]
                d_mask[iB,0,:,:] = 1.
            elif targ_idx ==4:
                target_img[iB,:,:,:] = e_img[iB,:,:,:]
                e_mask[iB,0,:,:] = 1.
            elif targ_idx ==5:
                target_img[iB,:,:,:] = f_img[iB,:,:,:]
                f_mask[iB,0,:,:] = 1.
            elif targ_idx ==6:
                target_img[iB,:,:,:] = g_img[iB,:,:,:]
                g_mask[iB,0,:,:] = 1.
            elif targ_idx ==7:
                target_img[iB,:,:,:] = n_img[iB,:,:,:]
                n_mask[iB,0,:,:] = 1.
            else:
                st()
            target_class_idx[iB] = targ_idx
        return target_class_idx, a_img, b_img, c_img, d_img,e_img,  f_img, g_img,n_img, a_mask, b_mask, c_mask, d_mask, e_mask, f_mask, g_mask, n_mask, tar_class_bools, target_img 

    '''this function is made for the rebuttal '''
    def getBatch_RGB_varInp_tarid_missid(self, start, end, tar_id, miss_id):
        
        nB = end-start
        end         = min([end,self.len])
        start = end-nB
        batch       = self.flist[start:end]
        # channel First :
        sz_a   = [nB,self.nCh_out,  self.nY, self.nX] 
        sz_M   = [nB,  1,  self.nY, self.nX] 

        target_class_idx = np.empty([nB,1],dtype=np.uint8)
        a_img = np.empty(sz_a, dtype=np.float32)
        b_img = np.empty(sz_a, dtype=np.float32)
        c_img = np.empty(sz_a, dtype=np.float32)
        d_img = np.empty(sz_a, dtype=np.float32)
        e_img = np.empty(sz_a, dtype=np.float32)
        f_img = np.empty(sz_a, dtype=np.float32)
        g_img = np.empty(sz_a, dtype=np.float32)
        n_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)
   
        a_mask = np.zeros(sz_M, dtype=np.float32)
        b_mask = np.zeros(sz_M, dtype=np.float32)
        c_mask = np.zeros(sz_M, dtype=np.float32)
        d_mask = np.zeros(sz_M, dtype=np.float32)
        e_mask = np.zeros(sz_M, dtype=np.float32)
        f_mask = np.zeros(sz_M, dtype=np.float32)
        g_mask = np.zeros(sz_M, dtype=np.float32)
        n_mask = np.zeros(sz_M, dtype=np.float32)
        
        targ_idx = tar_id #random.randint(0,self.N-1)
        # Here, choose the random null idx in the set, and which is not in the target
        if tar_id==miss_id:
            N_for_null = 0
        else:
            N_for_null = 1 # random.randint(0,len(self.null_N_set)-1)
        # 0: 7-->1 map
        # 6: 1-->1 map
        cur_list = self.list_for_null[targ_idx][N_for_null]
        
        if len(cur_list)==1:
            tar_class_bools = cur_list[0]
        else:
            if miss_id>tar_id:
                s = -1
            else:
                s = 0
            tar_class_bools = cur_list[miss_id+s]#random.randint(0,len(cur_list)-1)]
        
        for iB, aFname in enumerate(batch):
            aug_idx = random.randint(0,1)
            a_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[0]+'.png')) ,dtype=np.float32) 
            b_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[1]+'.png')) ,dtype=np.float32)
            c_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[2]+'.png')) ,dtype=np.float32) 
            d_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[3]+'.png')) ,dtype=np.float32) 
            e_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[4]+'.png')) ,dtype=np.float32) 
            f_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[5]+'.png')) ,dtype=np.float32) 
            g_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[6]+'.png')) ,dtype=np.float32) 
            n_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[7]+'.png')) ,dtype=np.float32) 

            if self.use_aug:
                if aug_idx==1:
                    a_tmp = np.flip(a_tmp,axis=3)
                    b_tmp = np.flip(b_tmp,axis=3)
                    c_tmp = np.flip(c_tmp,axis=3)
                    d_tmp = np.flip(d_tmp,axis=3)
                    e_tmp = np.flip(e_tmp,axis=3)
                    f_tmp = np.flip(f_tmp,axis=3)
                    g_tmp = np.flip(g_tmp,axis=3)
                    n_tmp = np.flip(n_tmp,axis=3)
            if self.use_norm_std:
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/np.std(a_tmp)
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/np.std(b_tmp)
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/np.std(c_tmp)
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/np.std(d_tmp)
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/np.std(e_tmp)
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/np.std(f_tmp)
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/np.std(g_tmp)
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/np.std(n_tmp)
            else:
                scale=255.0
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/scale
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/scale
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/scale
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/scale
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/scale
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/scale
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/scale
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/scale
                         
            if targ_idx ==0:
                target_img[iB,:,:,:] = a_img[iB,:,:,:]
                a_mask[iB,0,:,:] = 1.
            elif targ_idx ==1:
                target_img[iB,:,:,:] = b_img[iB,:,:,:]
                b_mask[iB,0,:,:] = 1.
            elif targ_idx ==2:
                target_img[iB,:,:,:] = c_img[iB,:,:,:]
                c_mask[iB,0,:,:] = 1.
            elif targ_idx ==3:
                target_img[iB,:,:,:] = d_img[iB,:,:,:]
                d_mask[iB,0,:,:] = 1.
            elif targ_idx ==4:
                target_img[iB,:,:,:] = e_img[iB,:,:,:]
                e_mask[iB,0,:,:] = 1.
            elif targ_idx ==5:
                target_img[iB,:,:,:] = f_img[iB,:,:,:]
                f_mask[iB,0,:,:] = 1.
            elif targ_idx ==6:
                target_img[iB,:,:,:] = g_img[iB,:,:,:]
                g_mask[iB,0,:,:] = 1.
            elif targ_idx ==7:
                target_img[iB,:,:,:] = n_img[iB,:,:,:]
                n_mask[iB,0,:,:] = 1.
            else:
                st()
            target_class_idx[iB] = targ_idx
        return target_class_idx, a_img, b_img, c_img, d_img,e_img,  f_img, g_img,n_img, a_mask, b_mask, c_mask, d_mask, e_mask, f_mask, g_mask, n_mask, tar_class_bools, target_img 



    def getBatch_RGB_varInpID(self, start, end):
        end         = min([end,self.len])
        batch       = self.flist[start:end]

        # channel First :
        sz_a   = [end-start,self.nCh_out,  self.nY, self.nX] 
        sz_M   = [end-start,  1,  self.nY, self.nX] 

        target_class_idx = np.empty([end-start,1],dtype=np.uint8)
        a_img = np.empty(sz_a, dtype=np.float32)
        b_img = np.empty(sz_a, dtype=np.float32)
        c_img = np.empty(sz_a, dtype=np.float32)
        d_img = np.empty(sz_a, dtype=np.float32)
        e_img = np.empty(sz_a, dtype=np.float32)
        f_img = np.empty(sz_a, dtype=np.float32)
        g_img = np.empty(sz_a, dtype=np.float32)
        n_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)
   
        a_mask = np.zeros(sz_M, dtype=np.float32)
        b_mask = np.zeros(sz_M, dtype=np.float32)
        c_mask = np.zeros(sz_M, dtype=np.float32)
        d_mask = np.zeros(sz_M, dtype=np.float32)
        e_mask = np.zeros(sz_M, dtype=np.float32)
        f_mask = np.zeros(sz_M, dtype=np.float32)
        g_mask = np.zeros(sz_M, dtype=np.float32)
        n_mask = np.zeros(sz_M, dtype=np.float32)


        targ_idx = random.randint(0,self.N-1)
        tar_class_bools = [ x==targ_idx for x in range(self.N) ]
 
        # Here, choose the random file in the set, and which is not in the target
        bFname      = self.flist[random.randint(0,self.len-1)]   
        random_for_batch2 = []

        random.seed(2)
        a_rand = random.randint(0,self.N-1)
        
        # change here to edit change ID N
        for i in range(1):
            while (a_rand in random_for_batch2) or (a_rand==targ_idx):
                a_rand = random.randint(0,self.N-1)
            random_for_batch2.append(a_rand)
        random_for_batch2.sort()
        random_bools = [ x in random_for_batch2 for x in range(self.N) ]
        
       
        for iB, aFname in enumerate(batch):
            aug_idx = random.randint(0,1)
            lFname = bFname if random_bools[0] else aFname
            a_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[0]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[1] else aFname
            b_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[1]+'.png')) ,dtype=np.float32)
            lFname = bFname if random_bools[2] else aFname
            c_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[2]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[3] else aFname
            d_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[3]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[4] else aFname
            e_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[4]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[5] else aFname
            f_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[5]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[6] else aFname
            g_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[6]+'.png')) ,dtype=np.float32) 
            lFname = bFname if random_bools[7] else aFname
            n_tmp = np.ndarray.astype( self.read_png( join(self.root, lFname+self.fExp[7]+'.png')) ,dtype=np.float32) 

            if self.use_aug:
                if aug_idx==1:
                    a_tmp = np.flip(a_tmp,axis=3)
                    b_tmp = np.flip(b_tmp,axis=3)
                    c_tmp = np.flip(c_tmp,axis=3)
                    d_tmp = np.flip(d_tmp,axis=3)
                    e_tmp = np.flip(e_tmp,axis=3)
                    f_tmp = np.flip(f_tmp,axis=3)
                    g_tmp = np.flip(g_tmp,axis=3)
                    n_tmp = np.flip(n_tmp,axis=3)
            if self.use_norm_std:
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/np.std(a_tmp)
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/np.std(b_tmp)
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/np.std(c_tmp)
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/np.std(d_tmp)
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/np.std(e_tmp)
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/np.std(f_tmp)
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/np.std(g_tmp)
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/np.std(n_tmp)
            else:
                scale=255.0
                a_img[iB,:,:,:] = a_tmp[:,:,:,0]/scale
                b_img[iB,:,:,:] = b_tmp[:,:,:,0]/scale
                c_img[iB,:,:,:] = c_tmp[:,:,:,0]/scale
                d_img[iB,:,:,:] = d_tmp[:,:,:,0]/scale
                e_img[iB,:,:,:] = e_tmp[:,:,:,0]/scale
                f_img[iB,:,:,:] = f_tmp[:,:,:,0]/scale
                g_img[iB,:,:,:] = g_tmp[:,:,:,0]/scale
                n_img[iB,:,:,:] = n_tmp[:,:,:,0]/scale
                         
            if targ_idx ==0:
                target_img[iB,:,:,:] = a_img[iB,:,:,:]
                a_mask[iB,0,:,:] = 1.
            elif targ_idx ==1:
                target_img[iB,:,:,:] = b_img[iB,:,:,:]
                b_mask[iB,0,:,:] = 1.
            elif targ_idx ==2:
                target_img[iB,:,:,:] = c_img[iB,:,:,:]
                c_mask[iB,0,:,:] = 1.
            elif targ_idx ==3:
                target_img[iB,:,:,:] = d_img[iB,:,:,:]
                d_mask[iB,0,:,:] = 1.
            elif targ_idx ==4:
                target_img[iB,:,:,:] = e_img[iB,:,:,:]
                e_mask[iB,0,:,:] = 1.
            elif targ_idx ==5:
                target_img[iB,:,:,:] = f_img[iB,:,:,:]
                f_mask[iB,0,:,:] = 1.
            elif targ_idx ==6:
                target_img[iB,:,:,:] = g_img[iB,:,:,:]
                g_mask[iB,0,:,:] = 1.
            elif targ_idx ==7:
                target_img[iB,:,:,:] = n_img[iB,:,:,:]
                n_mask[iB,0,:,:] = 1.
            else:
                st()
            target_class_idx[iB] = targ_idx
        return target_class_idx, a_img, b_img, c_img, d_img,e_img,  f_img, g_img,n_img, a_mask, b_mask, c_mask, d_mask, e_mask, f_mask, g_mask, n_mask, tar_class_bools, target_img 
    
    def getBatch_std(self, start, end):
        end         = min([end,self.len])
        batch       = self.flist[start:end]
       
        for iB, aFname in enumerate(batch):
            aug_idx = random.randint(0,1)
            a_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[0]+'.png')) ,dtype=np.float32) 
            b_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[1]+'.png')) ,dtype=np.float32)
            c_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[2]+'.png')) ,dtype=np.float32) 
            d_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[3]+'.png')) ,dtype=np.float32) 
            e_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[4]+'.png')) ,dtype=np.float32) 
            f_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[5]+'.png')) ,dtype=np.float32) 
            g_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[6]+'.png')) ,dtype=np.float32) 
            n_tmp = np.ndarray.astype( self.read_png( join(self.root, aFname+self.fExp[7]+'.png')) ,dtype=np.float32) 

            a_std = np.std(a_tmp)
            b_std = np.std(b_tmp)
            c_std = np.std(c_tmp)
            d_std = np.std(d_tmp)
            e_std = np.std(e_tmp)
            f_std = np.std(f_tmp)
            g_std = np.std(g_tmp)
            n_std = np.std(n_tmp)
                        
            return a_std, b_std, c_std, d_std, e_std, f_std, g_std, n_std
    
    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    def __len__(self):
        return self.len
    @staticmethod
    def read_png(filename):
        png = misc.imread(filename)
        png = np.swapaxes( png[np.newaxis,:,:,:], 0,3 )
        return png

if __name__ == "__main__":
    tmp = DB_HCP('../../data/MRI')


