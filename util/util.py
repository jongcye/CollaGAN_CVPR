import tensorflow as tf
import numpy as np
from ipdb import set_trace as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ipdb import set_trace as st
import png
import cv2 
from skimage.color import rgb2ypbpr, ypbpr2rgb
ch_dim=1
dtype = tf.float32

def apply_std(ycc, std):
    ycc_tmp = np.transpose(ycc[0,:,:,:],[1,2,0])
    ycc_tmp = ycc_tmp*std
    #ycbcr = rgb2ypbpr(ycc_tmp)*std
    return ypbpr2rgb(ycc_tmp*255.0)


def wpng(fname, img, nY=240, nX=240):
    #f = open(fname,'wb')
    #w = png.Writer(nY,nX)
    img = np.clip(img,0,255)
    img = np.concatenate([img[:,:,2,np.newaxis],img[:,:,1,np.newaxis],img[:,:,0,np.newaxis]],axis=2)
    cv2.imwrite(fname,img)
    #w.write(f,img)
    #f.close()

def tf_YC2R(Y,CbCr):
    YCC = tf.concat([Y,CbCr],axis=ch_dim)
    return tf.transpose(tf.image.yuv_to_rgb(tf.transpose(YCC,[0,2,3,1])),[0,3,1,2])

def tf_Y2R(YCC):
    return tf.transpose(tf.image.yuv_to_rgb(tf.transpose(YCC,[0,2,3,1])),[0,3,1,2])


def myNumExt(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)

def ri2ssos(inp):
    st()
    sz   = inp.shape
    nCh  = int(int(sz[3])/2)
    if nCh == 1:
        out  = tf.sqrt(tf.square(inp[:,:,:,0:nCh])+tf.square(inp[:,:,:,nCh:]))
        return out
    else:
        st()

