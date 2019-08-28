import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

def C(x, ch_out, name,k=3,s=1, reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(k,k), strides=(s,s), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv")))


def Conv2d(x, ch_out, name, reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv")))
def BN(x, is_Training, name):
    scope=name+'_bn'
    return tf.cond(is_Training, lambda: li.batch_norm(x, is_training=True, epsilon=0.000001, center=True, data_format=d_form_, updates_collections=None, scope=scope),
            lambda: li.batch_norm(x, is_training=False, updates_collections=None, epsilon=0.000001, center=True, data_format=d_form_,scope=scope, reuse=True) )

def IN(x, name):
    return tf.contrib.layers.instance_norm( x, epsilon=0.000001, center=True, data_format=d_form_, scope=name+'_in')

def Pool2d(x, ch_out, name):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(2,2), strides=(2,2), padding="SAME", data_format=d_form,use_bias=False, kernel_initializer=li.xavier_initializer(), name=name)

def Pool2d4x4(x, ch_out, _name):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(4,4), strides=(4,4), padding="SAME", data_format=d_form,use_bias=False, kernel_initializer=li.xavier_initializer(), name=_name)

def Conv2dT(x, ch_out, name,k=2,s=2):
    return tf.layers.conv2d_transpose(x, filters=ch_out, kernel_size=(k,k), strides=(s,s), padding="SAME",data_format=d_form,kernel_initializer=li.xavier_initializer(), name=name)

def ReLU(x,name):
    return tf.nn.relu(x, name="".join((name,"_R")))

def lReLU(x,name):
    return tf.nn.leaky_relu(x, name="".join((name,"_lR")))

def Conv1x1(x, ch_out, name, reg=None,use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(1,1), strides=(1,1), padding="SAME", use_bias=use_bias, data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv1x1")))

def Conv2d2x2(x, ch_out, name, kernel_size=3,reg=None, use_bias=False):
    return tf.layers.conv2d(x, filters=ch_out, kernel_size=(kernel_size,kernel_size), strides=(2,2), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv2x2")))

def CNR(inp, n_out, k=3, s=1, _name='', reg=None, use_bias=False):
    return lReLU( IN( C( inp, n_out, _name, k=k, s=s,reg=reg,use_bias=use_bias), _name),_name)

def CNRCN(inp, n_out, k=3, s=1, _name='', reg=None, use_bias=False):
    CNR = lReLU( IN( C( inp, n_out, _name, k=k, s=s,reg=reg,use_bias=use_bias), _name),_name)
    _name = _name+'2'
    return IN( C( CNR, n_out, _name, k=k, s=s,  reg=reg, use_bias=use_bias), _name)


def CBR(inp, n_out, is_Training, name='', reg=[], _1x1Conv=False):
    if _1x1Conv:
        return lReLU( BN( Conv1x1( inp, n_out, name, reg=reg),is_Training,name),name)
    else:
        return lReLU( BN( Conv2d( inp, n_out, name, reg=reg),is_Training,name),name)

def CCBR(inp, n_out, is_Training, name='', reg=[]):
    C1 = Conv1x1( inp, int(n_out/2), name, reg=reg)
    C3 = Conv2d( inp, int(n_out/2), name, reg=reg)
    CC  = tf.concat([C1, C3], axis=ch_dim)
    return lReLU( BN( CC,is_Training,name),name)


def tmpnet(inp, n_out, is_Training, reg_, nCh=64, name_='', _1x1Conv=False):
    return  Conv1x1( inp, n_out, name=name_+'out',  reg=reg_)


def StarG(inp, n_out, is_Training, reg_, nCh=64, name_='', _1x1Conv=False):
    '''downsample starts here'''
    down_1 = CNR(   inp, nCh  , k=7, s=1, _name='down1')
    down_2 = CNR(down_1, nCh*2, k=4, s=2, _name='down2')
    down_3 = CNR(down_2, nCh*4, k=4, s=2, _name='down3')
    '''res-block starts here '''
    res_1 =  down_3 + CNRCN( down_3, nCh*4,_name='res1')
    res_2 =   res_1 + CNRCN(  res_1, nCh*4,_name='res2')
    res_3 =   res_2 + CNRCN(  res_2, nCh*4,_name='res3')
    res_4 =   res_3 + CNRCN(  res_3, nCh*4,_name='res4')
    res_5 =   res_4 + CNRCN(  res_4, nCh*4,_name='res5')
    res_6 =   res_5 + CNRCN(  res_5, nCh*4,_name='res6')
    ''' upsample starts here '''
    up_1  = lReLU( IN( Conv2dT(res_6, nCh*2,name='up1', k=4,s=2), 'up1'), 'up1')
    up_2  = lReLU( IN( Conv2dT( up_1, nCh  ,name='up2', k=4,s=2), 'up2'), 'up2')
    return C(up_2, n_out, 'out',k=7,s=1, reg=reg_, use_bias=False)

def UnetINShallow(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CNR(   inp,   nCh, _name=name_+'lv0_1', reg=reg_)
    down0_2     =    CNR(  down0_1,  nCh,  _name=name_+'lv0_2', reg=reg_)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CNR(    pool1,  nCh*2, _name=name_+'lv1_1', reg=reg_) 
    down1_2     =    CNR(  down1_1,  nCh*2, _name=name_+'lv1_2', reg=reg_)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CNR(    pool2,  nCh*4, _name=name_+'lv2_1', reg=reg_) 
    down2_2     =    CNR(  down2_1,  nCh*4, _name=name_+'lv2_2', reg=reg_)
  
    pool3       = Pool2d(  down2_2,  nCh*8, name=name_+'lv3_p')
    down3_1     =    CNR(    pool3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    down3_2     =    CNR(  down3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT( down3_2,  nCh*4, name=name_+'lv3__up')
    
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def UnetINDiv3(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,9:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    anCh = nCh/3
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def UnetINShallowDiv4_addResBlock(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,12:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    anCh = nCh/4
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
   
    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')


    ''' precossing'''
    pool2        = tf.concat( [apool2, bpool2, cpool2, dpool2], axis=ch_dim)
    down2_1      = CNR(       pool2, nCh*4, _name=name_+'lv2_1', reg=reg_)
    down2_2      = CNR(     down2_1, nCh*4, _name=name_+'lv2_2', reg=reg_)

    down2_3      = CNR( pool2 + down2_2, nCh*4, _name=name_+'lv2_3', reg=reg_)
    down2_4      = CNR(         down2_3, nCh*4, _name=name_+'lv2_4', reg=reg_)

    down2_5      = CNR(down2_2+ down2_4, nCh*4, _name=name_+'lv2_5', reg=reg_)
    down2_6      = CNR(         down2_5, nCh*4, _name=name_+'lv2_6', reg=reg_)

    down2_7      = CNR(down2_4+ down2_6, nCh*4, _name=name_+'lv2_7', reg=reg_)
    down2_8      = CNR(         down2_7, nCh*4, _name=name_+'lv2_8', reg=reg_)
    
    up2         = Conv2dT(down2_6+down2_8,  nCh*2, name=name_+'lv2__up')

    ''' decoder '''
    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINMultiDiv8(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [  inp[:,12:15,:,:],mask],axis=ch_dim)
    finp = tf.concat( [  inp[:,15:18,:,:],mask],axis=ch_dim)
    ginp = tf.concat( [  inp[:,18:21,:,:],mask],axis=ch_dim)
    ninp = tf.concat( [  inp[:,21:24,:,:],mask],axis=ch_dim)
    anCh = nCh/8
     
    '''a path'''
    adown1x1_1    = CNR(            ainp,    anCh, _name=name_+'adown1x1_1', reg=reg_)
    adown1x1_2    = CNR(            adown1x1_1, anCh, _name=name_+'adown1x1_2', reg=reg_)
    adown1x1_3    = CNR(            adown1x1_2, anCh, _name=name_+'adown1x1_3', reg=reg_)
    aout1x1       =     adown1x1_1+ adown1x1_3   
    # 4x4
    ainp4x4       = Pool2d4x4(    ainp, anCh*4, _name=name_+'ainp4x4')
    adown4x4_1    = CNR(       ainp4x4, anCh*4, _name=name_+'adown4x4_1', reg=reg_)
    adown4x4_2    = CNR(    adown4x4_1, anCh*4, _name=name_+'adown4x4_2', reg=reg_)
    adown4x4_3    = CNR(    adown4x4_2, anCh*4, _name=name_+'adown4x4_3', reg=reg_)
    aout4x4       = adown4x4_1 + adown4x4_3
    # 16x16
    ainp16x16   = Pool2d4x4(   ainp4x4, anCh*16, _name=name_+'ainp16x16')
    adown16x16_1  = CNR(     ainp16x16, anCh*16, _name=name_+'adown16x16_1', reg=reg_)
    adown16x16_2  = CNR(  adown16x16_1, anCh*16, _name=name_+'adown16x16_2', reg=reg_)
    adown16x16_3  = CNR(  adown16x16_2, anCh*16, _name=name_+'adown16x16_3', reg=reg_)
    aout16x16     = adown16x16_1 + adown16x16_3

    '''b path'''
    bdown1x1_1    = CNR(            binp,       anCh, _name=name_+'bdown1x1_1', reg=reg_)
    bdown1x1_2    = CNR(            bdown1x1_1, anCh, _name=name_+'bdown1x1_2', reg=reg_)
    bdown1x1_3    = CNR(            bdown1x1_2, anCh, _name=name_+'bdown1x1_3', reg=reg_)
    bout1x1       =     bdown1x1_1+ bdown1x1_3
    # 4x4
    binp4x4       = Pool2d4x4(      binp,       anCh*4, _name=name_+'binp4x4')
    bdown4x4_1    = CNR(            binp4x4,    anCh*4, _name=name_+'bdown4x4_1', reg=reg_)
    bdown4x4_2    = CNR(            bdown4x4_1, anCh*4, _name=name_+'bdown4x4_2', reg=reg_)
    bdown4x4_3    = CNR(            bdown4x4_2, anCh*4, _name=name_+'bdown4x4_3', reg=reg_)
    bout4x4       =     bdown4x4_1+ bdown4x4_3
    # 16x16
    binp16x16   = Pool2d4x4(           binp4x4,      anCh*16, _name=name_+'binp16x16')
    bdown16x16_1  = CNR(               binp16x16,    anCh*16, _name=name_+'bdown16x16_1', reg=reg_)
    bdown16x16_2  = CNR(               bdown16x16_1, anCh*16, _name=name_+'bdown16x16_2', reg=reg_)
    bdown16x16_3  = CNR(               bdown16x16_2, anCh*16, _name=name_+'bdown16x16_3', reg=reg_)
    bout16x16     =      bdown16x16_1+ bdown16x16_3

    '''c path'''
    cdown1x1_1    = CNR(            cinp,       anCh, _name=name_+'cdown1x1_1', reg=reg_)
    cdown1x1_2    = CNR(            cdown1x1_1, anCh, _name=name_+'cdown1x1_2', reg=reg_)
    cdown1x1_3    = CNR(            cdown1x1_2, anCh, _name=name_+'cdown1x1_3', reg=reg_)
    cout1x1       =     cdown1x1_1+ cdown1x1_3
    # 4x4
    cinp4x4       = Pool2d4x4(      cinp,       anCh*4, _name=name_+'cinp4x4')
    cdown4x4_1    = CNR(            cinp4x4,    anCh*4, _name=name_+'cdown4x4_1', reg=reg_)
    cdown4x4_2    = CNR(            cdown4x4_1, anCh*4, _name=name_+'cdown4x4_2', reg=reg_)
    cdown4x4_3    = CNR(            cdown4x4_2, anCh*4, _name=name_+'cdown4x4_3', reg=reg_)
    cout4x4       =     cdown4x4_1+ cdown4x4_3
    # 16x16
    cinp16x16   = Pool2d4x4(           cinp4x4,      anCh*16, _name=name_+'cinp16x16')
    cdown16x16_1  = CNR(               cinp16x16,    anCh*16, _name=name_+'cdown16x16_1', reg=reg_)
    cdown16x16_2  = CNR(               cdown16x16_1, anCh*16, _name=name_+'cdown16x16_2', reg=reg_)
    cdown16x16_3  = CNR(               cdown16x16_2, anCh*16, _name=name_+'cdown16x16_3', reg=reg_)
    cout16x16     =      cdown16x16_1+ cdown16x16_3

    '''d path'''
    ddown1x1_1    = CNR(            dinp,       anCh, _name=name_+'ddown1x1_1', reg=reg_)
    ddown1x1_2    = CNR(            ddown1x1_1, anCh, _name=name_+'ddown1x1_2', reg=reg_)
    ddown1x1_3    = CNR(            ddown1x1_2, anCh, _name=name_+'ddown1x1_3', reg=reg_)
    dout1x1       =     ddown1x1_1+ ddown1x1_3
    # 4x4
    dinp4x4       = Pool2d4x4(      dinp,       anCh*4, _name=name_+'dinp4x4')
    ddown4x4_1    = CNR(            dinp4x4,    anCh*4, _name=name_+'ddown4x4_1', reg=reg_)
    ddown4x4_2    = CNR(            ddown4x4_1, anCh*4, _name=name_+'ddown4x4_2', reg=reg_)
    ddown4x4_3    = CNR(            ddown4x4_2, anCh*4, _name=name_+'ddown4x4_3', reg=reg_)
    dout4x4       =     ddown4x4_1+ ddown4x4_3
    # 16x16
    dinp16x16   = Pool2d4x4(           dinp4x4,      anCh*16, _name=name_+'dinp16x16')
    ddown16x16_1  = CNR(               dinp16x16,    anCh*16, _name=name_+'ddown16x16_1', reg=reg_)
    ddown16x16_2  = CNR(               ddown16x16_1, anCh*16, _name=name_+'ddown16x16_2', reg=reg_)
    ddown16x16_3  = CNR(               ddown16x16_2, anCh*16, _name=name_+'ddown16x16_3', reg=reg_)
    dout16x16     =      ddown16x16_1+ ddown16x16_3

    '''e path'''
    edown1x1_1    = CNR(            einp,       anCh, _name=name_+'edown1x1_1', reg=reg_)
    edown1x1_2    = CNR(            edown1x1_1, anCh, _name=name_+'edown1x1_2', reg=reg_)
    edown1x1_3    = CNR(            edown1x1_2, anCh, _name=name_+'edown1x1_3', reg=reg_)
    eout1x1       =     edown1x1_1+ edown1x1_3
    # 4x4
    einp4x4       = Pool2d4x4(      einp,       anCh*4, _name=name_+'einp4x4')
    edown4x4_1    = CNR(            einp4x4,    anCh*4, _name=name_+'edown4x4_1', reg=reg_)
    edown4x4_2    = CNR(            edown4x4_1, anCh*4, _name=name_+'edown4x4_2', reg=reg_)
    edown4x4_3    = CNR(            edown4x4_2, anCh*4, _name=name_+'edown4x4_3', reg=reg_)
    eout4x4       =     edown4x4_1+ edown4x4_3
    # 16x16
    einp16x16   = Pool2d4x4(           einp4x4,      anCh*16, _name=name_+'einp16x16')
    edown16x16_1  = CNR(               einp16x16,    anCh*16, _name=name_+'edown16x16_1', reg=reg_)
    edown16x16_2  = CNR(               edown16x16_1, anCh*16, _name=name_+'edown16x16_2', reg=reg_)
    edown16x16_3  = CNR(               edown16x16_2, anCh*16, _name=name_+'edown16x16_3', reg=reg_)
    eout16x16     =      edown16x16_1+ edown16x16_3

    '''f path'''
    fdown1x1_1    = CNR(            finp,       anCh, _name=name_+'fdown1x1_1', reg=reg_)
    fdown1x1_2    = CNR(            fdown1x1_1, anCh, _name=name_+'fdown1x1_2', reg=reg_)
    fdown1x1_3    = CNR(            fdown1x1_2, anCh, _name=name_+'fdown1x1_3', reg=reg_)
    fout1x1       =     fdown1x1_1+ fdown1x1_3
    # 4x4
    finp4x4       = Pool2d4x4(      finp,       anCh*4, _name=name_+'finp4x4')
    fdown4x4_1    = CNR(            finp4x4,    anCh*4, _name=name_+'fdown4x4_1', reg=reg_)
    fdown4x4_2    = CNR(            fdown4x4_1, anCh*4, _name=name_+'fdown4x4_2', reg=reg_)
    fdown4x4_3    = CNR(            fdown4x4_2, anCh*4, _name=name_+'fdown4x4_3', reg=reg_)
    fout4x4       =     fdown4x4_1+ fdown4x4_3
    # 16x16
    finp16x16   = Pool2d4x4(           finp4x4,      anCh*16, _name=name_+'finp16x16')
    fdown16x16_1  = CNR(               finp16x16,    anCh*16, _name=name_+'fdown16x16_1', reg=reg_)
    fdown16x16_2  = CNR(               fdown16x16_1, anCh*16, _name=name_+'fdown16x16_2', reg=reg_)
    fdown16x16_3  = CNR(               fdown16x16_2, anCh*16, _name=name_+'fdown16x16_3', reg=reg_)
    fout16x16     =      fdown16x16_1+ fdown16x16_3

    '''g path'''
    gdown1x1_1    = CNR(            ginp,       anCh, _name=name_+'gdown1x1_1', reg=reg_)
    gdown1x1_2    = CNR(            gdown1x1_1, anCh, _name=name_+'gdown1x1_2', reg=reg_)
    gdown1x1_3    = CNR(            gdown1x1_2, anCh, _name=name_+'gdown1x1_3', reg=reg_)
    gout1x1       =     gdown1x1_1+ gdown1x1_3
    # 4x4
    ginp4x4       = Pool2d4x4(      ginp,       anCh*4, _name=name_+'ginp4x4')
    gdown4x4_1    = CNR(            ginp4x4,    anCh*4, _name=name_+'gdown4x4_1', reg=reg_)
    gdown4x4_2    = CNR(            gdown4x4_1, anCh*4, _name=name_+'gdown4x4_2', reg=reg_)
    gdown4x4_3    = CNR(            gdown4x4_2, anCh*4, _name=name_+'gdown4x4_3', reg=reg_)
    gout4x4       =     gdown4x4_1+ gdown4x4_3
    # 16x16
    ginp16x16   = Pool2d4x4(           ginp4x4,      anCh*16, _name=name_+'ginp16x16')
    gdown16x16_1  = CNR(               ginp16x16,    anCh*16, _name=name_+'gdown16x16_1', reg=reg_)
    gdown16x16_2  = CNR(               gdown16x16_1, anCh*16, _name=name_+'gdown16x16_2', reg=reg_)
    gdown16x16_3  = CNR(               gdown16x16_2, anCh*16, _name=name_+'gdown16x16_3', reg=reg_)
    gout16x16     =      gdown16x16_1+ gdown16x16_3

    '''n path'''
    ndown1x1_1    = CNR(            ninp,       anCh, _name=name_+'ndown1x1_1', reg=reg_)
    ndown1x1_2    = CNR(            ndown1x1_1, anCh, _name=name_+'ndown1x1_2', reg=reg_)
    ndown1x1_3    = CNR(            ndown1x1_2, anCh, _name=name_+'ndown1x1_3', reg=reg_)
    nout1x1       =     ndown1x1_1+ ndown1x1_3
    # 4x4
    ninp4x4       = Pool2d4x4(      ninp,       anCh*4, _name=name_+'ninp4x4')
    ndown4x4_1    = CNR(            ninp4x4,    anCh*4, _name=name_+'ndown4x4_1', reg=reg_)
    ndown4x4_2    = CNR(            ndown4x4_1, anCh*4, _name=name_+'ndown4x4_2', reg=reg_)
    ndown4x4_3    = CNR(            ndown4x4_2, anCh*4, _name=name_+'ndown4x4_3', reg=reg_)
    nout4x4       =     ndown4x4_1+ ndown4x4_3
    # 16x16
    ninp16x16   = Pool2d4x4(           ninp4x4,      anCh*16, _name=name_+'ninp16x16')
    ndown16x16_1  = CNR(               ninp16x16,    anCh*16, _name=name_+'ndown16x16_1', reg=reg_)
    ndown16x16_2  = CNR(               ndown16x16_1, anCh*16, _name=name_+'ndown16x16_2', reg=reg_)
    ndown16x16_3  = CNR(               ndown16x16_2, anCh*16, _name=name_+'ndown16x16_3', reg=reg_)
    nout16x16     =      ndown16x16_1+ ndown16x16_3


    ''' decoder 16x16'''
    pool16x16    = tf.concat([ aout16x16, bout16x16, cout16x16, dout16x16, eout16x16, fout16x16, gout16x16, nout16x16], axis=ch_dim)
    proc16x16_1  = CNR(               pool16x16,    nCh*16, _name=name_+'proc16x16_1', reg=reg_)
    proc16x16_2  = CNR(               proc16x16_1,  nCh*16, _name=name_+'proc16x16_2', reg=reg_)
    proc16x16_3  = CNR(               proc16x16_2,  nCh*16, _name=name_+'proc16x16_3', reg=reg_)
    proc16x16_4  = CNR( proc16x16_1+  proc16x16_3,  nCh*16, _name=name_+'proc16x16_4', reg=reg_)
    proc16x16_5  = CNR(               proc16x16_4,  nCh*16, _name=name_+'proc16x16_5', reg=reg_)
    proc16x16_6  = CNR( proc16x16_3+  proc16x16_5,  nCh*16, _name=name_+'proc16x16_6', reg=reg_)
    proc16x16_7  = CNR(               proc16x16_6,  nCh*16, _name=name_+'proc16x16_7', reg=reg_)
    proc16x16_8  = CNR( proc16x16_5+  proc16x16_7,  nCh*16, _name=name_+'proc16x16_8', reg=reg_)
    out16x16     = Conv2dT(           proc16x16_8,  nCh*4, name=name_+'lv16x16__up', k=4, s=4)

    '''decoder 4x4'''
    pool4x4    = tf.concat([out16x16, aout4x4, bout4x4, cout4x4, dout4x4, eout4x4, fout4x4, gout4x4, nout4x4], axis=ch_dim)
    proc4x4_1  = CNR(               pool4x4,     nCh*4, _name=name_+'proc4x4_1', reg=reg_)
    proc4x4_2  = CNR(               proc4x4_1,   nCh*4, _name=name_+'proc4x4_2', reg=reg_)
    proc4x4_3  = CNR(               proc4x4_2,   nCh*4, _name=name_+'proc4x4_3', reg=reg_)
    proc4x4_4  = CNR( proc4x4_1+    proc4x4_3,   nCh*4, _name=name_+'proc4x4_4', reg=reg_)
    proc4x4_5  = CNR(               proc4x4_4,   nCh*4, _name=name_+'proc4x4_5', reg=reg_)
    proc4x4_6  = CNR( proc4x4_3+    proc4x4_5,   nCh*4, _name=name_+'proc4x4_6', reg=reg_)
    proc4x4_7  = CNR(               proc4x4_6,   nCh*4, _name=name_+'proc4x4_7', reg=reg_)
    proc4x4_8  = CNR( proc4x4_5+    proc4x4_7,   nCh*4, _name=name_+'proc4x4_8', reg=reg_)
    out4x4     = Conv2dT(           proc4x4_8,   nCh, name=name_+'lv4x4__up', k=4, s=4)
    
    '''decoder 1x1'''
    pool1x1    = tf.concat([out4x4, aout1x1, bout1x1, cout1x1, dout1x1,eout1x1, fout1x1, gout1x1,nout1x1], axis=ch_dim)
    proc1x1_1  = CNR(               pool1x1,     nCh, _name=name_+'proc1x1_1', reg=reg_)
    proc1x1_2  = CNR(               proc1x1_1,   nCh, _name=name_+'proc1x1_2', reg=reg_)
    proc1x1_3  = CNR(               proc1x1_2,   nCh, _name=name_+'proc1x1_3', reg=reg_)
    proc1x1_4  = CNR( proc1x1_1+    proc1x1_3,   nCh, _name=name_+'proc1x1_4', reg=reg_)
    proc1x1_5  = CNR(               proc1x1_4,   nCh, _name=name_+'proc1x1_5', reg=reg_)
    proc1x1_6  = CNR( proc1x1_3+    proc1x1_5,   nCh, _name=name_+'proc1x1_6', reg=reg_)
    proc1x1_7  = CNR(               proc1x1_6,   nCh, _name=name_+'proc1x1_7', reg=reg_)
    proc1x1_8  = CNR( proc1x1_5+    proc1x1_7,   nCh, _name=name_+'proc1x1_8', reg=reg_)
    return       Conv1x1(           proc1x1_8, n_out,  name=name_+'conv1x1')



def UnetINMultiDiv5(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [  inp[:,12:15,:,:],mask],axis=ch_dim)
    anCh = nCh/5
     
    '''a path'''
    adown1x1_1    = CNR(            ainp,    anCh, _name=name_+'adown1x1_1', reg=reg_)
    adown1x1_2    = CNR(            adown1x1_1, anCh, _name=name_+'adown1x1_2', reg=reg_)
    adown1x1_3    = CNR(            adown1x1_2, anCh, _name=name_+'adown1x1_3', reg=reg_)
    aout1x1       =     adown1x1_1+ adown1x1_3   
    # 4x4
    ainp4x4       = Pool2d4x4(    ainp, anCh*4, _name=name_+'ainp4x4')
    adown4x4_1    = CNR(       ainp4x4, anCh*4, _name=name_+'adown4x4_1', reg=reg_)
    adown4x4_2    = CNR(    adown4x4_1, anCh*4, _name=name_+'adown4x4_2', reg=reg_)
    adown4x4_3    = CNR(    adown4x4_2, anCh*4, _name=name_+'adown4x4_3', reg=reg_)
    aout4x4       = adown4x4_1 + adown4x4_3
    # 16x16
    ainp16x16   = Pool2d4x4(   ainp4x4, anCh*16, _name=name_+'ainp16x16')
    adown16x16_1  = CNR(     ainp16x16, anCh*16, _name=name_+'adown16x16_1', reg=reg_)
    adown16x16_2  = CNR(  adown16x16_1, anCh*16, _name=name_+'adown16x16_2', reg=reg_)
    adown16x16_3  = CNR(  adown16x16_2, anCh*16, _name=name_+'adown16x16_3', reg=reg_)
    aout16x16     = adown16x16_1 + adown16x16_3

    '''b path'''
    bdown1x1_1    = CNR(            binp,       anCh, _name=name_+'bdown1x1_1', reg=reg_)
    bdown1x1_2    = CNR(            bdown1x1_1, anCh, _name=name_+'bdown1x1_2', reg=reg_)
    bdown1x1_3    = CNR(            bdown1x1_2, anCh, _name=name_+'bdown1x1_3', reg=reg_)
    bout1x1       =     bdown1x1_1+ bdown1x1_3
    # 4x4
    binp4x4       = Pool2d4x4(      binp,       anCh*4, _name=name_+'binp4x4')
    bdown4x4_1    = CNR(            binp4x4,    anCh*4, _name=name_+'bdown4x4_1', reg=reg_)
    bdown4x4_2    = CNR(            bdown4x4_1, anCh*4, _name=name_+'bdown4x4_2', reg=reg_)
    bdown4x4_3    = CNR(            bdown4x4_2, anCh*4, _name=name_+'bdown4x4_3', reg=reg_)
    bout4x4       =     bdown4x4_1+ bdown4x4_3
    # 16x16
    binp16x16   = Pool2d4x4(           binp4x4,      anCh*16, _name=name_+'binp16x16')
    bdown16x16_1  = CNR(               binp16x16,    anCh*16, _name=name_+'bdown16x16_1', reg=reg_)
    bdown16x16_2  = CNR(               bdown16x16_1, anCh*16, _name=name_+'bdown16x16_2', reg=reg_)
    bdown16x16_3  = CNR(               bdown16x16_2, anCh*16, _name=name_+'bdown16x16_3', reg=reg_)
    bout16x16     =      bdown16x16_1+ bdown16x16_3

    '''c path'''
    cdown1x1_1    = CNR(            cinp,       anCh, _name=name_+'cdown1x1_1', reg=reg_)
    cdown1x1_2    = CNR(            cdown1x1_1, anCh, _name=name_+'cdown1x1_2', reg=reg_)
    cdown1x1_3    = CNR(            cdown1x1_2, anCh, _name=name_+'cdown1x1_3', reg=reg_)
    cout1x1       =     cdown1x1_1+ cdown1x1_3
    # 4x4
    cinp4x4       = Pool2d4x4(      cinp,       anCh*4, _name=name_+'cinp4x4')
    cdown4x4_1    = CNR(            cinp4x4,    anCh*4, _name=name_+'cdown4x4_1', reg=reg_)
    cdown4x4_2    = CNR(            cdown4x4_1, anCh*4, _name=name_+'cdown4x4_2', reg=reg_)
    cdown4x4_3    = CNR(            cdown4x4_2, anCh*4, _name=name_+'cdown4x4_3', reg=reg_)
    cout4x4       =     cdown4x4_1+ cdown4x4_3
    # 16x16
    cinp16x16   = Pool2d4x4(           cinp4x4,      anCh*16, _name=name_+'cinp16x16')
    cdown16x16_1  = CNR(               cinp16x16,    anCh*16, _name=name_+'cdown16x16_1', reg=reg_)
    cdown16x16_2  = CNR(               cdown16x16_1, anCh*16, _name=name_+'cdown16x16_2', reg=reg_)
    cdown16x16_3  = CNR(               cdown16x16_2, anCh*16, _name=name_+'cdown16x16_3', reg=reg_)
    cout16x16     =      cdown16x16_1+ cdown16x16_3

    '''d path'''
    ddown1x1_1    = CNR(            dinp,       anCh, _name=name_+'ddown1x1_1', reg=reg_)
    ddown1x1_2    = CNR(            ddown1x1_1, anCh, _name=name_+'ddown1x1_2', reg=reg_)
    ddown1x1_3    = CNR(            ddown1x1_2, anCh, _name=name_+'ddown1x1_3', reg=reg_)
    dout1x1       =     ddown1x1_1+ ddown1x1_3
    # 4x4
    dinp4x4       = Pool2d4x4(      dinp,       anCh*4, _name=name_+'dinp4x4')
    ddown4x4_1    = CNR(            dinp4x4,    anCh*4, _name=name_+'ddown4x4_1', reg=reg_)
    ddown4x4_2    = CNR(            ddown4x4_1, anCh*4, _name=name_+'ddown4x4_2', reg=reg_)
    ddown4x4_3    = CNR(            ddown4x4_2, anCh*4, _name=name_+'ddown4x4_3', reg=reg_)
    dout4x4       =     ddown4x4_1+ ddown4x4_3
    # 16x16
    dinp16x16   = Pool2d4x4(           dinp4x4,      anCh*16, _name=name_+'dinp16x16')
    ddown16x16_1  = CNR(               dinp16x16,    anCh*16, _name=name_+'ddown16x16_1', reg=reg_)
    ddown16x16_2  = CNR(               ddown16x16_1, anCh*16, _name=name_+'ddown16x16_2', reg=reg_)
    ddown16x16_3  = CNR(               ddown16x16_2, anCh*16, _name=name_+'ddown16x16_3', reg=reg_)
    dout16x16     =      ddown16x16_1+ ddown16x16_3

    '''e path'''
    edown1x1_1    = CNR(            einp,       anCh, _name=name_+'edown1x1_1', reg=reg_)
    edown1x1_2    = CNR(            edown1x1_1, anCh, _name=name_+'edown1x1_2', reg=reg_)
    edown1x1_3    = CNR(            edown1x1_2, anCh, _name=name_+'edown1x1_3', reg=reg_)
    eout1x1       =     edown1x1_1+ edown1x1_3
    # 4x4
    einp4x4       = Pool2d4x4(      einp,       anCh*4, _name=name_+'einp4x4')
    edown4x4_1    = CNR(            einp4x4,    anCh*4, _name=name_+'edown4x4_1', reg=reg_)
    edown4x4_2    = CNR(            edown4x4_1, anCh*4, _name=name_+'edown4x4_2', reg=reg_)
    edown4x4_3    = CNR(            edown4x4_2, anCh*4, _name=name_+'edown4x4_3', reg=reg_)
    eout4x4       =     edown4x4_1+ edown4x4_3
    # 16x16
    einp16x16   = Pool2d4x4(           einp4x4,      anCh*16, _name=name_+'einp16x16')
    edown16x16_1  = CNR(               einp16x16,    anCh*16, _name=name_+'edown16x16_1', reg=reg_)
    edown16x16_2  = CNR(               edown16x16_1, anCh*16, _name=name_+'edown16x16_2', reg=reg_)
    edown16x16_3  = CNR(               edown16x16_2, anCh*16, _name=name_+'edown16x16_3', reg=reg_)
    eout16x16     =      edown16x16_1+ edown16x16_3

    ''' decoder 16x16'''
    pool16x16    = tf.concat([ aout16x16, bout16x16, cout16x16, dout16x16, eout16x16], axis=ch_dim)
    proc16x16_1  = CNR(               pool16x16,    nCh*16, _name=name_+'proc16x16_1', reg=reg_)
    proc16x16_2  = CNR(               proc16x16_1,  nCh*16, _name=name_+'proc16x16_2', reg=reg_)
    proc16x16_3  = CNR(               proc16x16_2,  nCh*16, _name=name_+'proc16x16_3', reg=reg_)
    proc16x16_4  = CNR( proc16x16_1+  proc16x16_3,  nCh*16, _name=name_+'proc16x16_4', reg=reg_)
    proc16x16_5  = CNR(               proc16x16_4,  nCh*16, _name=name_+'proc16x16_5', reg=reg_)
    proc16x16_6  = CNR( proc16x16_3+  proc16x16_5,  nCh*16, _name=name_+'proc16x16_6', reg=reg_)
    proc16x16_7  = CNR(               proc16x16_6,  nCh*16, _name=name_+'proc16x16_7', reg=reg_)
    proc16x16_8  = CNR( proc16x16_5+  proc16x16_7,  nCh*16, _name=name_+'proc16x16_8', reg=reg_)
    out16x16     = Conv2dT(           proc16x16_8,  nCh*4, name=name_+'lv16x16__up', k=4, s=4)

    '''decoder 4x4'''
    pool4x4    = tf.concat([out16x16, aout4x4, bout4x4, cout4x4, dout4x4, eout4x4], axis=ch_dim)
    proc4x4_1  = CNR(               pool4x4,     nCh*4, _name=name_+'proc4x4_1', reg=reg_)
    proc4x4_2  = CNR(               proc4x4_1,   nCh*4, _name=name_+'proc4x4_2', reg=reg_)
    proc4x4_3  = CNR(               proc4x4_2,   nCh*4, _name=name_+'proc4x4_3', reg=reg_)
    proc4x4_4  = CNR( proc4x4_1+    proc4x4_3,   nCh*4, _name=name_+'proc4x4_4', reg=reg_)
    proc4x4_5  = CNR(               proc4x4_4,   nCh*4, _name=name_+'proc4x4_5', reg=reg_)
    proc4x4_6  = CNR( proc4x4_3+    proc4x4_5,   nCh*4, _name=name_+'proc4x4_6', reg=reg_)
    proc4x4_7  = CNR(               proc4x4_6,   nCh*4, _name=name_+'proc4x4_7', reg=reg_)
    proc4x4_8  = CNR( proc4x4_5+    proc4x4_7,   nCh*4, _name=name_+'proc4x4_8', reg=reg_)
    out4x4     = Conv2dT(           proc4x4_8,   nCh, name=name_+'lv4x4__up', k=4, s=4)
    
    '''decoder 1x1'''
    pool1x1    = tf.concat([out4x4, aout1x1, bout1x1, cout1x1, dout1x1,eout1x1], axis=ch_dim)
    proc1x1_1  = CNR(               pool1x1,     nCh, _name=name_+'proc1x1_1', reg=reg_)
    proc1x1_2  = CNR(               proc1x1_1,   nCh, _name=name_+'proc1x1_2', reg=reg_)
    proc1x1_3  = CNR(               proc1x1_2,   nCh, _name=name_+'proc1x1_3', reg=reg_)
    proc1x1_4  = CNR( proc1x1_1+    proc1x1_3,   nCh, _name=name_+'proc1x1_4', reg=reg_)
    proc1x1_5  = CNR(               proc1x1_4,   nCh, _name=name_+'proc1x1_5', reg=reg_)
    proc1x1_6  = CNR( proc1x1_3+    proc1x1_5,   nCh, _name=name_+'proc1x1_6', reg=reg_)
    proc1x1_7  = CNR(               proc1x1_6,   nCh, _name=name_+'proc1x1_7', reg=reg_)
    proc1x1_8  = CNR( proc1x1_5+    proc1x1_7,   nCh, _name=name_+'proc1x1_8', reg=reg_)
    return       Conv1x1(           proc1x1_8, n_out,  name=name_+'conv1x1')



def UnetINMultiDiv4(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,12:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    anCh = nCh/4
     
    '''a path'''
    adown1x1_1    = CNR(            ainp,    anCh, _name=name_+'adown1x1_1', reg=reg_)
    adown1x1_2    = CNR(            adown1x1_1, anCh, _name=name_+'adown1x1_2', reg=reg_)
    adown1x1_3    = CNR(            adown1x1_2, anCh, _name=name_+'adown1x1_3', reg=reg_)
    aout1x1       =     adown1x1_1+ adown1x1_3   
    # 4x4
    ainp4x4       = Pool2d4x4(    ainp, anCh*4, _name=name_+'ainp4x4')
    adown4x4_1    = CNR(       ainp4x4, anCh*4, _name=name_+'adown4x4_1', reg=reg_)
    adown4x4_2    = CNR(    adown4x4_1, anCh*4, _name=name_+'adown4x4_2', reg=reg_)
    adown4x4_3    = CNR(    adown4x4_2, anCh*4, _name=name_+'adown4x4_3', reg=reg_)
    aout4x4       = adown4x4_1 + adown4x4_3
    # 16x16
    ainp16x16   = Pool2d4x4(   ainp4x4, anCh*16, _name=name_+'ainp16x16')
    adown16x16_1  = CNR(     ainp16x16, anCh*16, _name=name_+'adown16x16_1', reg=reg_)
    adown16x16_2  = CNR(  adown16x16_1, anCh*16, _name=name_+'adown16x16_2', reg=reg_)
    adown16x16_3  = CNR(  adown16x16_2, anCh*16, _name=name_+'adown16x16_3', reg=reg_)
    aout16x16     = adown16x16_1 + adown16x16_3

    '''b path'''
    bdown1x1_1    = CNR(            binp,       anCh, _name=name_+'bdown1x1_1', reg=reg_)
    bdown1x1_2    = CNR(            bdown1x1_1, anCh, _name=name_+'bdown1x1_2', reg=reg_)
    bdown1x1_3    = CNR(            bdown1x1_2, anCh, _name=name_+'bdown1x1_3', reg=reg_)
    bout1x1       =     bdown1x1_1+ bdown1x1_3
    # 4x4
    binp4x4       = Pool2d4x4(      binp,       anCh*4, _name=name_+'binp4x4')
    bdown4x4_1    = CNR(            binp4x4,    anCh*4, _name=name_+'bdown4x4_1', reg=reg_)
    bdown4x4_2    = CNR(            bdown4x4_1, anCh*4, _name=name_+'bdown4x4_2', reg=reg_)
    bdown4x4_3    = CNR(            bdown4x4_2, anCh*4, _name=name_+'bdown4x4_3', reg=reg_)
    bout4x4       =     bdown4x4_1+ bdown4x4_3
    # 16x16
    binp16x16   = Pool2d4x4(           binp4x4,      anCh*16, _name=name_+'binp16x16')
    bdown16x16_1  = CNR(               binp16x16,    anCh*16, _name=name_+'bdown16x16_1', reg=reg_)
    bdown16x16_2  = CNR(               bdown16x16_1, anCh*16, _name=name_+'bdown16x16_2', reg=reg_)
    bdown16x16_3  = CNR(               bdown16x16_2, anCh*16, _name=name_+'bdown16x16_3', reg=reg_)
    bout16x16     =      bdown16x16_1+ bdown16x16_3

    '''c path'''
    cdown1x1_1    = CNR(            cinp,       anCh, _name=name_+'cdown1x1_1', reg=reg_)
    cdown1x1_2    = CNR(            cdown1x1_1, anCh, _name=name_+'cdown1x1_2', reg=reg_)
    cdown1x1_3    = CNR(            cdown1x1_2, anCh, _name=name_+'cdown1x1_3', reg=reg_)
    cout1x1       =     cdown1x1_1+ cdown1x1_3
    # 4x4
    cinp4x4       = Pool2d4x4(      cinp,       anCh*4, _name=name_+'cinp4x4')
    cdown4x4_1    = CNR(            cinp4x4,    anCh*4, _name=name_+'cdown4x4_1', reg=reg_)
    cdown4x4_2    = CNR(            cdown4x4_1, anCh*4, _name=name_+'cdown4x4_2', reg=reg_)
    cdown4x4_3    = CNR(            cdown4x4_2, anCh*4, _name=name_+'cdown4x4_3', reg=reg_)
    cout4x4       =     cdown4x4_1+ cdown4x4_3
    # 16x16
    cinp16x16   = Pool2d4x4(           cinp4x4,      anCh*16, _name=name_+'cinp16x16')
    cdown16x16_1  = CNR(               cinp16x16,    anCh*16, _name=name_+'cdown16x16_1', reg=reg_)
    cdown16x16_2  = CNR(               cdown16x16_1, anCh*16, _name=name_+'cdown16x16_2', reg=reg_)
    cdown16x16_3  = CNR(               cdown16x16_2, anCh*16, _name=name_+'cdown16x16_3', reg=reg_)
    cout16x16     =      cdown16x16_1+ cdown16x16_3

    '''d path'''
    ddown1x1_1    = CNR(            dinp,       anCh, _name=name_+'ddown1x1_1', reg=reg_)
    ddown1x1_2    = CNR(            ddown1x1_1, anCh, _name=name_+'ddown1x1_2', reg=reg_)
    ddown1x1_3    = CNR(            ddown1x1_2, anCh, _name=name_+'ddown1x1_3', reg=reg_)
    dout1x1       =     ddown1x1_1+ ddown1x1_3
    # 4x4
    dinp4x4       = Pool2d4x4(      dinp,       anCh*4, _name=name_+'dinp4x4')
    ddown4x4_1    = CNR(            dinp4x4,    anCh*4, _name=name_+'ddown4x4_1', reg=reg_)
    ddown4x4_2    = CNR(            ddown4x4_1, anCh*4, _name=name_+'ddown4x4_2', reg=reg_)
    ddown4x4_3    = CNR(            ddown4x4_2, anCh*4, _name=name_+'ddown4x4_3', reg=reg_)
    dout4x4       =     ddown4x4_1+ ddown4x4_3
    # 16x16
    dinp16x16   = Pool2d4x4(           dinp4x4,      anCh*16, _name=name_+'dinp16x16')
    ddown16x16_1  = CNR(               dinp16x16,    anCh*16, _name=name_+'ddown16x16_1', reg=reg_)
    ddown16x16_2  = CNR(               ddown16x16_1, anCh*16, _name=name_+'ddown16x16_2', reg=reg_)
    ddown16x16_3  = CNR(               ddown16x16_2, anCh*16, _name=name_+'ddown16x16_3', reg=reg_)
    dout16x16     =      ddown16x16_1+ ddown16x16_3

    ''' decoder 16x16'''
    pool16x16    = tf.concat([ aout16x16, bout16x16, cout16x16, dout16x16], axis=ch_dim)
    proc16x16_1  = CNR(               pool16x16,    nCh*16, _name=name_+'proc16x16_1', reg=reg_)
    proc16x16_2  = CNR(               proc16x16_1,  nCh*16, _name=name_+'proc16x16_2', reg=reg_)
    proc16x16_3  = CNR(               proc16x16_2,  nCh*16, _name=name_+'proc16x16_3', reg=reg_)
    proc16x16_4  = CNR( proc16x16_1+  proc16x16_3,  nCh*16, _name=name_+'proc16x16_4', reg=reg_)
    proc16x16_5  = CNR(               proc16x16_4,  nCh*16, _name=name_+'proc16x16_5', reg=reg_)
    proc16x16_6  = CNR( proc16x16_3+  proc16x16_5,  nCh*16, _name=name_+'proc16x16_6', reg=reg_)
    proc16x16_7  = CNR(               proc16x16_6,  nCh*16, _name=name_+'proc16x16_7', reg=reg_)
    proc16x16_8  = CNR( proc16x16_5+  proc16x16_7,  nCh*16, _name=name_+'proc16x16_8', reg=reg_)
    out16x16     = Conv2dT(           proc16x16_8,  nCh*4, name=name_+'lv16x16__up', k=4, s=4)

    '''decoder 4x4'''
    pool4x4    = tf.concat([out16x16, aout4x4, bout4x4, cout4x4, dout4x4], axis=ch_dim)
    proc4x4_1  = CNR(               pool4x4,     nCh*4, _name=name_+'proc4x4_1', reg=reg_)
    proc4x4_2  = CNR(               proc4x4_1,   nCh*4, _name=name_+'proc4x4_2', reg=reg_)
    proc4x4_3  = CNR(               proc4x4_2,   nCh*4, _name=name_+'proc4x4_3', reg=reg_)
    proc4x4_4  = CNR( proc4x4_1+    proc4x4_3,   nCh*4, _name=name_+'proc4x4_4', reg=reg_)
    proc4x4_5  = CNR(               proc4x4_4,   nCh*4, _name=name_+'proc4x4_5', reg=reg_)
    proc4x4_6  = CNR( proc4x4_3+    proc4x4_5,   nCh*4, _name=name_+'proc4x4_6', reg=reg_)
    proc4x4_7  = CNR(               proc4x4_6,   nCh*4, _name=name_+'proc4x4_7', reg=reg_)
    proc4x4_8  = CNR( proc4x4_5+    proc4x4_7,   nCh*4, _name=name_+'proc4x4_8', reg=reg_)
    out4x4     = Conv2dT(           proc4x4_8,   nCh, name=name_+'lv4x4__up', k=4, s=4)
    
    '''decoder 1x1'''
    pool1x1    = tf.concat([out4x4, aout1x1, bout1x1, cout1x1, dout1x1], axis=ch_dim)
    proc1x1_1  = CNR(               pool1x1,     nCh, _name=name_+'proc1x1_1', reg=reg_)
    proc1x1_2  = CNR(               proc1x1_1,   nCh, _name=name_+'proc1x1_2', reg=reg_)
    proc1x1_3  = CNR(               proc1x1_2,   nCh, _name=name_+'proc1x1_3', reg=reg_)
    proc1x1_4  = CNR( proc1x1_1+    proc1x1_3,   nCh, _name=name_+'proc1x1_4', reg=reg_)
    proc1x1_5  = CNR(               proc1x1_4,   nCh, _name=name_+'proc1x1_5', reg=reg_)
    proc1x1_6  = CNR( proc1x1_3+    proc1x1_5,   nCh, _name=name_+'proc1x1_6', reg=reg_)
    proc1x1_7  = CNR(               proc1x1_6,   nCh, _name=name_+'proc1x1_7', reg=reg_)
    proc1x1_8  = CNR( proc1x1_5+    proc1x1_7,   nCh, _name=name_+'proc1x1_8', reg=reg_)
    return       Conv1x1(           proc1x1_8, n_out,  name=name_+'conv1x1')


def UnetINDiv4_addResBlock(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,12:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    anCh = nCh/4
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')


    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)

    down4_3     =    CNR(down4_2 +  pool4, nCh*16, _name=name_+'lv4_3', reg=reg_) 
    down4_4     =    CNR(         down4_3, nCh*16, _name=name_+'lv4_4', reg=reg_)

    down4_5     =    CNR(down4_4 +down4_2, nCh*16, _name=name_+'lv4_5', reg=reg_) 
    down4_6     =    CNR(         down4_5, nCh*16, _name=name_+'lv4_6', reg=reg_)

    down4_7     =    CNR(down4_6 +down4_4, nCh*16, _name=name_+'lv4_7', reg=reg_) 
    down4_8     =    CNR(         down4_7, nCh*16, _name=name_+'lv4_8', reg=reg_)

    up4         = Conv2dT(down4_8+down4_6,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2, ddown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2, ddown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def UnetINDiv4_addResBlockWOskip(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,12:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    anCh = nCh/4
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')


    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)

    down4_3     =    CNR(down4_2 +  pool4, nCh*16, _name=name_+'lv4_3', reg=reg_) 
    down4_4     =    CNR(         down4_3, nCh*16, _name=name_+'lv4_4', reg=reg_)

    down4_5     =    CNR(down4_4 +down4_2, nCh*16, _name=name_+'lv4_5', reg=reg_) 
    down4_6     =    CNR(         down4_5, nCh*16, _name=name_+'lv4_6', reg=reg_)

    down4_7     =    CNR(down4_6 +down4_4, nCh*16, _name=name_+'lv4_7', reg=reg_) 
    down4_8     =    CNR(         down4_7, nCh*16, _name=name_+'lv4_8', reg=reg_)

    up4         = Conv2dT(down4_8+down4_6,  nCh*8, name=name_+'lv4__up')
    
    #down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2, ddown3_2], axis=ch_dim)
    #CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        up4,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    #down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2, ddown2_2], axis=ch_dim)
    #CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      up3,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    #down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2], axis=ch_dim)
    #CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      up2,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    #down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2], axis=ch_dim)
    #CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      up1,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINDiv4(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,12:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [  inp[:,9:12,:,:],mask],axis=ch_dim)
    anCh = nCh/4
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')


    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2, ddown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2, ddown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINDiv5_addResBlockWOskip(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [ inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [inp[:,12:15,:,:],mask],axis=ch_dim)
    anCh = nCh/5
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')

    '''e path'''
    edown0_1     =    CNR(  einp,      anCh,  _name=name_+'elv0_1', reg=reg_)
    edown0_2     =    CNR(  edown0_1,  anCh,  _name=name_+'elv0_2', reg=reg_)
    epool1       = Pool2d(  edown0_2,  anCh*2, name=name_+'elv1_p') 
    edown1_1     =    CNR(  epool1,    anCh*2,_name=name_+'elv1_1', reg=reg_) 
    edown1_2     =    CNR(  edown1_1,  anCh*2,_name=name_+'elv1_2', reg=reg_)
    epool2       = Pool2d(  edown1_2,  anCh*4, name=name_+'elv2_p')
    edown2_1     =    CNR(  epool2,    anCh*4,_name=name_+'elv2_1', reg=reg_) 
    edown2_2     =    CNR(  edown2_1,  anCh*4,_name=name_+'elv2_2', reg=reg_)
    epool3       = Pool2d(  edown2_2,  anCh*8, name=name_+'elv3_p')
    edown3_1     =    CNR(  epool3,    anCh*8,_name=name_+'elv3_1', reg=reg_) 
    edown3_2     =    CNR(  edown3_1,  anCh*8,_name=name_+'elv3_2', reg=reg_)
    epool4       = Pool2d(  edown3_2,  anCh*16, name=name_+'elv4_p')

    '''decoder'''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4,epool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    down4_3     =    CNR(down4_2 +  pool4, nCh*16, _name=name_+'lv4_3', reg=reg_) 
    down4_4     =    CNR(         down4_3, nCh*16, _name=name_+'lv4_4', reg=reg_)
    down4_5     =    CNR(down4_4 +down4_2, nCh*16, _name=name_+'lv4_5', reg=reg_) 
    down4_6     =    CNR(         down4_5, nCh*16, _name=name_+'lv4_6', reg=reg_)
    down4_7     =    CNR(down4_6 +down4_4, nCh*16, _name=name_+'lv4_7', reg=reg_) 
    down4_8     =    CNR(         down4_7, nCh*16, _name=name_+'lv4_8', reg=reg_)
    up4         = Conv2dT(down4_8+down4_6,  nCh*8, name=name_+'lv4__up')
    
    #down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2,ddown3_2,edown3_2], axis=ch_dim)
    #CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        up4,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    #down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2,ddown2_2,edown2_2], axis=ch_dim)
    #CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      up3,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    #down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2,edown1_2], axis=ch_dim)
    #CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      up2,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    #down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2,edown0_2], axis=ch_dim)
    #CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      up1,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def UnetINDiv5_addResBlock(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [ inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [inp[:,12:15,:,:],mask],axis=ch_dim)
    anCh = nCh/5
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')

    '''e path'''
    edown0_1     =    CNR(  einp,      anCh,  _name=name_+'elv0_1', reg=reg_)
    edown0_2     =    CNR(  edown0_1,  anCh,  _name=name_+'elv0_2', reg=reg_)
    epool1       = Pool2d(  edown0_2,  anCh*2, name=name_+'elv1_p') 
    edown1_1     =    CNR(  epool1,    anCh*2,_name=name_+'elv1_1', reg=reg_) 
    edown1_2     =    CNR(  edown1_1,  anCh*2,_name=name_+'elv1_2', reg=reg_)
    epool2       = Pool2d(  edown1_2,  anCh*4, name=name_+'elv2_p')
    edown2_1     =    CNR(  epool2,    anCh*4,_name=name_+'elv2_1', reg=reg_) 
    edown2_2     =    CNR(  edown2_1,  anCh*4,_name=name_+'elv2_2', reg=reg_)
    epool3       = Pool2d(  edown2_2,  anCh*8, name=name_+'elv3_p')
    edown3_1     =    CNR(  epool3,    anCh*8,_name=name_+'elv3_1', reg=reg_) 
    edown3_2     =    CNR(  edown3_1,  anCh*8,_name=name_+'elv3_2', reg=reg_)
    epool4       = Pool2d(  edown3_2,  anCh*16, name=name_+'elv4_p')

    '''decoder'''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4,epool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    down4_3     =    CNR(down4_2 +  pool4, nCh*16, _name=name_+'lv4_3', reg=reg_) 
    down4_4     =    CNR(         down4_3, nCh*16, _name=name_+'lv4_4', reg=reg_)
    down4_5     =    CNR(down4_4 +down4_2, nCh*16, _name=name_+'lv4_5', reg=reg_) 
    down4_6     =    CNR(         down4_5, nCh*16, _name=name_+'lv4_6', reg=reg_)
    down4_7     =    CNR(down4_6 +down4_4, nCh*16, _name=name_+'lv4_7', reg=reg_) 
    down4_8     =    CNR(         down4_7, nCh*16, _name=name_+'lv4_8', reg=reg_)
    up4         = Conv2dT(down4_8+down4_6,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2,ddown3_2,edown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2,ddown2_2,edown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2,edown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2,edown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINDiv8(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [ inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [inp[:,12:15,:,:],mask],axis=ch_dim)
    finp = tf.concat( [  inp[:,15:18,:,:],mask],axis=ch_dim)
    ginp = tf.concat( [ inp[:,18:21,:,:],mask],axis=ch_dim)
    ninp = tf.concat( [inp[:,21:24,:,:],mask],axis=ch_dim)
    anCh = nCh/8
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')

    '''e path'''
    edown0_1     =    CNR(  einp,      anCh,  _name=name_+'elv0_1', reg=reg_)
    edown0_2     =    CNR(  edown0_1,  anCh,  _name=name_+'elv0_2', reg=reg_)
    epool1       = Pool2d(  edown0_2,  anCh*2, name=name_+'elv1_p') 
    edown1_1     =    CNR(  epool1,    anCh*2,_name=name_+'elv1_1', reg=reg_) 
    edown1_2     =    CNR(  edown1_1,  anCh*2,_name=name_+'elv1_2', reg=reg_)
    epool2       = Pool2d(  edown1_2,  anCh*4, name=name_+'elv2_p')
    edown2_1     =    CNR(  epool2,    anCh*4,_name=name_+'elv2_1', reg=reg_) 
    edown2_2     =    CNR(  edown2_1,  anCh*4,_name=name_+'elv2_2', reg=reg_)
    epool3       = Pool2d(  edown2_2,  anCh*8, name=name_+'elv3_p')
    edown3_1     =    CNR(  epool3,    anCh*8,_name=name_+'elv3_1', reg=reg_) 
    edown3_2     =    CNR(  edown3_1,  anCh*8,_name=name_+'elv3_2', reg=reg_)
    epool4       = Pool2d(  edown3_2,  anCh*16, name=name_+'elv4_p')

    '''f path'''
    fdown0_1     =    CNR(  finp,      anCh,  _name=name_+'flv0_1', reg=reg_)
    fdown0_2     =    CNR(  fdown0_1,  anCh,  _name=name_+'flv0_2', reg=reg_)
    fpool1       = Pool2d(  fdown0_2,  anCh*2, name=name_+'flv1_p') 
    fdown1_1     =    CNR(  fpool1,    anCh*2,_name=name_+'flv1_1', reg=reg_) 
    fdown1_2     =    CNR(  fdown1_1,  anCh*2,_name=name_+'flv1_2', reg=reg_)
    fpool2       = Pool2d(  fdown1_2,  anCh*4, name=name_+'flv2_p')
    fdown2_1     =    CNR(  fpool2,    anCh*4,_name=name_+'flv2_1', reg=reg_) 
    fdown2_2     =    CNR(  fdown2_1,  anCh*4,_name=name_+'flv2_2', reg=reg_)
    fpool3       = Pool2d(  fdown2_2,  anCh*8, name=name_+'flv3_p')
    fdown3_1     =    CNR(  fpool3,    anCh*8,_name=name_+'flv3_1', reg=reg_) 
    fdown3_2     =    CNR(  fdown3_1,  anCh*8,_name=name_+'flv3_2', reg=reg_)
    fpool4       = Pool2d(  fdown3_2,  anCh*16, name=name_+'flv4_p')

    '''g path'''
    gdown0_1     =    CNR(  ginp,      anCh,  _name=name_+'glv0_1', reg=reg_)
    gdown0_2     =    CNR(  gdown0_1,  anCh,  _name=name_+'glv0_2', reg=reg_)
    gpool1       = Pool2d(  gdown0_2,  anCh*2, name=name_+'glv1_p') 
    gdown1_1     =    CNR(  gpool1,    anCh*2,_name=name_+'glv1_1', reg=reg_) 
    gdown1_2     =    CNR(  gdown1_1,  anCh*2,_name=name_+'glv1_2', reg=reg_)
    gpool2       = Pool2d(  gdown1_2,  anCh*4, name=name_+'glv2_p')
    gdown2_1     =    CNR(  gpool2,    anCh*4,_name=name_+'glv2_1', reg=reg_) 
    gdown2_2     =    CNR(  gdown2_1,  anCh*4,_name=name_+'glv2_2', reg=reg_)
    gpool3       = Pool2d(  gdown2_2,  anCh*8, name=name_+'glv3_p')
    gdown3_1     =    CNR(  gpool3,    anCh*8,_name=name_+'glv3_1', reg=reg_) 
    gdown3_2     =    CNR(  gdown3_1,  anCh*8,_name=name_+'glv3_2', reg=reg_)
    gpool4       = Pool2d(  gdown3_2,  anCh*16, name=name_+'glv4_p')

    '''n path'''
    ndown0_1     =    CNR(  ninp,      anCh,  _name=name_+'nlv0_1', reg=reg_)
    ndown0_2     =    CNR(  ndown0_1,  anCh,  _name=name_+'nlv0_2', reg=reg_)
    npool1       = Pool2d(  ndown0_2,  anCh*2, name=name_+'nlv1_p') 
    ndown1_1     =    CNR(  npool1,    anCh*2,_name=name_+'nlv1_1', reg=reg_) 
    ndown1_2     =    CNR(  ndown1_1,  anCh*2,_name=name_+'nlv1_2', reg=reg_)
    npool2       = Pool2d(  ndown1_2,  anCh*4, name=name_+'nlv2_p')
    ndown2_1     =    CNR(  npool2,    anCh*4,_name=name_+'nlv2_1', reg=reg_) 
    ndown2_2     =    CNR(  ndown2_1,  anCh*4,_name=name_+'nlv2_2', reg=reg_)
    npool3       = Pool2d(  ndown2_2,  anCh*8, name=name_+'nlv3_p')
    ndown3_1     =    CNR(  npool3,    anCh*8,_name=name_+'nlv3_1', reg=reg_) 
    ndown3_2     =    CNR(  ndown3_1,  anCh*8,_name=name_+'nlv3_2', reg=reg_)
    npool4       = Pool2d(  ndown3_2,  anCh*16, name=name_+'nlv4_p')

    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4,epool4,fpool4,gpool4,npool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2,ddown3_2,edown3_2,fdown3_2,gdown3_2,ndown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2,ddown2_2,edown2_2,fdown2_2,gdown2_2,ndown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2,edown1_2,fdown1_2,gdown1_2,ndown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2,edown0_2,fdown0_2,gdown0_2,ndown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINDiv5(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [ inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [inp[:,12:15,:,:],mask],axis=ch_dim)
    anCh = nCh/5
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')
    adown3_1     =    CNR(  apool3,    anCh*8,_name=name_+'alv3_1', reg=reg_) 
    adown3_2     =    CNR(  adown3_1,  anCh*8,_name=name_+'alv3_2', reg=reg_)
    apool4       = Pool2d(  adown3_2,  anCh*16, name=name_+'alv4_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')
    bdown3_1     =    CNR(  bpool3,    anCh*8,_name=name_+'blv3_1', reg=reg_) 
    bdown3_2     =    CNR(  bdown3_1,  anCh*8,_name=name_+'blv3_2', reg=reg_)
    bpool4       = Pool2d(  bdown3_2,  anCh*16, name=name_+'blv4_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')
    cdown3_1     =    CNR(  cpool3,    anCh*8,_name=name_+'clv3_1', reg=reg_) 
    cdown3_2     =    CNR(  cdown3_1,  anCh*8,_name=name_+'clv3_2', reg=reg_)
    cpool4       = Pool2d(  cdown3_2,  anCh*16, name=name_+'clv4_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')
    ddown3_1     =    CNR(  dpool3,    anCh*8,_name=name_+'dlv3_1', reg=reg_) 
    ddown3_2     =    CNR(  ddown3_1,  anCh*8,_name=name_+'dlv3_2', reg=reg_)
    dpool4       = Pool2d(  ddown3_2,  anCh*16, name=name_+'dlv4_p')

    '''e path'''
    edown0_1     =    CNR(  einp,      anCh,  _name=name_+'elv0_1', reg=reg_)
    edown0_2     =    CNR(  edown0_1,  anCh,  _name=name_+'elv0_2', reg=reg_)
    epool1       = Pool2d(  edown0_2,  anCh*2, name=name_+'elv1_p') 
    edown1_1     =    CNR(  epool1,    anCh*2,_name=name_+'elv1_1', reg=reg_) 
    edown1_2     =    CNR(  edown1_1,  anCh*2,_name=name_+'elv1_2', reg=reg_)
    epool2       = Pool2d(  edown1_2,  anCh*4, name=name_+'elv2_p')
    edown2_1     =    CNR(  epool2,    anCh*4,_name=name_+'elv2_1', reg=reg_) 
    edown2_2     =    CNR(  edown2_1,  anCh*4,_name=name_+'elv2_2', reg=reg_)
    epool3       = Pool2d(  edown2_2,  anCh*8, name=name_+'elv3_p')
    edown3_1     =    CNR(  epool3,    anCh*8,_name=name_+'elv3_1', reg=reg_) 
    edown3_2     =    CNR(  edown3_1,  anCh*8,_name=name_+'elv3_2', reg=reg_)
    epool4       = Pool2d(  edown3_2,  anCh*16, name=name_+'elv4_p')


    ''' decoder '''
    pool4 = tf.concat([apool4,bpool4,cpool4,dpool4,epool4], axis=ch_dim)
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    down3_2 = tf.concat([adown3_2,bdown3_2,cdown3_2,ddown3_2,edown3_2], axis=ch_dim)
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1     =    CNR(        CC3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    up3_2     =    CNR(      up3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2,ddown2_2,edown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2,edown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2,edown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def UnetINShallowDiv5(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):

    mask = inp[:,15:,:,:]   
    ainp = tf.concat( [  inp[:,0:3,:,:],mask],axis=ch_dim)
    binp = tf.concat( [  inp[:,3:6,:,:],mask],axis=ch_dim)
    cinp = tf.concat( [  inp[:,6:9,:,:],mask],axis=ch_dim)
    dinp = tf.concat( [ inp[:,9:12,:,:],mask],axis=ch_dim)
    einp = tf.concat( [inp[:,12:15,:,:],mask],axis=ch_dim)
    anCh = nCh/5
    
    '''a path'''
    adown0_1     =    CNR(  ainp,      anCh,  _name=name_+'alv0_1', reg=reg_)
    adown0_2     =    CNR(  adown0_1,  anCh,  _name=name_+'alv0_2', reg=reg_)
    apool1       = Pool2d(  adown0_2,  anCh*2, name=name_+'alv1_p') 
    adown1_1     =    CNR(  apool1,    anCh*2,_name=name_+'alv1_1', reg=reg_) 
    adown1_2     =    CNR(  adown1_1,  anCh*2,_name=name_+'alv1_2', reg=reg_)
    apool2       = Pool2d(  adown1_2,  anCh*4, name=name_+'alv2_p')
    adown2_1     =    CNR(  apool2,    anCh*4,_name=name_+'alv2_1', reg=reg_) 
    adown2_2     =    CNR(  adown2_1,  anCh*4,_name=name_+'alv2_2', reg=reg_)
    apool3       = Pool2d(  adown2_2,  anCh*8, name=name_+'alv3_p')

    '''b path'''
    bdown0_1     =    CNR(  binp,      anCh,  _name=name_+'blv0_1', reg=reg_)
    bdown0_2     =    CNR(  bdown0_1,  anCh,  _name=name_+'blv0_2', reg=reg_)
    bpool1       = Pool2d(  bdown0_2,  anCh*2, name=name_+'blv1_p') 
    bdown1_1     =    CNR(  bpool1,    anCh*2,_name=name_+'blv1_1', reg=reg_) 
    bdown1_2     =    CNR(  bdown1_1,  anCh*2,_name=name_+'blv1_2', reg=reg_)
    bpool2       = Pool2d(  bdown1_2,  anCh*4, name=name_+'blv2_p')
    bdown2_1     =    CNR(  bpool2,    anCh*4,_name=name_+'blv2_1', reg=reg_) 
    bdown2_2     =    CNR(  bdown2_1,  anCh*4,_name=name_+'blv2_2', reg=reg_)
    bpool3       = Pool2d(  bdown2_2,  anCh*8, name=name_+'blv3_p')

    '''c path'''
    cdown0_1     =    CNR(  cinp,      anCh,  _name=name_+'clv0_1', reg=reg_)
    cdown0_2     =    CNR(  cdown0_1,  anCh,  _name=name_+'clv0_2', reg=reg_)
    cpool1       = Pool2d(  cdown0_2,  anCh*2, name=name_+'clv1_p') 
    cdown1_1     =    CNR(  cpool1,    anCh*2,_name=name_+'clv1_1', reg=reg_) 
    cdown1_2     =    CNR(  cdown1_1,  anCh*2,_name=name_+'clv1_2', reg=reg_)
    cpool2       = Pool2d(  cdown1_2,  anCh*4, name=name_+'clv2_p')
    cdown2_1     =    CNR(  cpool2,    anCh*4,_name=name_+'clv2_1', reg=reg_) 
    cdown2_2     =    CNR(  cdown2_1,  anCh*4,_name=name_+'clv2_2', reg=reg_)
    cpool3       = Pool2d(  cdown2_2,  anCh*8, name=name_+'clv3_p')

    '''d path'''
    ddown0_1     =    CNR(  dinp,      anCh,  _name=name_+'dlv0_1', reg=reg_)
    ddown0_2     =    CNR(  ddown0_1,  anCh,  _name=name_+'dlv0_2', reg=reg_)
    dpool1       = Pool2d(  ddown0_2,  anCh*2, name=name_+'dlv1_p') 
    ddown1_1     =    CNR(  dpool1,    anCh*2,_name=name_+'dlv1_1', reg=reg_) 
    ddown1_2     =    CNR(  ddown1_1,  anCh*2,_name=name_+'dlv1_2', reg=reg_)
    dpool2       = Pool2d(  ddown1_2,  anCh*4, name=name_+'dlv2_p')
    ddown2_1     =    CNR(  dpool2,    anCh*4,_name=name_+'dlv2_1', reg=reg_) 
    ddown2_2     =    CNR(  ddown2_1,  anCh*4,_name=name_+'dlv2_2', reg=reg_)
    dpool3       = Pool2d(  ddown2_2,  anCh*8, name=name_+'dlv3_p')

    '''e path'''
    edown0_1     =    CNR(  einp,      anCh,  _name=name_+'elv0_1', reg=reg_)
    edown0_2     =    CNR(  edown0_1,  anCh,  _name=name_+'elv0_2', reg=reg_)
    epool1       = Pool2d(  edown0_2,  anCh*2, name=name_+'elv1_p') 
    edown1_1     =    CNR(  epool1,    anCh*2,_name=name_+'elv1_1', reg=reg_) 
    edown1_2     =    CNR(  edown1_1,  anCh*2,_name=name_+'elv1_2', reg=reg_)
    epool2       = Pool2d(  edown1_2,  anCh*4, name=name_+'elv2_p')
    edown2_1     =    CNR(  epool2,    anCh*4,_name=name_+'elv2_1', reg=reg_) 
    edown2_2     =    CNR(  edown2_1,  anCh*4,_name=name_+'elv2_2', reg=reg_)
    epool3       = Pool2d(  edown2_2,  anCh*8, name=name_+'elv3_p')

    ''' decoder '''
    pool3 = tf.concat([apool3,bpool3,cpool3,dpool3,epool3], axis=ch_dim)
    down3_1     =    CNR(    pool3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    down3_2     =    CNR(  down3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    up3         = Conv2dT( down3_2,  nCh*4, name=name_+'lv3__up')
    
    down2_2 = tf.concat([adown2_2,bdown2_2,cdown2_2,ddown2_2,edown2_2], axis=ch_dim)
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')

    down1_2 = tf.concat([adown1_2,bdown1_2,cdown1_2,ddown1_2,edown1_2], axis=ch_dim)
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')

    down0_2 = tf.concat([adown0_2,bdown0_2,cdown0_2,ddown0_2,edown0_2], axis=ch_dim)
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')



def Unet_shallow(inp, n_out, is_Training, reg_, nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CBR(   inp,   nCh, is_Training, name=name_+'lv0_1', reg=reg_, _1x1Conv=_1x1Conv)
    down0_2     =    CBR(  down0_1,  nCh,  is_Training,name=name_+'lv0_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CBR(    pool1,  nCh*2, is_Training, name=name_+'lv1_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down1_2     =    CBR(  down1_1,  nCh*2, is_Training, name=name_+'lv1_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CBR(    pool2,  nCh*4, is_Training, name=name_+'lv2_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down2_2     =    CBR(  down2_1,  nCh*4, is_Training, name=name_+'lv2_2', reg=reg_, _1x1Conv=_1x1Conv)
    up2         = Conv2dT( down2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CBR(      CC1,  nCh*2, is_Training, name=name_+'lv1__1', reg=reg_, _1x1Conv=_1x1Conv)
    up1_2       =    CBR(    up1_1,  nCh*2, is_Training, name=name_+'lv1__2', reg=reg_, _1x1Conv=_1x1Conv)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CBR(      CC0,   nCh, is_Training, name=name_+'lv0__1', reg=reg_, _1x1Conv=_1x1Conv)
    
    up0_2   =    CBR(    up0_1,   nCh, is_Training, name=name_+'lv0__2', reg=reg_, _1x1Conv=_1x1Conv)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')
 
def UnetL3(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CBR(   inp,   nCh, is_Training, name=name_+'lv0_1', reg=reg_, _1x1Conv=_1x1Conv)
    down0_2     =    CBR(  down0_1,  nCh,  is_Training,name=name_+'lv0_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CBR(    pool1,  nCh*2, is_Training, name=name_+'lv1_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down1_2     =    CBR(  down1_1,  nCh*2, is_Training, name=name_+'lv1_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CBR(    pool2,  nCh*4, is_Training, name=name_+'lv2_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down2_2     =    CBR(  down2_1,  nCh*4, is_Training, name=name_+'lv2_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool3       = Pool2d(  down2_2,  nCh*8, name=name_+'lv3_p')
    down3_1     =    CBR(    pool3,  nCh*8, is_Training, name=name_+'lv3_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down3_2     =    CBR(  down3_1,  nCh*8, is_Training, name=name_+'lv3_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool4       = Pool2d(  down3_2, nCh*16, name=name_+'lv4_p')
    down4_1     =    CBR(    pool4, nCh*16, is_Training, name=name_+'lv4_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down4_2     =    CBR(  down4_1, nCh*16, is_Training, name=name_+'lv4_2', reg=reg_, _1x1Conv=_1x1Conv)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1       =    CBR(      CC3,  nCh*8, is_Training, name=name_+'lv3__1',reg=reg_, _1x1Conv=_1x1Conv)
    up3_2       =    CBR(    up3_1,  nCh*8, is_Training, name=name_+'lv3__2',reg=reg_, _1x1Conv=_1x1Conv)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CBR(      CC2,  nCh*4, is_Training, name=name_+'lv2__1', reg=reg_, _1x1Conv=_1x1Conv)
    up2_2       =    CBR(    up2_1,  nCh*4, is_Training, name=name_+'lv2__2', reg=reg_, _1x1Conv=_1x1Conv)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CBR(      CC1,  nCh*2, is_Training, name=name_+'lv1__1', reg=reg_, _1x1Conv=_1x1Conv)
    up1_2       =    CBR(    up1_1,  nCh*2, is_Training, name=name_+'lv1__2', reg=reg_, _1x1Conv=_1x1Conv)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CBR(      CC0,   nCh, is_Training, name=name_+'lv0__1', reg=reg_, _1x1Conv=_1x1Conv)
    up0_2       =    CBR(    up0_1,   nCh, is_Training, name=name_+'lv0__2', reg=reg_, _1x1Conv=_1x1Conv)

    return  Conv2d( up0_2, n_out, name=name_+'conv3x3Last', reg=reg)

def UnetIN(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CNR(   inp,   nCh, _name=name_+'lv0_1', reg=reg_)
    down0_2     =    CNR(  down0_1,  nCh,  _name=name_+'lv0_2', reg=reg_)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CNR(    pool1,  nCh*2, _name=name_+'lv1_1', reg=reg_) 
    down1_2     =    CNR(  down1_1,  nCh*2, _name=name_+'lv1_2', reg=reg_)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CNR(    pool2,  nCh*4, _name=name_+'lv2_1', reg=reg_) 
    down2_2     =    CNR(  down2_1,  nCh*4, _name=name_+'lv2_2', reg=reg_)
    
    pool3       = Pool2d(  down2_2,  nCh*8, name=name_+'lv3_p')
    down3_1     =    CNR(    pool3,  nCh*8, _name=name_+'lv3_1', reg=reg_) 
    down3_2     =    CNR(  down3_1,  nCh*8, _name=name_+'lv3_2', reg=reg_)
    
    pool4       = Pool2d(  down3_2, nCh*16, name=name_+'lv4_p')
    down4_1     =    CNR(    pool4, nCh*16, _name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CNR(  down4_1, nCh*16, _name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1       =    CNR(      CC3,  nCh*8, _name=name_+'lv3__1',reg=reg_)
    up3_2       =    CNR(    up3_1,  nCh*8, _name=name_+'lv3__2',reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CNR(      CC2,  nCh*4, _name=name_+'lv2__1', reg=reg_)
    up2_2       =    CNR(    up2_1,  nCh*4, _name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CNR(      CC1,  nCh*2, _name=name_+'lv1__1', reg=reg_)
    up1_2       =    CNR(    up1_1,  nCh*2, _name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CNR(      CC0,   nCh, _name=name_+'lv0__1', reg=reg_)
    up0_2       =    CNR(    up0_1,   nCh, _name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')


def Unet(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CBR(   inp,   nCh, is_Training, name=name_+'lv0_1', reg=reg_, _1x1Conv=_1x1Conv)
    down0_2     =    CBR(  down0_1,  nCh,  is_Training,name=name_+'lv0_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CBR(    pool1,  nCh*2, is_Training, name=name_+'lv1_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down1_2     =    CBR(  down1_1,  nCh*2, is_Training, name=name_+'lv1_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CBR(    pool2,  nCh*4, is_Training, name=name_+'lv2_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down2_2     =    CBR(  down2_1,  nCh*4, is_Training, name=name_+'lv2_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool3       = Pool2d(  down2_2,  nCh*8, name=name_+'lv3_p')
    down3_1     =    CBR(    pool3,  nCh*8, is_Training, name=name_+'lv3_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down3_2     =    CBR(  down3_1,  nCh*8, is_Training, name=name_+'lv3_2', reg=reg_, _1x1Conv=_1x1Conv)
    
    pool4       = Pool2d(  down3_2, nCh*16, name=name_+'lv4_p')
    down4_1     =    CBR(    pool4, nCh*16, is_Training, name=name_+'lv4_1', reg=reg_, _1x1Conv=_1x1Conv) 
    down4_2     =    CBR(  down4_1, nCh*16, is_Training, name=name_+'lv4_2', reg=reg_, _1x1Conv=_1x1Conv)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1       =    CBR(      CC3,  nCh*8, is_Training, name=name_+'lv3__1',reg=reg_, _1x1Conv=_1x1Conv)
    up3_2       =    CBR(    up3_1,  nCh*8, is_Training, name=name_+'lv3__2',reg=reg_, _1x1Conv=_1x1Conv)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CBR(      CC2,  nCh*4, is_Training, name=name_+'lv2__1', reg=reg_, _1x1Conv=_1x1Conv)
    up2_2       =    CBR(    up2_1,  nCh*4, is_Training, name=name_+'lv2__2', reg=reg_, _1x1Conv=_1x1Conv)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CBR(      CC1,  nCh*2, is_Training, name=name_+'lv1__1', reg=reg_, _1x1Conv=_1x1Conv)
    up1_2       =    CBR(    up1_1,  nCh*2, is_Training, name=name_+'lv1__2', reg=reg_, _1x1Conv=_1x1Conv)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CBR(      CC0,   nCh, is_Training, name=name_+'lv0__1', reg=reg_, _1x1Conv=_1x1Conv)
    up0_2       =    CBR(    up0_1,   nCh, is_Training, name=name_+'lv0__2', reg=reg_, _1x1Conv=_1x1Conv)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def Unet31(inp, n_out, is_Training, reg_=[], nCh=64, name_='', _1x1Conv=False):
    down0_1     =    CCBR(   inp,   nCh, is_Training, name=name_+'lv0_1', reg=reg_)
    down0_2     =    CCBR(  down0_1,  nCh,  is_Training,name=name_+'lv0_2', reg=reg_)
    
    pool1       = Pool2d(  down0_2,  nCh*2, name=name_+'lv1_p') 
    down1_1     =    CCBR(    pool1,  nCh*2, is_Training, name=name_+'lv1_1', reg=reg_) 
    down1_2     =    CCBR(  down1_1,  nCh*2, is_Training, name=name_+'lv1_2', reg=reg_)
    
    pool2       = Pool2d(  down1_2,  nCh*4, name=name_+'lv2_p')
    down2_1     =    CCBR(    pool2,  nCh*4, is_Training, name=name_+'lv2_1', reg=reg_) 
    down2_2     =    CCBR(  down2_1,  nCh*4, is_Training, name=name_+'lv2_2', reg=reg_)
    
    pool3       = Pool2d(  down2_2,  nCh*8, name=name_+'lv3_p')
    down3_1     =    CCBR(    pool3,  nCh*8, is_Training, name=name_+'lv3_1', reg=reg_) 
    down3_2     =    CCBR(  down3_1,  nCh*8, is_Training, name=name_+'lv3_2', reg=reg_)
    
    pool4       = Pool2d(  down3_2, nCh*16, name=name_+'lv4_p')
    down4_1     =    CCBR(    pool4, nCh*16, is_Training, name=name_+'lv4_1', reg=reg_) 
    down4_2     =    CCBR(  down4_1, nCh*16, is_Training, name=name_+'lv4_2', reg=reg_)
    up4         = Conv2dT( down4_2,  nCh*8, name=name_+'lv4__up')
    
    CC3         = tf.concat([down3_2, up4], axis=ch_dim)
    up3_1       =    CCBR(      CC3,  nCh*8, is_Training, name=name_+'lv3__1',reg=reg_)
    up3_2       =    CCBR(    up3_1,  nCh*8, is_Training, name=name_+'lv3__2',reg=reg_)
    up3         = Conv2dT(   up3_2,  nCh*4, name=name_+'lv3__up')
    
    CC2         = tf.concat([down2_2, up3], axis=ch_dim)
    up2_1       =    CCBR(      CC2,  nCh*4, is_Training, name=name_+'lv2__1', reg=reg_)
    up2_2       =    CCBR(    up2_1,  nCh*4, is_Training, name=name_+'lv2__2', reg=reg_)
    up2         = Conv2dT(   up2_2,  nCh*2, name=name_+'lv2__up')
    
    CC1         = tf.concat([down1_2, up2], axis=ch_dim)
    up1_1       =    CCBR(      CC1,  nCh*2, is_Training, name=name_+'lv1__1', reg=reg_)
    up1_2       =    CCBR(    up1_1,  nCh*2, is_Training, name=name_+'lv1__2', reg=reg_)
    up1         = Conv2dT(   up1_2,   nCh, name=name_+'lv1__up')
    
    CC0         = tf.concat([down0_2, up1], axis=ch_dim)
    up0_1       =    CCBR(      CC0,   nCh, is_Training, name=name_+'lv0__1', reg=reg_)
    up0_2       =    CCBR(    up0_1,   nCh, is_Training, name=name_+'lv0__2', reg=reg_)

    return  Conv1x1(   up0_2, n_out,name=name_+'conv1x1')

def SRnet(inputs, n_out, is_Training, reg_, nCh=64, _1x1Conv=False):
    down0_1     =    CBR(   inputs,   nCh, is_Training, name='lv0_1', reg=reg_,_1x1Conv=_1x1Conv)
    down0_2     =    CBR(  down0_1,   nCh,  is_Training,name='lv0_2', reg=reg_,_1x1Conv=_1x1Conv)
    
    down1_1     =    CBR(  down0_2,   nCh, is_Training, name='lv1_1', reg=reg_,_1x1Conv=_1x1Conv)
    down1_2     =    CBR(  down1_1,   nCh, is_Training, name='lv1_2', reg=reg_,_1x1Conv=_1x1Conv)
    
    down2_1     =    CBR(  down1_2,   nCh, is_Training, name='lv2_1', reg=reg_,_1x1Conv=_1x1Conv) 
    down2_2     =    CBR(  down2_1,   nCh, is_Training, name='lv2_2', reg=reg_,_1x1Conv=_1x1Conv)
    
    down3_1     =    CBR(  down2_2,   nCh, is_Training, name='lv3_1', reg=reg_,_1x1Conv=_1x1Conv) 
    down3_2     =    CBR(  down3_1,   nCh, is_Training, name='lv3_2', reg=reg_,_1x1Conv=_1x1Conv)
    
    down4_1     =    CBR(  down3_2,   nCh, is_Training, name='lv4_1', reg=reg_,_1x1Conv=_1x1Conv) 
    down4_2     =    CBR(  down4_1,   nCh, is_Training, name='lv4_2', reg=reg_,_1x1Conv=_1x1Conv)
    
    CC3         = tf.concat([down3_2, down4_2], axis=ch_dim)
    up3_1       =    CBR(      CC3,   nCh, is_Training, name='lv3__1',reg=reg_,_1x1Conv=_1x1Conv)
    up3_2       =    CBR(    up3_1,   nCh, is_Training, name='lv3__2',reg=reg_,_1x1Conv=_1x1Conv)
    
    CC2         = tf.concat([down2_2, up3_2], axis=ch_dim)
    up2_1       =    CBR(      CC2,   nCh, is_Training, name='lv2__1', reg=reg_,_1x1Conv=_1x1Conv)
    up2_2       =    CBR(    up2_1,   nCh, is_Training, name='lv2__2', reg=reg_,_1x1Conv=_1x1Conv)
    
    CC1         = tf.concat([down1_2, up2_2], axis=ch_dim)
    up1_1       =    CBR(      CC1,   nCh, is_Training, name='lv1__1', reg=reg_,_1x1Conv=_1x1Conv)
    up1_2       =    CBR(    up1_1,   nCh, is_Training, name='lv1__2', reg=reg_,_1x1Conv=_1x1Conv)
    
    CC0         = tf.concat([down0_2, up1_2], axis=ch_dim)
    up0_1       =    CBR(      CC0,   nCh, is_Training, name='lv0__1', reg=reg_,_1x1Conv=_1x1Conv)
    up0_2       =    CBR(    up0_1,   nCh, is_Training, name='lv0__2', reg=reg_,_1x1Conv=_1x1Conv)
    
    return  Conv1x1(   up0_2, n_out,name='conv1x1')

def SRnet31(inputs, n_out, is_Training,reg_, nCh=64, class_N=4, _1x1Conv=False):
    down0_1     =    CCBR(   inputs,   nCh, is_Training, name='lv0_1', reg=reg_)
    down0_2     =    CCBR(  down0_1,   nCh,  is_Training,name='lv0_2', reg=reg_)
    
    down1_1     =    CCBR(  down0_2,   nCh, is_Training, name='lv1_1', reg=reg_)
    down1_2     =    CCBR(  down1_1,   nCh, is_Training, name='lv1_2', reg=reg_)
    
    down2_1     =    CCBR(  down1_2,   nCh, is_Training, name='lv2_1', reg=reg_) 
    down2_2     =    CCBR(  down2_1,   nCh, is_Training, name='lv2_2', reg=reg_)
    
    down3_1     =    CCBR(  down2_2,   nCh, is_Training, name='lv3_1', reg=reg_) 
    down3_2     =    CCBR(  down3_1,   nCh, is_Training, name='lv3_2', reg=reg_)
    
    down4_1     =    CCBR(  down3_2,   nCh, is_Training, name='lv4_1', reg=reg_) 
    down4_2     =    CCBR(  down4_1,   nCh, is_Training, name='lv4_2', reg=reg_)
   
#    if use_zLogit:
#        latent_z    = tf.layers.dropout(down4_2,rate=0.99,training=is_Training)
#        logits = tf.layers.dense(tf.contrib.layers.flatten(latent_z),class_N,use_bias=False)
    
    CC3         = tf.concat([down3_2, down4_2], axis=ch_dim)
    up3_1       =    CCBR(      CC3,   nCh, is_Training, name='lv3__1',reg=reg_)
    up3_2       =    CCBR(    up3_1,   nCh, is_Training, name='lv3__2',reg=reg_)
    
    CC2         = tf.concat([down2_2, up3_2], axis=ch_dim)
    up2_1       =    CCBR(      CC2,   nCh, is_Training, name='lv2__1', reg=reg_)
    up2_2       =    CCBR(    up2_1,   nCh, is_Training, name='lv2__2', reg=reg_)
    
    CC1         = tf.concat([down1_2, up2_2], axis=ch_dim)
    up1_1       =    CCBR(      CC1,   nCh, is_Training, name='lv1__1', reg=reg_)
    up1_2       =    CCBR(    up1_1,   nCh, is_Training, name='lv1__2', reg=reg_)
    
    CC0         = tf.concat([down0_2, up1_2], axis=ch_dim)
    up0_1       =    CCBR(      CC0,   nCh, is_Training, name='lv0__1', reg=reg_)
    up0_2       =    CCBR(    up0_1,   nCh, is_Training, name='lv0__2', reg=reg_)
    
#    if use_zLogit:
#        return  Conv1x1(   up0_2, n_out,name='conv1x1'), logits
#    else:
    return  Conv1x1(   up0_2, n_out,name='conv1x1')



def SRnet2(inputs, n_out, is_Training,reg_, nCh=64, class_N=4, use_zLogit=False):
    down0_1     =    CCBR(   inputs,   nCh, is_Training, name='lv0_1', reg=reg_)
    down0_2     =    CCBR(  down0_1,   nCh,  is_Training,name='lv0_2', reg=reg_)
    
    down1_1     =    CCBR(  down0_2,   nCh, is_Training, name='lv1_1', reg=reg_)
    down1_2     =    CCBR(  down1_1,   nCh, is_Training, name='lv1_2', reg=reg_)
    
    down2_1     =    CCBR(  down1_2,   nCh, is_Training, name='lv2_1', reg=reg_) 
    down2_2     =    CCBR(  down2_1,   nCh, is_Training, name='lv2_2', reg=reg_)
    
    down3_1     =    CCBR(  down2_2,   nCh, is_Training, name='lv3_1', reg=reg_) 
    down3_2     =    CCBR(  down3_1,   nCh, is_Training, name='lv3_2', reg=reg_)
    
    down4_1     =    CCBR(  down3_2,   nCh, is_Training, name='lv4_1', reg=reg_) 
    down4_2     =    CCBR(  down4_1,   nCh, is_Training, name='lv4_2', reg=reg_)
   
    if use_zLogit:
        latent_z    = tf.layers.dropout(down4_2,rate=0.99,training=is_Training)
        logits = tf.layers.dense(tf.contrib.layers.flatten(latent_z),class_N,use_bias=False)
    
    CC3         = tf.concat([down3_2, down4_2], axis=ch_dim)
    up3_1       =    CCBR(      CC3,   nCh, is_Training, name='lv3__1',reg=reg_)
    up3_2       =    CCBR(    up3_1,   nCh, is_Training, name='lv3__2',reg=reg_)
    
    CC2         = tf.concat([down2_2, up3_2], axis=ch_dim)
    up2_1       =    CCBR(      CC2,   nCh, is_Training, name='lv2__1', reg=reg_)
    up2_2       =    CCBR(    up2_1,   nCh, is_Training, name='lv2__2', reg=reg_)
    
    CC1         = tf.concat([down1_2, up2_2], axis=ch_dim)
    up1_1       =    CCBR(      CC1,   nCh, is_Training, name='lv1__1', reg=reg_)
    up1_2       =    CCBR(    up1_1,   nCh, is_Training, name='lv1__2', reg=reg_)
    
    CC0         = tf.concat([down0_2, up1_2], axis=ch_dim)
    up0_1       =    CCBR(      CC0,   nCh, is_Training, name='lv0__1', reg=reg_)
    up0_2       =    CCBR(    up0_1,   nCh, is_Training, name='lv0__2', reg=reg_)
    
    if use_zLogit:
        return  Conv1x1(   up0_2, n_out,name='conv1x1'), logits
    else:
        return  Conv1x1(   up0_2, n_out,name='conv1x1')

def StartG(inputs,n_out,is_Training, reg_, nCh=64, class_N=4):
    c1 = tf.layers.conv2d(x, filters=ch_out, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=use_bias,data_format=d_form, kernel_initializer=li.xavier_initializer(), kernel_regularizer=reg, name="".join((name,"_Conv")))


