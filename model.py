import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.engine.topology import Layer
from denoise_net import denoise_model

def _get_inputs(img_shape=(None,None,1),kernel_shape=(None,None)):
    x_in = Input(shape=img_shape,    name="x_in")
    y    = Input(shape=img_shape,    name="y")
    k    = Input(shape=kernel_shape, name="k")
    s    = Input(shape=(1,),         name="s")
    return x_in, y, k, s

# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers
def psf2otf(psf, img_shape):
    # shape and type of the point spread function(s)
    psf_shape = tf.shape(psf)
    psf_type = psf.dtype

    # coordinates for 'cutting up' the psf tensor
    midH = tf.floor_div(psf_shape[0], 2)
    midW = tf.floor_div(psf_shape[1], 2)

    # slice the psf tensor into four parts
    top_left     = psf[:midH, :midW, :, :]
    top_right    = psf[:midH, midW:, :, :]
    bottom_left  = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]

    # prepare zeros for filler
    zeros_bottom = tf.zeros([psf_shape[0] - midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    zeros_top    = tf.zeros([midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)

    # construct top and bottom row of new tensor
    top    = tf.concat([bottom_right, zeros_bottom, bottom_left], 1)
    bottom = tf.concat([top_right,    zeros_top,    top_left],    1)

    # prepare additional filler zeros and put everything together
    zeros_mid = tf.zeros([img_shape[0] - psf_shape[0], img_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    pre_otf = tf.concat([top, zeros_mid, bottom], 0)
    # output shape: [img_shape[0], img_shape[1], channels_in, channels_out]

    # fast fourier transform, transposed because tensor must have shape [..., height, width] for this
    otf = tf.fft2d(tf.cast(tf.transpose(pre_otf, perm=[2,3,0,1]), tf.complex64))

    # output shape: [channels_in, channels_out, img_shape[0], img_shape[1]]
    return otf


class IRCNNstage(Layer):
    def __init__(self, rho, model_idx=7, **kwargs):
        self.rho = rho
        self.model_idx = model_idx
        super(IRCNNstage, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(IRCNNstage, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        x_t, y, k, s = inputs

        imagesize = tf.shape(x_t)[1:3]
        kk = tf.expand_dims(tf.transpose(k, [1,2,0]), -1)
        fft_k = psf2otf(kk, imagesize)[:,0,:,:]
        denominator = tf.square(tf.abs(fft_k))
        fft_y = tf.fft2d(tf.cast(y[:,:,:,0], tf.complex64))

        upperleft = tf.conj(fft_k) * fft_y

        fft_x = tf.fft2d(tf.cast(x_t[:,:,:,0], tf.complex64))

        rho = tf.expand_dims(self.rho, -1)
        z = (upperleft + tf.cast(rho, tf.complex64)*fft_x) / tf.cast((tf.cast(denominator, tf.float64) + rho), tf.complex64)
        z = tf.to_float(tf.ifft2d(z))
        z1 = tf.expand_dims(z, -1)

        net = denoise_model()
        net.load_weights('./model_keras/net%d.hdf5'%(self.model_idx))
        residual = net(z1)
        z = z1 - residual
        return z

def model_stage(rho, net_idx=7):
    x_t, y, k, s = _get_inputs()
    x_out = IRCNNstage(rho, net_idx)([x_t, y, k, s])
    return Model([x_t, y, k, s], x_out)


def model_stack(stage, rho_idx, net_idx):
    x, y, k, s = _get_inputs()
    x0 = y
    output = []
    for i in range(stage):
        model = model_stage(rho_idx[i], net_idx[i])
        output.append(model([(output[-1] if i>0 else x0), y, k, s]))

    return Model([x, y, k, s], output[-1])

