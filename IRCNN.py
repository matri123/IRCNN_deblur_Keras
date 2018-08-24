import os
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import keras.backend as K

from psnr_main import PSNR
from scipy.signal import fftconvolve
from keras.backend.tensorflow_backend import set_session

from model import model_stack

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def get_rho_and_net_index(sigma, iter_num):
    rho_index=np.zeros([iter_num,])
    net_index=np.zeros([iter_num,])

    lambda_=(sigma**2)/3
    modelSigma1=49
    modelSigma2=13
    modelSigmaS=np.logspace(np.log10(modelSigma1),np.log10(modelSigma2),iter_num)
    for i in range(iter_num):
        rho_index[i]=(lambda_*255**2)/(modelSigmaS[i]**2)
    
    net_index=np.ceil(modelSigmaS/2)
    net_index=np.clip(net_index,1,25)

    return rho_index, net_index

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def pad_for_kernel(img,kernel,mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p,p] + (img.ndim-2)*[(0,0)]
    return np.pad(img, padding, mode)

def crop_for_kernel(img,kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0],-p[0]),slice(p[1],-p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]

def edgetaper_alpha(kernel,img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel,1-i),img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z,z[0:1]],0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)

# use edge processing if necessary
def edgetaper(img,kernel,n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[...,np.newaxis]
        alpha  = alpha[...,np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img,_kernel,'wrap'),kernel,mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img

def test(model,sigma):

    im_list = ['05','06','07','08']
    ke_list = ['01','02','03','04','05','06','07','08']
    psnrsum = 0

    for im_idx in im_list:
        for ke_idx in ke_list:
            print(str(im_idx) + ' ' + str(ke_idx))
            name = "./Levin09blurdata/im" + im_idx +"_flit" + ke_idx + ".mat"
            data = scipy.io.loadmat(name)
            blurred = data['y']
            kernel  = data['f']
            kernel = np.rot90(kernel,2)
            gt      = data['x']

            y = to_tensor(edgetaper(pad_for_kernel(blurred,kernel,'edge'),kernel))
            x0 = y
            k = np.tile(kernel[np.newaxis], (y.shape[0],1,1))
            sigmas = np.tile(sigma, (y.shape[0], 1)).astype(np.float32)

            pred = model.predict_on_batch([x0, y, k, sigmas])
            predi = crop_for_kernel(from_tensor(pred),kernel)

            psnr, ssim = PSNR(predi, gt, kernel.shape[0])
            print('im:%s ke:%s PSNR:%f'%(im_idx, ke_idx, psnr))
            psnrsum += psnr

    print(psnrsum / (len(im_list) * len(ke_list)))


if __name__=="__main__":
    K.clear_session()

    iter_num=30
    sigma = 0.01

    rho_index, net_index = get_rho_and_net_index(sigma, iter_num)
    model = model_stack(iter_num, rho_index, net_index)
    test(model, sigma)

