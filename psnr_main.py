import numpy as np
from comp_psnr import comp_quantitative

def PSNR(blur, clear, kernel_size = 19):
    psnr, ssim = comp_quantitative(blur,clear,kernel_size)
    return psnr, ssim

def psnr(x,y):
    return -10 * np.math.log10(((x -  y)** 2).mean())

def comp_quantitative(fe, sharp_ground,k):
    fe = fe.astype('float64')

    ks = int(np.floor(k/2))
    fe1 = fe[0 + ks: fe.shape[-2] - ks , 0 + ks : fe.shape[-1] - ks]
    m = fe1.shape[-2]
    n = fe1.shape[-1]
    psnr0 = 0.0
    ssim0 = 0.0
    m1 = sharp_ground.shape[0]
    n1 = sharp_ground.shape[1]
    for i in range(0,m1-m ,1):
        for j in range(0,n1-n ,1):
            sharp_ground1 = sharp_ground[i:m+i,j:j+n]
            psnr1 = psnr(fe1,sharp_ground1)
            #ssim1 = compare_ssim(fe1, sharp_ground1)
            ssim1 = 0
            psnr0 = max(psnr0,psnr1)
            ssim0 = max(ssim0,ssim1)
    return psnr0, ssim0




