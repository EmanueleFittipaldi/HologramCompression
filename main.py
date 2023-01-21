import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.fft

from imageApproach import image_based_compression, image_based_decompression
from svd_image_compression import svd_compression, svd_decompression
from waveletApproach import wavelet_compression, wavelet_decompression
from zfp_compression import zfp_compression, zfp_decompression


def main():
    # Caricamento ologramma
    holo_file_name = 'Hol_2D_multi.mat'
    f = scipy.io.loadmat(holo_file_name)  # aprire il file .mat
    print(f.keys())
    # Dice Parameters
    pp = np.matrix(f['pitch'][0])  # pixel pitch
    wlen = np.matrix(f['wlen'][0])  # wavelenght
    dist = np.matrix(f['zobj1'][0])  # propogation depth
    # holo è la matrice di numeri complessi
    holo = np.matrix(f['Hol'])

    # SVD
    k_value_svd = 20
    svd_compression(holo_file_name, k_value_svd, holo, pp, wlen, dist)
    svd_decompression(holo_file_name, k_value_svd, pp, wlen, dist)

    # WAVELET
    wavelet_value = 200000000
    wavelet_compression(holo, pp, wlen, dist, holo_file_name, wavelet='db4', mode='hard', value=wavelet_value)
    wavelet_decompression(holo_file_name, pp, wlen, dist, wavelet='db4')

    # ZFP
    zfp_compression(holo, holo_file_name, pp, wlen, dist, rate=2)
    zfp_decompression(holo_file_name, pp, wlen, dist)

    # IMAGE BASED
    image_based_compression(holo, holo_file_name, pp, wlen, dist)
    image_based_decompression(holo_file_name, pp, wlen, dist)



if __name__ == '__main__':
    main()
