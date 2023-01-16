import os
import cv2
import zfpy
import matplotlib
import numpy as np
import scipy.io
from numpy import ndarray

from HoloUtils import getComplex, hologramReconstruction


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}", f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}", f"{num:.1f}Yi{suffix}"

holoFileName = 'Hol_2D_dice.mat'
f = scipy.io.loadmat(holoFileName)  # aprire il file .mat
print(f.keys())
#Parametri

pp = 8e-6  # pixel pitch
pp = np.matrix(pp)
wlen = 632.8e-9  # wavelenght
wlen = np.matrix(wlen)
dist = 9e-1  # propogation depth
dist = np.matrix(dist)

# holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])

#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
holo = holo[:, 420:]
holo = holo[:, :-420]


#Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
imagMatrix = np.imag(holo)
realMatrix = np.real(holo)
np.savez('fpzipCompression/immaginaria_NC', imagMatrix)
np.savez('fpzipCompression/reale_NC', realMatrix)



# Comprimo matrice immaginaria con zfpy
compressed_data_imag: object = zfpy.compress_numpy(imagMatrix, tolerance=1e-1)
np.savez('fpzipCompression/immaginaria_C', compressed_data_imag)
decompressed_array_imag: object = zfpy.decompress_numpy(compressed_data_imag)

# confirm lossy compression/decompression
np.testing.assert_allclose(imagMatrix, decompressed_array_imag, atol=1e-1)
#print( data_again_imag)
matplotlib.image.imsave('matrice_immaginaria.bmp', decompressed_array_imag, cmap = 'gray')

# Comprimo matrice reale con zfpy
compressed_data_real = zfpy.compress_numpy(realMatrix,tolerance=1e-1)
np.savez('fpzipCompression/reale_C', compressed_data_real)
decompressed_array_real: object = zfpy.decompress_numpy(compressed_data_real)

# confirm lossy compression/decompression
np.testing.assert_allclose(realMatrix, decompressed_array_real, atol=1e-1)
matplotlib.image.imsave('matrice_reale.bmp', decompressed_array_real, cmap = 'gray')


img = cv2.imread('matrice_reale.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('matrice_immaginaria.bmp', cv2.IMREAD_GRAYSCALE)
complexMatrix = getComplex(img,img2)
hologramReconstruction(complexMatrix, pp, dist, wlen)

total_size_HOL_NC = os.path.getsize('fpzipCompression/immaginaria_NC.npz') + os.path.getsize('fpzipCompression/reale_NC.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_NC)
print('NON COMPRESSA: ', total_size_HOL_P_formatted)

total_size_HOL_C = os.path.getsize('fpzipCompression/immaginaria_C.npz') + os.path.getsize('fpzipCompression/reale_C.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_C)
print('COMPRESSA: ', total_size_HOL_P_formatted)

rate = (float(total_size_HOL_C) / float(total_size_HOL_NC)) * 100
print(f"Rate: {(100 - rate):.2f} %")














