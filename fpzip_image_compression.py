import os
import cv2
import fpzip
import matplotlib
import numpy as np
import scipy.io


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
# Dice Parameters
pp = np.matrix(f['pitch'][0]) # pixel pitch
wlen = np.matrix(f['wlen'][0]) # wavelenght
dist = np.matrix(f['zobj'][0]) # propogation depth
# holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])


#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
holo = holo[:, 420:]
holo = holo[:, :-420]

#Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
imagMatrix = np.imag(holo)
realMatrix = np.real(holo)
np.savez('fpzipCompression/immaginaria_NC', imagMatrix)
np.savez('fpzipCompression/reale_NC', imagMatrix)

#print(imagMatrix)


# Comprimo matrice immaginaria con fpzip
dataImag = np.array(imagMatrix) # up to 4d float or double array
compress_bytes_imag  =  fpzip.compress (dataImag ,  precision = 0 ,  order = 'F' ) #restituisce un byte string
np.savez('fpzipCompression/immaginaria_C', compress_bytes_imag)
# Decompressione parte immaginaria
data_again_imag = fpzip.decompress(compress_bytes_imag, order = 'F')
matplotlib.image.imsave('matrice_immaginaria.bmp', data_again_imag, cmap = 'gray')
#print( data_again_imag)


#Comprimo matrice reale con fpzip
dataReal = np.array(realMatrix) # up to 4d float or double array
compress_bytes_real  =  fpzip.compress( dataReal ,  precision = 0,  order = 'F' )
np.savez('fpzipCompression/reale_C', compress_bytes_real)
#Decompressione parte reale
data_again_real = fpzip.decompress(compress_bytes_real, order = 'F')
matplotlib.image.imsave('matrice_reale.bmp', data_again_real, cmap = 'gray')
#print(compress_bytes_real)

img = cv2.imread('matrice_reale.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('matrice_immaginaria.bmp', cv2.IMREAD_GRAYSCALE)


complexMatrix = getComplex(img,img2)


# hologramReconstruction(complexMatrix, pp, dist, wlen)

total_size_HOL_NC = os.path.getsize('fpzipCompression/immaginaria_NC.npz') + os.path.getsize('fpzipCompression/reale_NC.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_NC)
print('NON COMPRESSA: ', total_size_HOL_P_formatted)



total_size_HOL_C = os.path.getsize('fpzipCompression/immaginaria_C.npz') + os.path.getsize('fpzipCompression/reale_C.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_C)
print('COMPRESSA: ', total_size_HOL_P_formatted)

rate = (float(total_size_HOL_C) / float(total_size_HOL_NC)) * 100
print(f"Rate: {(100 - rate):.2f} %")










