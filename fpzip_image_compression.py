import base64
import io
import PIL.Image as Image
import cv2
import fpzip
import matplotlib
import numpy as np
import scipy.io



from HoloUtils import  hologramReconstruction, getComplex

# Dice Parameters
pp = 8e-6  # pixel pitch
pp = np.matrix(pp)
wlen = 632.8e-9  # wavelenght
wlen = np.matrix(wlen)
dist = 9e-1  # propogation depth
dist = np.matrix(dist)

# holo è la matrice di numeri complessi
f = scipy.io.loadmat('Hol_2D_dice.mat')

# holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])

#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
holo = holo[:, 420:]
holo = holo[:, :-420]


#Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
imagMatrix = np.imag(holo)
realMatrix = np.real(holo)
#print(imagMatrix)

#Comprimo matrice immaginaria con fpzip
dataImag = np.array(imagMatrix) # up to 4d float or double array
compress_bytes_imag  =  fpzip.compress (dataImag ,  precision = 0 ,  order = 'F' ) #restituisce un byte string
# Decompressione parte immaginaria
data_again_imag = fpzip.decompress(compress_bytes_imag, order='F')
np.savez('fpzipCompression/ImmaginariaCompressa', compress_bytes_imag)
matplotlib.image.imsave('matrice_immaginaria.bmp', data_again_imag, cmap='gray')
#print( data_again_imag)

#Comprimo matrice reale con fpzip
dataReal = np.array(realMatrix) # up to 4d float or double array
compress_bytes_real  =  fpzip.compress ( dataReal ,  precision = 0 ,  order = 'F' )
#Decompressione parte reale
data_again_real = fpzip.decompress(compress_bytes_real, order='F')
matplotlib.image.imsave('matrice_reale.bmp', data_again_real, cmap='gray')
#print(compress_bytes_real)

img = cv2.imread('matrice_immaginaria.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('matrice_reale.bmp', cv2.IMREAD_GRAYSCALE)

complexMatrix = getComplex(img,img2)

hologramReconstruction(complexMatrix,pp,dist,wlen)












