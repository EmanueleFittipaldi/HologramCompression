import cv2
import matplotlib
import scipy
import numpy as np
import os
import pillow_jpls
from PIL import Image

from HoloUtils import getComplex, hologramReconstruction, intFresnel2D, fourierPhaseMultiply

holoFileName = 'Hol_2D_dice.mat'
f = scipy.io.loadmat(holoFileName)  # aprire il file .mat
print(f.keys())
# Dice Parameters
pp = np.matrix(f['pitch'][0]) # pixel pitch
wlen = np.matrix(f['wlen'][0]) # wavelenght
dist = np.matrix(f['zobj'][0]) # propogation depth

#Holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])
print(holo.shape)
#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
holo = holo[:, 420:]
holo = holo[:, :-420]
print(holo.shape)
np.savez('Matrix_HOLO', holo)
#Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
imagMatrix = np.imag(holo)
realMatrix = np.real(holo)
name_real = 'matrice_reale.jpg'
name_imag = 'matrice_immaginaria.jpg'

#JPG
matplotlib.image.imsave(name_real, realMatrix, cmap='gray')
matplotlib.image.imsave(name_imag, imagMatrix, cmap='gray')
img_real = cv2.imread(name_real, cv2.IMREAD_GRAYSCALE)
img_imag = cv2.imread(name_imag, cv2.IMREAD_GRAYSCALE)


#BMAP
# matplotlib.image.imsave(name_real, realMatrix, cmap='gray')
# matplotlib.image.imsave(name_imag, imagMatrix, cmap='gray')
#
# img = cv2.imread(name_real, cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(name_imag, cv2.IMREAD_GRAYSCALE)

#Ricostruisco la matrice dei numeri complessi a partire dalle matrici
#contenenti le parti immaginarie e reali e ricostruisco l'ologramma per
#verificare la quantità del degradamento dell'immagine
#complexMatrix = getComplex(img,img2)
# hologramReconstruction(complexMatrix,pp,dist,wlen)

total_size_HOL_NC = os.path.getsize('Matrix_HOLO.npz')
total_size_HOL_C = os.path.getsize(name_real) + os.path.getsize(name_imag)
rate = (float(total_size_HOL_C) / float(total_size_HOL_NC)) * 100
print(f"Rate compressione: {(100 - rate):.2f} %")
print('1:',int(total_size_HOL_NC/total_size_HOL_C))