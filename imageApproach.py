import cv2
import matplotlib
import scipy
import numpy as np


from HoloUtils import getComplex,hologramReconstruction

f = scipy.io.loadmat('Hol_2D_dice.mat')  # aprire il file .mat

#Parametri del dado
pp = 8e-6  # pixel pitch
pp = np.matrix(pp)
wlen = 632.8e-9  # wavelenght
wlen = np.matrix(wlen)
dist = 9e-1  # propogation depth
dist = np.matrix(dist)

#Holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])

#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
holo = holo[:, 420:]
holo = holo[:, :-420]

#Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
imagMatrix = np.imag(holo)
realMatrix = np.real(holo)

matplotlib.image.imsave('matrice_reale.bmp', realMatrix, cmap='gray')
matplotlib.image.imsave('matrice_immaginaria.bmp', imagMatrix, cmap='gray')

img = cv2.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_reale.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_immaginaria.bmp', cv2.IMREAD_GRAYSCALE)


#Ricostruisco la matrice dei numeri complessi a partire dalle matrici
#contenenti le parti immaginarie e reali e ricostruisco l'ologramma per
#verificare la quantità del degradamento dell'immagine
complexMatrix = getComplex(img,img2)
hologramReconstruction(complexMatrix,pp,dist,wlen)