import cv2
import matplotlib
import numpy as np
import os

from HoloUtils import getComplex, hologramReconstruction, intFresnel2D, fourierPhaseMultiply, rate, sizeof_fmt

dict_name = 'image_based_compression/'

def image_based_compression(holo, filename, pp, wlen, dist):
    print('IMAGE BASED COMPRESSION ALGORITHM')
    if not os.path.isdir(dict_name + filename):
        os.makedirs(dict_name + filename)
    np.savez(dict_name + filename + '/matrix_HOLO', holo)
    # holog = holo
    # holog = holog[:, 420:]
    # holog = holog[:, :-420]
    # hologramReconstruction(holog,pp,dist,wlen)

    #Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
    imagMatrix = np.imag(holo)
    realMatrix = np.real(holo)
    name_real = dict_name + filename + '/matrice_reale.jpg'
    name_imag = dict_name + filename + '/matrice_immaginaria.jpg'

    #JPG
    matplotlib.image.imsave(name_real, realMatrix, cmap='gray')
    matplotlib.image.imsave(name_imag, imagMatrix, cmap='gray')

def image_based_decompression(filename, pp, wlen, dist):
    print('IMAGE BASED DECOMPRESSION ALGORITHM')
    name_real = dict_name + filename + '/matrice_reale.jpg'
    name_imag = dict_name + filename + '/matrice_immaginaria.jpg'

    img_real = cv2.imread(name_real, cv2.IMREAD_GRAYSCALE)
    img_imag = cv2.imread(name_imag, cv2.IMREAD_GRAYSCALE)

    #Ricostruisco la matrice dei numeri complessi a partire dalle matrici
    #contenenti le parti immaginarie e reali e ricostruisco l'ologramma per
    #verificare la quantità del degradamento dell'immagine
    complexMatrix = getComplex(img_real,img_imag)
    # holog = complexMatrix
    # holog = holog[:, 420:]
    # holog = holog[:, :-420]
    # hologramReconstruction(holog, pp, dist, wlen)


    total_size_HOL_NC = os.path.getsize(dict_name + filename + '/Matrix_HOLO.npz')
    total_size_HOL_C = os.path.getsize(name_real) + os.path.getsize(name_imag)
    _, total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_NC)
    print('NON COMPRESSA: ', total_size_HOL_P_formatted)

    _, total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_C)
    print('COMPRESSA: ', total_size_HOL_P_formatted)

    rate(total_size_HOL_NC, total_size_HOL_C)