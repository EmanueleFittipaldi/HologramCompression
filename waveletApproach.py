import pywt
import scipy.io
import numpy as np

from HoloUtils import hologramReconstruction, compressBZIP2

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

def compress_hologram(hologram, filename, wavelet='db4', mode='hard'):
    """The wavedec2 method from the pywt library is used to perform a 2D wavelet transform on a given signal or image.
    This method decomposes the input data into a set of wavelet coefficients, which represent
    the different frequency components of the data. The wavedec2 method returns a list of these coefficients,
    which can then be manipulated or analyzed.

    A wavelet transform is a type of mathematical operation that can be used to analyze
    the frequency content of a signal or image. It allows for the representation of a signal
    or image in terms of both time and frequency, making it useful for many applications,
    such as image and audio compression, noise removal, and signal denoising."""

    # Perform a 2D wavelet transform on the hologram data
    coefficients = pywt.wavedec2(hologram, wavelet=wavelet)

    # Apply lossy compression to the wavelet coefficients
    # value = 200000000
    for coeffs in coefficients:
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i])):
                coeffs[i][j] = pywt.threshold(coeffs[i][j],value=200000000, mode=mode)


    coefficients = np.array(coefficients,dtype='object')
    np.savez_compressed('waveletCoeff.npz', coefficients)

    #Reconstruct the compressed hologram data
    coefficients =  coefficients.tolist()
    compressed_hologram = pywt.waverec2(coefficients, wavelet=wavelet)
    hologramReconstruction(compressed_hologram,pp,dist,wlen)

compress_hologram(holo,'compressedHologram.txt',wavelet='db4',mode='hard')
