import pywt
import scipy.io
import numpy as np
import os
from HoloUtils import hologramReconstruction, compressBZIP2

holoFileName = 'CornellBox2_10K.mat'
f = scipy.io.loadmat(holoFileName)  # aprire il file .mat
print(f.keys())
# Dice Parameters
pp = np.matrix(f['pp'][0]) # pixel pitch
wlen = np.matrix(f['wlen'][0]) # wavelenght
dist = np.matrix(f['zrec'][0]) # propogation depth
# holo è la matrice di numeri complessi
holo = np.matrix(f['H'])
#Holo è la matrice di numeri complessi

#Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
# holo = holo[:, 420:]
# holo = holo[:, :-420]
np.savez('Matrix_HOLO', holo)

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
    # hologramReconstruction(compressed_hologram,pp,dist,wlen)
    compressa = os.path.getsize('waveletCoeff.npz')
    original = os.path.getsize('Matrix_HOLO.npz')
    print(compressa)
    print(original)
    rate = (float(compressa) / float(original)) * 100
    print(f"Rate: {(100 - rate):.2f} %")
    print('1:', int(original / compressa))
compress_hologram(holo,'compressedHologram.txt',wavelet='db4',mode='hard')
