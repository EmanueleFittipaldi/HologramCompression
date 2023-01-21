import pywt
import numpy as np
import os
from HoloUtils import rate

dict_name = 'wavelet_compression/'


def wavelet_compression(hologram, pp, wlen, dist, filename, wavelet, mode, value):
    """The wavedec2 method from the pywt library is used to perform a 2D wavelet transform on a given signal or image.
    This method decomposes the input data into a set of wavelet coefficients, which represent
    the different frequency components of the data. The wavedec2 method returns a list of these coefficients,
    which can then be manipulated or analyzed.

    A wavelet transform is a type of mathematical operation that can be used to analyze
    the frequency content of a signal or image. It allows for the representation of a signal
    or image in terms of both time and frequency, making it useful for many applications,
    such as image and audio compression, noise removal, and signal denoising."""
    print('WAVELET COMPRESSION ALGORITHM')
    #SALVATAGGIO MATRICI
    if not os.path.isdir(dict_name + filename):
        os.makedirs(dict_name + filename)

    np.savez(dict_name + filename + '/matrix_HOLO', hologram)
    # holo = hologram
    # holo = holo[:, 420:]
    # holo = holo[:, :-420]
    # hologramReconstruction(holo,pp,dist,wlen)
    # Perform a 2D wavelet transform on the hologram data
    coefficients = pywt.wavedec2(hologram, wavelet=wavelet)

    # Apply lossy compression to the wavelet coefficients
    for coeffs in coefficients:
        for i in range(len(coeffs)):
            for j in range(len(coeffs[i])):
                coeffs[i][j] = pywt.threshold(coeffs[i][j], value=value, mode=mode)

    coefficients = np.array(coefficients, dtype='object')
    np.savez_compressed(dict_name + filename +'/wavelet_coeff.npz', coefficients)


def wavelet_decompression(filename, pp, wlen, dist, wavelet):

    with np.load(dict_name + filename +'/wavelet_coeff.npz', allow_pickle=True) as data:
        # ottieni tutti gli array presenti nel file
        coefficients = data['arr_0']
    # Reconstruct the compressed hologram data
    coefficients = coefficients.tolist()
    compressed_hologram = pywt.waverec2(coefficients, wavelet=wavelet)
    # holo = compressed_hologram
    # holo = holo[:, 420:]
    # holo = holo[:, :-420]
    # hologramReconstruction(holo,pp,dist,wlen)
    compressa = os.path.getsize(dict_name + filename +'/wavelet_coeff.npz')
    original = os.path.getsize(dict_name + filename +'/matrix_HOLO.npz')
    print(compressa)
    print(original)
    rate(original, compressa)
