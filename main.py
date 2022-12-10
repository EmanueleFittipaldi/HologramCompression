import bz2
import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math as math
import scipy.fft
import fourierfresnel1D
import gzip
from givensphaserot import givenphaserot

# Per scalare il plot dell'ologramma ricostruito
plt.rcParams["figure.figsize"] = (60, 60)


# Queste due funzioni servono per ricostruire l'ologramma.
def intfresnel2D(x, fw, pp, z, wlen):
    """
    Apply the Fresnel diffraction formula to a 2D array.

    This function takes a 2D array and applies the Fresnel diffraction formula
    to it, using the given pixel pitch, distance, and wavelength. The resulting
    transformed array is returned.

    Args:
        x (ndarray): The input 2D array as a NumPy array.
        fw (bool): A flag indicating whether the input array is in the forward (True)
                   or inverse (False) Fourier domain.
        pp (float): The pixel pitch of the input array in meters.
        z (float): The distance between the input array and the diffraction plane in meters.
        wlen (float): The wavelength of the light used to record the input array in meters.

    Returns:
        ndarray: The transformed array after applying the Fresnel diffraction formula.

    Raises:
        AssertionError: If the size of the input array is not even, or if the input array
                        is not square.
    """
    assert (np.size(x) % 2) == 0, "La dimensione dell'ologramma deve essere pari"
    assert np.size(x, 0) == np.size(x, 1), "L'ologramma deve essere quadrato (da risolvere nelle prossime versioni)"

    def fw_fresnel(r):
        return np.round(fpropfun(r, pp, fz, wlen))

    def bw_fresnel(r):
        return np.round(bpropfun(r, pp, bz, wlen))

    def fwdfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, True)

    def revfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, False)

    def apply_transform():
        if fw:
            o = np.conjugate(x[:, 1::2])
            e = x[:, 0::2]

            e = e + bw_fresnel(o)
            o = o - fw_fresnel(e)
            e = e + bw_fresnel(o)

            x[:, 0::2] = -o
            x[:, 1::2] = np.conjugate(e)

            return x
        else:
            e = np.conj(x[:, 1::2])  # righe le prende tutte, colonne prende a 2 a 2
            o = -x[:, 0::2]  # righe le prende tutte, colonne parte dalla prima  e a step di 2

            e = e - bw_fresnel(o)
            o = o + fw_fresnel(e)
            e = e - bw_fresnel(o)

            x[:, 0::2] = e
            x[:, 1::2] = np.conj(o)
            return x

    # Applico la diffrazione di Fresnel
    fpropfun = fwdfourierfresnel1D
    fz = z
    bpropfun = revfourierfresnel1D
    bz = z

    for i in [-1, 1]:
        if fw:
            x = np.rot90(x, i)
        x = apply_transform()
        if not fw:
            x = np.rot90(x, i)
    return x


def fourier_phase_multiply(r, fw, pp, z, wlen):
    """
    Apply a phase rotation to a 2D Fourier transform.

    This function takes a 2D Fourier transform and applies a phase rotation to it,
    based on the given pixel pitch, distance, and wavelength. The resulting
    transformed array is returned.

    Args:
        r (ndarray): The input 2D Fourier transform as a NumPy array.
        fw (bool): A flag indicating whether the input array is in the forward (True)
                   or inverse (False) Fourier domain.
        pp (float): The pixel pitch of the input array in meters.
        z (float): The distance between the input array and the phase plane in meters.
        wlen (float): The wavelength of the light used to record the input array in meters.

    Returns:
        ndarray: The transformed array after applying the phase rotation.
    """
    n = np.size(r, 0)
    xx = np.power(np.matrix(np.arange(-n / 2, n / 2)), 2)
    ppout = wlen * np.abs(z) / n / pp

    temp = (math.pi * np.power(ppout, 2)) / (wlen * np.abs(z))
    p = np.multiply(temp, (xx + xx.transpose())) + (2 * np.pi * z) / wlen
    return givenphaserot(r, p, fw)


def ricostruzioneOlogramma(holo, pp, dist, wlen):
    """
    Reconstruct an hologram using the Fresnel diffraction method.

    This function takes an input hologram and applies the Fresnel diffraction
    formula to reconstruct the original object from the hologram. It then displays
    the resulting image using matplotlib.

    Args:
        holo (ndarray): The input hologram as a NumPy array.
        pp (float): The pixel pitch of the hologram in meters.
        dist (float): The distance between the hologram and the reconstruction plane in meters.
        wlen (float): The wavelength of the light used to record the hologram in meters.

    Returns:
        None
    """
    # Ricostruzione dell'ologramma tramite diffrazione di Fresnel
    t = intfresnel2D(np.csingle(holo), False, pp, dist, wlen)
    t = fourier_phase_multiply(t, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray')
    plt.show()


def compressGZIP(matrix):
    """
    Compress a complex-valued matrix using the Matrix Market format and gzip.

    This function saves the input matrix to a file using the Matrix Market format,
    which supports complex numbers. It then compresses the file using gzip. The
    resulting compressed file can be read and manipulated by other tools and
    libraries that support the Matrix Market format and gzip compression.

    Args:
        matrix (ndarray): The input complex-valued matrix as a NumPy array.

    Returns:
        None
    """
    scipy.io.mmwrite('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/ComplexMatrix.mtx', matrix,
                     field='complex', precision=1)
    # Open the input file in binary mode
    with open('ComplexMatrix.mtx', 'rb') as input_file:
        # Open the output file in binary mode, with gzip compression enabled
        with gzip.open('ComplexMatrix.gz', 'wb') as output_file:
            # Read the input file in chunks
            chunk = input_file.read(1024)
            while chunk:
                # Write each chunk to the output file
                output_file.write(chunk)
                # Read the next chunk
                chunk = input_file.read(1024)


def getComplex(matrix_REAL, matrix_IMAG):
    """
        Create a complex-valued matrix from separate real and imaginary matrices.

        This function takes two input matrices, one containing the real part of
        the complex values, and the other containing the imaginary part. It uses
        the `complex()` function to combine the real and imaginary parts into
        complex values, and stores these values in a new matrix. The resulting
        complex matrix is returned.

        Args:
            matrix_REAL (ndarray): The input matrix containing the real part of the complex values.
            matrix_IMAG (ndarray): The input matrix containing the imaginary part of the complex values.

        Returns:
            ndarray: The complex-valued matrix created from the input matrices.
        """
    # Create an empty matrix to store the complex values
    complex_matrix = np.empty(matrix_REAL.shape, dtype=complex)

    # Loop over the rows and columns of the matrix
    for i in range(matrix_REAL.shape[0]):
        for j in range(matrix_REAL.shape[1]):
            # Use the complex() function to combine the real and imaginary parts
            complex_matrix[i, j] = complex(matrix_REAL[i, j], matrix_IMAG[i, j])
    return complex_matrix


def compressBZIP2(matrix):
    """
        Compress a complex-valued matrix using the Matrix Market format and bzip2.

        This function saves the input matrix to a file using the Matrix Market format,
        which supports complex numbers. It then compresses the file using bzip2. The
        resulting compressed file can be read and manipulated by other tools and
        libraries that support the Matrix Market format and bzip2 compression.

        Args:
            matrix (ndarray): The input complex-valued matrix as a NumPy array.

        Returns:
            None
        """
    scipy.io.mmwrite('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/ComplexMatrix.mtx', matrix,
                     field='complex', precision=1)
    # Open the input file in binary mode
    with open('ComplexMatrix.mtx', 'rb') as input_file:
        # Open the output file in binary mode, with bzip2 compression enabled
        with bz2.open('ComplexMatrix.bz2', 'wb') as output_file:
            # Read the input file in chunks
            chunk = input_file.read(1024)
            while chunk:
                # Write each chunk to the output file
                output_file.write(chunk)
                # Read the next chunk
                chunk = input_file.read(1024)


def main():
    f = scipy.io.loadmat('Hol_2D_dice.mat')  # aprire il file .mat

    # Dice Parameters
    pp = 8e-6  # pixel pitch
    pp = np.matrix(pp)
    wlen = 632.8e-9  # wavelenght
    wlen = np.matrix(wlen)
    dist = 9e-1  # propogation depth
    dist = np.matrix(dist)

    # holo è la matrice di numeri complessi
    holo = np.matrix(f['Hol'])

    # Effettuo un crop da 1920*1080 a 1080*1080 perché l'algoritmo per la
    holo = holo[:, 420:]
    holo = holo[:, :-420]

    # Per visualizzare la parte reale ed immaginaria dell'ologramma
    # plt.imshow(np.imag(holo), cmap="gray")
    # plt.show()  # mostro l'ologramma pre - compressione
    # plt.imshow(np.real(holo), cmap="gray")
    # plt.show()

    # Prova di ricostruzione ologramma dalle due jpeg salvate

    # img = cv.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_reale.jpg', cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_immaginaria.jpg', cv.IMREAD_GRAYSCALE)
    # ricostruzioneOlogramma(getComplex(img,img2), pp, dist, wlen)

    # Creazione delle immagini della matrice reale ed immaginaria

    # matriceReale = np.real(holo)
    # matriceImmaginaria = np.imag(holo)
    # matplotlib.image.imsave('matrice_reale.jpg', matriceReale, cmap='gray')
    # matplotlib.image.imsave('matrice_immaginaria.jpg', matriceImmaginaria, cmap='gray')
    # img_real = Image.open('img_real.jpg')
    # img_real_gray = img_real.convert('LA')

    # Prova di bitplane slicing

    # bitPlaneSlicing('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_immaginaria.jpg')


if __name__ == '__main__':
    main()
