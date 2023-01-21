import bz2
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math as math
import scipy.fft
import gzip

plt.rcParams["figure.figsize"] = (60, 60)
def intFresnel2D(x, fw, pp, z, wlen):
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
        return fourierFresnel1D(x, pp, z, wlen, True)

    def revfourierfresnel1D(x, pp, z, wlen):
        return fourierFresnel1D(x, pp, z, wlen, False)

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


def fourierPhaseMultiply(r, fw, pp, z, wlen):
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
    return givenPhaseRot(r, p, fw)

from scipy.special import fresnel
def fresnel_inverse_2d(U, x, y, wavelength, distance):
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2 + distance**2)
    C, S = fresnel(x/(wavelength*distance))
    return (np.exp(1j*k*r)/(1j*wavelength*distance))*U*(C+1j*S)

def hologramReconstruction(holo, pp, dist, wlen):
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
    t = intFresnel2D(np.csingle(holo), False, pp, dist, wlen)
    t = fourierPhaseMultiply(t, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray_r')
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


def dftSlow(x):
    """
     Compute the discrete Fourier Transform of the 1D array x.

     This function takes a 1D array and computes its discrete Fourier Transform
     using a slow, but intuitive algorithm. The resulting complex-valued array
     is returned.

     Args:
         x (ndarray): The input 1D array as a NumPy array.

     Returns:
         ndarray: The discrete Fourier Transform of the input array.
     """
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def synthesize(f_hat, axis=0):
    """
        Synthesize a signal from its Fourier Transform.

        This function takes the Fourier Transform of a signal, and uses the inverse
        Fourier Transform to synthesize the original signal. The resulting signal
        is returned.

        Args:
            f_hat (ndarray): The Fourier Transform of the signal as a NumPy array.
            axis (int, optional): The axis along which to apply the Fourier Transform. Defaults to 0.

        Returns:
            ndarray: The synthesized signal.
        """
    f_hat = np.fft.ifftshift(f_hat * f_hat.shape[axis], axes=axis)
    f = np.fft.ifft(f_hat, axis=axis)
    return f


def idft(x):
    """
        Compute the inverse discrete Fourier Transform of the 1D array x.

        This function takes a 1D array and computes its inverse discrete Fourier
        Transform using a matrix multiplication algorithm. The resulting complex-
        valued array is returned.

        Args:
            x (ndarray): The input 1D array as a NumPy array.

        Returns:
            ndarray: The inverse discrete Fourier Transform of the input array.
        """
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    molt = np.dot(M, x)
    return molt / N


def fourierFresnel1D(x, pp, z, wlen, forward):
    """
        Compute the 1D Fourier-Fresnel Transform of the array x.

        This function takes a 1D array and computes its Fourier-Fresnel Transform
        using the discrete Fourier Transform and the Fresnel Transform. The resulting
        complex-valued array is returned.

        Args:
            x (ndarray): The input 1D array as a NumPy array.
            pp (float): The pixel pitch of the input array.
            z (float): The distance to the image plane.
            wlen (float): The wavelength of the light used to generate the input array.
            forward (bool): Whether to compute the forward (True) or inverse (False) Fourier-Fresnel Transform.

        Returns:
            ndarray: The Fourier-Fresnel Transform of the input array.
        """
    c = 1j * np.pi / wlen / z  # c valore corretto
    if not forward:
        c = -c
    n = np.size(x, 0)
    ppout = wlen * np.abs(z) / n / pp
    w = np.matrix(np.arange(-n / 2, n / 2)).getH()
    e = np.exp(c[0, 0] * (np.power((w * pp), 2)))

    if forward:
        x = np.fft.fftshift(np.multiply(x, e), 0)
        y = np.multiply(e, np.fft.fftshift(dftSlow(x), 0)) / np.sqrt(n)
        return y
    else:
        mult = np.multiply(e, x)
        x = idft(np.fft.ifftshift(mult, 0))
        y = np.sqrt(np.size(x, 0)) * np.multiply(np.fft.ifftshift(x, 0), e)
        return y


def convFresnel1D(x, pp, z, wlen):
    """
        Propagate an hologram by a given distance using the Fresnel approximation.

        This function takes an input hologram and a distance, and uses the Fresnel
        approximation to propagate the hologram by the given distance. The resulting
        hologram is returned.

        Args:
            x (ndarray): The input hologram as a NumPy array.
            wlen (float): The wavelength of the light used to create the hologram.
            z (float): The distance by which to propagate the hologram.

        Returns:
            ndarray: The propagated hologram.
        """
    c = -1j * wlen * z * math.pi
    n = x.shape[0]
    w = np.matrix(np.arange(-n / 2, n / 2)).getH() / (np.matmul(np.matrix(n), pp))  # valori ok
    potenza = np.power(w, 2)
    molt = c[0, 0] * potenza
    espon = np.exp(molt)
    # ifftshift = np.fft.ifftshift(espon)

    # mult = np.multiply(DFT_slow(x), np.fft.ifftshift(espon))
    y = idft((np.multiply(dftSlow(x), np.fft.ifftshift(espon))))

    # print("y: ", y[1023,511])

    return y


def givenPhaseRot(x, p, fw):
    """The function givenphaserot computes the Givens rotation of an input array x
    using the phase values in another input array p. The direction of the rotation
    is determined by the fw boolean input, with True indicating a forward rotation
    and False indicating a backward rotation. The function first checks that the dimensions
    of x and p are equal. It then applies a Givens rotation to x using the values in p,
    and applies a quadrant swap if necessary depending on the direction of the rotation
    and the values in p. The output is the rotated version of x."""

    assert np.size(x) == np.size(p), "Input argument X & P must be equal dimension"

    # givens rotation
    def givens_rot(theta, x):
        m = np.abs(theta) < peps

        a = np.divide((np.cos(theta) - 1), np.sin(theta))
        a[m] = 0

        b = np.sin(theta)
        b[m] = 0

        if fw:
            sg = 1
        else:
            sg = -1

        x = x + sg * np.round(np.multiply(a, np.imag(x)))
        # print(x[542,521])
        x = x + sg * 1j * np.round(np.multiply(b, np.real(x)))
        # print(x[542,521])
        x = x + sg * np.round(np.multiply(a, np.imag(x)))
        # print(x[542,521])
        return x

    # determine in what phase quadrant  to operate
    def quadrant_swap(xtemp, q):
        m = np.logical_or(q == 1, q == 3)
        # xtemp[m] = (-np.imag(xtemp[m])) + (1j*np.real(xtemp[m]))
        np.putmask(xtemp, m, (-np.imag(xtemp)) + (1j * np.real(xtemp)))
        np.putmask(xtemp, q > 1, -xtemp)
        # xtemp[q>1] = -xtemp[q>1]
        return xtemp

    q = np.mod(np.floor(2 * p / np.pi + 0.5), 4)
    # print(q)
    somma = np.pi / 2
    t = np.mod(p + (np.pi / 4), np.pi / 2) - (np.pi / 4)
    # print(t[4,4]) #0.09190...

    peps = 1e-7  # phase epsilon
    if fw:
        x = quadrant_swap(x, q)

    x = givens_rot(t, x)

    if not fw:
        x = quadrant_swap(x, np.mod(4 - q, 4))
    return x

import cv2

def bitPlaneSlicing(imgPath):
    # Load the image in grayscale
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    # Create an array of 8 elements with values 1, 2, 4, 8, 16, 32, 64, 128
    masks = [1, 2, 4, 8, 16, 32, 64, 128]

    result = np.empty(img.shape)

    # Iterate over the masks
    for mask in masks:
        # Extract the bit plane corresponding to the mask
        bitplane = cv2.bitwise_and(img, mask)

        if mask not in [1,2,4,8,16,32,64]:
            result = cv2.bitwise_or(result, bitplane)

    cv2.imshow('risultato',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}", f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}", f"{num:.1f}Yi{suffix}"

def rate(holo_original, holo_compressed):
    rate = (float(holo_compressed) / float(holo_original)) * 100
    print(f"Rate compressione: {(100 - rate):.2f} %")
    print(str(int(holo_original / holo_compressed))+':1')







