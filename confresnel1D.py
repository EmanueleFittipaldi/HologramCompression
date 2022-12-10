import numpy as np
import math as math

def DFT_slow(x):
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

def IDFT(x):
    """
        Compute the inverse discrete Fourier Transform of the 1D array x.

        This function takes a 1D array and computes its inverse discrete Fourier Transform.
        The resulting complex-valued array is returned.

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

def convfresnel1D(x, pp, z, wlen):
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
    y = IDFT((np.multiply(DFT_slow(x), np.fft.ifftshift(espon))))

    # print("y: ", y[1023,511])

    return y


