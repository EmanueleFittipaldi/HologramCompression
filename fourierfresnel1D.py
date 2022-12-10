import numpy as np

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

def IDFT(x):
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

def fourierfresnel1D(x, pp, z, wlen, forward):
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
    e = np.exp(c[0,0] * (np.power((w * pp), 2)))

    if forward:
        x = np.fft.fftshift(np.multiply(x, e), 0)
        y = np.multiply(e, np.fft.fftshift(DFT_slow(x), 0)) / np.sqrt(n)
        return y
    else:
        mult = np.multiply(e,x)
        x = IDFT(np.fft.ifftshift(mult,0))
        y = np.sqrt(np.size(x, 0)) * np.multiply(np.fft.ifftshift(x, 0), e)
        return y
