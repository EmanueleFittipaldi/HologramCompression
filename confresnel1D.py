import numpy as np
import math as math

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT(x):
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    molt = np.dot(M, x)
    return molt / N

def convfresnel1D(x, pp, z, wlen):
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


