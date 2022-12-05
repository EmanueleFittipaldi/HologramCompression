import numpy as np

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def synthesize(f_hat, axis=0):
    f_hat = np.fft.ifftshift(f_hat * f_hat.shape[axis], axes=axis)
    f = np.fft.ifft(f_hat, axis=axis)
    return f

def IDFT(x):
    x = np.matrix(x)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    molt = np.dot(M, x)
    return molt / N

def fourierfresnel1D(x, pp, z, wlen, forward):
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
