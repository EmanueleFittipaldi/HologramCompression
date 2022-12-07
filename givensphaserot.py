import numpy as np

"""% applies a lossless Givens phase rotation to an integer signal X, with phase delay P
"""

def givenphaserot(x, p, fw):
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
        #print(x[542,521])
        x = x + sg * 1j * np.round(np.multiply(b, np.real(x)))
        #print(x[542,521])
        x = x + sg * np.round(np.multiply(a, np.imag(x)))
        #print(x[542,521])
        return x

    # determine in what phase quadrant  to operate
    def quadrant_swap(xtemp, q):
        m = np.logical_or(q == 1, q == 3)
        # xtemp[m] = (-np.imag(xtemp[m])) + (1j*np.real(xtemp[m]))
        np.putmask(xtemp, m, (-np.imag(xtemp)) + (1j * np.real(xtemp)))
        np.putmask(xtemp, q > 1, -xtemp)
        # xtemp[q>1] = -xtemp[q>1]
        return xtemp

    q = np.mod(np.floor(2*p/np.pi+0.5), 4)
    #print(q)
    somma = np.pi/2
    t = np.mod(p+(np.pi/4),np.pi/2) - (np.pi/4)
    #print(t[4,4]) #0.09190...

    peps = 1e-7  # phase epsilon
    if fw:
        x = quadrant_swap(x,q)

    x = givens_rot(t, x)

    if not fw:
        x = quadrant_swap(x,np.mod(4 - q, 4))
    return x
