import io
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from HoloUtils import getComplex, intFresnel2D, fourierPhaseMultiply

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


def reconstImg(U, sigma, V, start, end, jump):
    for i in range(start, end, jump):
        reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    return reconstimg

def showHologram(complex_matrix, title):
    t = intFresnel2D(complex_matrix, False, pp, dist, wlen)
    t = fourierPhaseMultiply(t, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray')
    plt.title(title)
    plt.show()

#creazione di SVD parte immaginaria
U_IMAG, sigma_IMAG, V_IMAG = np.linalg.svd(np.imag(holo))
#creazione di SVD parte reale
U_REAL, sigma_REAL, V_REAL = np.linalg.svd(np.real(holo))

# # parte immaginaria ricostruita
# reconstImage_IMAG = reconstImg(U_IMAG, sigma_IMAG, V_IMAG, 5, 101, 5) #TOTALE
#
# # parte real ricostruita
# reconstImage_REAL = reconstImg(U_REAL, sigma_REAL, V_REAL, 5, 101, 5) #TOTALE
#
# # costruizione della matrice complessa ottenuta dalla ricostruzione delle due parti
# complex_matrix = getComplex(reconstImage_REAL, reconstImage_IMAG)

CROP_RATE = 20
# COMPRESSIONE -> CROP DELLE MATRICI
# tagliata la matrice U REAL da 1080x1080 a 1080x200 -> fino a 200 c'è informazione
U_REAL_CUT = U_REAL[:, :CROP_RATE]
# tagliata la matrice V REAL da 1080x1080 a 200x1080 -> fino a 200 c'è informazione
V_REAL_CUT = V_REAL[:CROP_RATE, :]
# tagliata la matrice U IMAG da 1080x1080 a 1080x200 -> fino a 200 c'è informazione
U_IMAG_CUT = U_IMAG[:, :CROP_RATE]
# tagliata la matrice V IMAG da 1080x1080 a 200x1080 -> fino a 200 c'è informazione
V_IMAG_CUT = V_IMAG[:CROP_RATE, :]

# PARTE REAL
np.savez('svdImageCompression/U_REAL_NP', U_REAL)
np.savez('svdImageCompression/U_REAL_P', U_REAL_CUT)
np.savez('svdImageCompression/V_REAL_NP', V_REAL)
np.savez('svdImageCompression/V_REAL_P', V_REAL_CUT)
np.savez('svdImageCompression/SIGMA_REAL', sigma_REAL)

# PARTE IMAG
np.savez('svdImageCompression/U_IMAG_NP', U_IMAG)
np.savez('svdImageCompression/U_IMAG_P', U_IMAG_CUT)
np.savez('svdImageCompression/V_IMAG_NP', V_IMAG)
np.savez('svdImageCompression/V_IMAG_P', V_IMAG_CUT)
np.savez('svdImageCompression/SIGMA_IMAG', sigma_IMAG)

# DECOMPRESSIONE

# carica il file npz
with np.load('svdImageCompression/U_REAL_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    U_R_COMPRESS = data['arr_0']

with np.load('svdImageCompression/V_REAL_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    V_R_COMPRESS = data['arr_0']

with np.load('svdImageCompression/SIGMA_REAL.npz') as data:
    # ottieni tutti gli array presenti nel file
    SIGMA_R = data['arr_0']

with np.load('svdImageCompression/U_IMAG_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    U_I_COMPRESS = data['arr_0']

with np.load('svdImageCompression/V_IMAG_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    V_I_COMPRESS = data['arr_0']

with np.load('svdImageCompression/SIGMA_IMAG.npz') as data:
    # ottieni tutti gli array presenti nel file
    SIGMA_I = data['arr_0']

# parte immaginaria ricostruita
reconstCompResImage_IMAG = reconstImg(U_I_COMPRESS, SIGMA_I, V_I_COMPRESS, 5, CROP_RATE+1, 5)
# parte real ricostruita
reconstCompResImage_REAL = reconstImg(U_R_COMPRESS, SIGMA_R, V_R_COMPRESS, 5, CROP_RATE+1, 5)
# costruizione della matrice complessa ottenuta dalla ricostruzione delle due parti
complex_matrix = getComplex(reconstCompResImage_REAL, reconstCompResImage_IMAG)
showHologram(complex_matrix, 'Ricostruita')


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}", f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}", f"{num:.1f}Yi{suffix}"


total_size_HOL_NP = os.path.getsize('svdImageCompression/U_REAL_NP.npz') + os.path.getsize('svdImageCompression/V_REAL_NP.npz') + os.path.getsize('svdImageCompression/SIGMA_REAL.npz') + os.path.getsize('svdImageCompression/U_IMAG_NP.npz') + os.path.getsize('svdImageCompression/V_IMAG_NP.npz') + os.path.getsize('svdImageCompression/SIGMA_IMAG.npz')
_ , total_size_HOL_NP_formatted = sizeof_fmt(total_size_HOL_NP)
print('NON COMPRESSA: ', total_size_HOL_NP_formatted)

total_size_HOL_P = os.path.getsize('svdImageCompression/U_REAL_P.npz') + os.path.getsize('svdImageCompression/V_REAL_P.npz') + os.path.getsize('svdImageCompression/SIGMA_REAL.npz') + os.path.getsize('svdImageCompression/U_IMAG_P.npz') + os.path.getsize('svdImageCompression/V_IMAG_P.npz') + os.path.getsize('svdImageCompression/SIGMA_IMAG.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_P)
print('COMPRESSA: ', total_size_HOL_P_formatted)

rate = (float(total_size_HOL_P) / float(total_size_HOL_NP)) * 100
print(f"Rate: {(100 - rate):.2f} %")

