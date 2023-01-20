import os.path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from HoloUtils import getComplex, intFresnel2D, fourierPhaseMultiply

holoFileName = 'Hol_2D_dice.mat'
f = scipy.io.loadmat(holoFileName)  # aprire il file .mat
print(f.keys())
# Dice Parameters
pp = np.matrix(f['pitch'][0]) # pixel pitch
wlen = np.matrix(f['wlen'][0]) # wavelenght
dist = np.matrix(f['zobj'][0]) # propogation depth
# holo è la matrice di numeri complessi
holo = np.matrix(f['Hol'])


# pad_width = ((0, holo.shape[1]-holo.shape[0]), (0, 0))
# padded = np.pad(holo, pad_width, mode='constant', constant_values=0)

def reconst_matrix(U, sigma, V, start, end, jump):
    for i in range(start, end, jump):
        reconst_matrix = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

    return reconst_matrix

def showHologram(complex_matrix, title, final):
    if (final):
        complex_matrix = complex_matrix[:, 420:]
        complex_matrix = complex_matrix[:, :-420]

    t = intFresnel2D(complex_matrix, False, pp, dist, wlen)
    t = fourierPhaseMultiply(t, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray_r')
    plt.title(title)
    plt.show()

print(holo.shape)
# #original hologram
# showHologram(holo, 'Originale', False)
print('Matrice complessa: ', holo.shape)
U_HOLO, SIGMA_HOLO, V_HOLO = np.linalg.svd(holo)
print('Matrice U: ', U_HOLO.shape)
print('Matrice SIGMA: ', SIGMA_HOLO.shape)
print('Matrice V: ', V_HOLO.shape)


#COMPRESSIONE UTILIZZANDO IL VALORE k
K_VALUE = 20
print('K_VALUE: ', K_VALUE)
# tagliata la matrice U REAL utilizzando k specificato
U_HOLO_CUT = U_HOLO[:, :K_VALUE]
# tagliata la matrice V REAL utilizzando k specificato
V_HOLO_CUT = V_HOLO[:K_VALUE, :]
# tagliata la diagonale SIGMA REAL prendendo i k valori singolari
SIGMA_HOLO_CUT = SIGMA_HOLO[:K_VALUE]
#print delle shapes delle matrici tagliate
print('Matrice U ridotta: ', U_HOLO_CUT.shape)
print('Matrice SIGMA ridotta: ', SIGMA_HOLO_CUT.shape)
print('Matrice V ridotta: ', V_HOLO_CUT.shape)

#SALVATAGGIO MATRICI
if not os.path.isdir('svdImageCompression/' + holoFileName):
    os.makedirs('svdImageCompression/' + holoFileName)

#Matrix Original
np.savez('svdImageCompression/' + holoFileName + '/Matrix_HOLO', holo)

# PARTE IMAG
np.savez('svdImageCompression/' + holoFileName + '/U_HOLO_P', U_HOLO_CUT)
np.savez('svdImageCompression/' + holoFileName + '/V_HOLO_P', V_HOLO_CUT)
np.savez('svdImageCompression/' + holoFileName + '/SIGMA_HOLO_P', SIGMA_HOLO_CUT)

# DECOMPRESSIONE

# carica il file npz
with np.load('svdImageCompression/' + holoFileName + '/U_HOLO_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    U_COMPRESS = data['arr_0']

with np.load('svdImageCompression/' + holoFileName + '/V_HOLO_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    V_COMPRESS = data['arr_0']

with np.load('svdImageCompression/' + holoFileName + '/SIGMA_HOLO_P.npz') as data:
    # ottieni tutti gli array presenti nel file
    SIGMA_COMPRESS = data['arr_0']

print('Matrice U caricata: ', U_COMPRESS.shape)
print('Matrice SIGMA caricata: ', SIGMA_COMPRESS.shape)
print('Matrice V caricata: ', V_COMPRESS.shape)

# matrice ricostruita
matrix_rec = reconst_matrix(U_COMPRESS, SIGMA_COMPRESS, V_COMPRESS, 5, K_VALUE+1, 5)
# showHologram(matrix_rec, 'Ologramma decompresso', True)
# np.savez('svdImageCompression/' + holoFileName + '/Matrix_RICOSTRUITA', matrix_rec)

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}", f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}", f"{num:.1f}Yi{suffix}"

total_size_HOL_ORIGINAL = os.path.getsize('svdImageCompression/' + holoFileName + '/Matrix_HOLO.npz')
_ , total_size_HOL_ORIGINAL_formatted = sizeof_fmt(total_size_HOL_ORIGINAL)
print('Dimensione matrice originale: ', total_size_HOL_ORIGINAL)

print('Dimensione matrice U compressa: ', os.path.getsize('svdImageCompression/' + holoFileName + '/U_HOLO_P.npz'))
print('Dimensione matrice SIGMA compressa: ', os.path.getsize('svdImageCompression/' + holoFileName + '/SIGMA_HOLO_P.npz'))
print('Dimensione matrice V compressa: ', os.path.getsize('svdImageCompression/' + holoFileName + '/V_HOLO_P.npz'))

total_size_HOL_P = os.path.getsize('svdImageCompression/' + holoFileName + '/U_HOLO_P.npz') + os.path.getsize('svdImageCompression/' + holoFileName + '/V_HOLO_P.npz') + os.path.getsize('svdImageCompression/' + holoFileName + '/SIGMA_HOLO_P.npz')
_ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_P)
print('Somma dimensioni matrici compresse: ', total_size_HOL_P)
print(total_size_HOL_ORIGINAL)
print(total_size_HOL_P)
rate = (float(total_size_HOL_P) / float(total_size_HOL_ORIGINAL)) * 100
print(f"Rate compressione: {(100 - rate):.2f} %")

print('1:',int(total_size_HOL_ORIGINAL/total_size_HOL_P))