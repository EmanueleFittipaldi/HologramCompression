import os.path
import matplotlib.pyplot as plt
import numpy as np
from HoloUtils import intFresnel2D, fourierPhaseMultiply, sizeof_fmt, rate

dict_name = 'svd_compression/'

def reconst_matrix(U, sigma, V, start, end, jump):
    for i in range(start, end, jump):
        reconst_matrix = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

    return reconst_matrix

def showHologram(complex_matrix, pp, wlen, dist, title, only_fourier):
    matrix_s = complex_matrix
    if matrix_s.shape[0] != matrix_s.shape[1]:
        if matrix_s.shape[0] < matrix_s.shape[1]:
            pad_width = ((0, matrix_s.shape[1] - matrix_s.shape[0]), (0, 0))
            matrix_s = np.pad(matrix_s, pad_width, mode='constant', constant_values=0)
        else:
            pad_width = ((0, 0),(matrix_s.shape[0] - matrix_s.shape[1],0))
            matrix_s = np.pad(matrix_s, pad_width, mode='constant', constant_values=0)
    if not only_fourier:
        t = intFresnel2D(matrix_s, False, pp, dist, wlen)
        t = fourierPhaseMultiply(t, False, pp, dist, wlen)
    else:
        t = fourierPhaseMultiply(matrix_s, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray_r')
    plt.title(title)
    plt.show()

def svd_compression(holo_file_name, k_value, holo, pp, wlen, dist):

    print('SVD COMPRESSION ALGORITHM')
    # original hologram
    # showHologram(holo, pp, wlen, dist, 'Originale', False)
    print('Matrice complessa: ', holo.shape)
    # decomposizione in tre matrici
    U_HOLO, SIGMA_HOLO, V_HOLO = np.linalg.svd(holo)
    print('Matrice U: ', U_HOLO.shape)
    print('Matrice SIGMA: ', SIGMA_HOLO.shape)
    print('Matrice V: ', V_HOLO.shape)

    # COMPRESSIONE UTILIZZANDO IL VALORE k
    print('k_value: ', k_value)
    # tagliata la matrice U REAL utilizzando k specificato
    U_HOLO_CUT = U_HOLO[:, :k_value]
    # tagliata la matrice V REAL utilizzando k specificato
    V_HOLO_CUT = V_HOLO[:k_value, :]
    # tagliata la diagonale SIGMA REAL prendendo i k valori singolari
    SIGMA_HOLO_CUT = SIGMA_HOLO[:k_value]
    #print delle shapes delle matrici tagliate
    print('Matrice U ridotta: ', U_HOLO_CUT.shape)
    print('Matrice SIGMA ridotta: ', SIGMA_HOLO_CUT.shape)
    print('Matrice V ridotta: ', V_HOLO_CUT.shape)

    #SALVATAGGIO MATRICI
    if not os.path.isdir(dict_name + holo_file_name):
        os.makedirs(dict_name + holo_file_name)

    #Matrix Original
    np.savez(dict_name + holo_file_name + '/Matrix_HOLO', holo)

    np.savez(dict_name + holo_file_name + '/U_HOLO_P', U_HOLO_CUT)
    np.savez(dict_name + holo_file_name + '/V_HOLO_P', V_HOLO_CUT)
    np.savez(dict_name + holo_file_name + '/SIGMA_HOLO_P', SIGMA_HOLO_CUT)

def svd_decompression(holo_file_name, k_value, pp, wlen, dist):
    print('SVD DECOMPRESSION ALGORITHM')
    # DECOMPRESSIONE
    # carica i file npz
    with np.load(dict_name + holo_file_name + '/U_HOLO_P.npz') as data:
        # ottieni tutti gli array presenti nel file
        U_COMPRESS = data['arr_0']

    with np.load(dict_name + holo_file_name + '/V_HOLO_P.npz') as data:
        # ottieni tutti gli array presenti nel file
        V_COMPRESS = data['arr_0']

    with np.load(dict_name + holo_file_name + '/SIGMA_HOLO_P.npz') as data:
        # ottieni tutti gli array presenti nel file
        SIGMA_COMPRESS = data['arr_0']

    print('Matrice U caricata: ', U_COMPRESS.shape)
    print('Matrice SIGMA caricata: ', SIGMA_COMPRESS.shape)
    print('Matrice V caricata: ', V_COMPRESS.shape)

    # matrice ricostruita
    matrix_rec = reconst_matrix(U_COMPRESS, SIGMA_COMPRESS, V_COMPRESS, 5, k_value+1, 5)
    # showHologram(matrix_rec, pp, wlen, dist, 'Ologramma decompresso', False)
    # np.savez(dict_name + holo_file_name + '/Matrix_RICOSTRUITA', matrix_rec)
    results(holo_file_name)

def results(holo_file_name):
    total_size_HOL_ORIGINAL = os.path.getsize(dict_name + holo_file_name + '/Matrix_HOLO.npz')
    _ , total_size_HOL_ORIGINAL_formatted = sizeof_fmt(total_size_HOL_ORIGINAL)
    print('Dimensione matrice originale: ', total_size_HOL_ORIGINAL)

    print('Dimensione matrice U compressa: ', os.path.getsize(dict_name + holo_file_name + '/U_HOLO_P.npz'))
    print('Dimensione matrice SIGMA compressa: ', os.path.getsize(dict_name + holo_file_name + '/SIGMA_HOLO_P.npz'))
    print('Dimensione matrice V compressa: ', os.path.getsize(dict_name + holo_file_name + '/V_HOLO_P.npz'))

    total_size_HOL_P = os.path.getsize(dict_name + holo_file_name + '/U_HOLO_P.npz') + os.path.getsize(dict_name + holo_file_name + '/V_HOLO_P.npz') + os.path.getsize(dict_name + holo_file_name + '/SIGMA_HOLO_P.npz')
    _ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_P)
    print('Somma dimensioni matrici compresse: ', total_size_HOL_P)
    rate(total_size_HOL_ORIGINAL,total_size_HOL_P)