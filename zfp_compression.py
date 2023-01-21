import os
import numpy as np
import zfpy

from HoloUtils import getComplex, hologramReconstruction, sizeof_fmt, rate

dict_name = 'fpzip_compression/'

def zfp_compression(holo, filename, pp, wlen, dist, rate):
    print('ZFP COMPRESSION ALGORITHM')
    if not os.path.isdir(dict_name + filename):
        os.makedirs(dict_name + filename)
    np.savez(dict_name + filename + '/matrix_HOLO', holo)
    # holog = holo
    # holog = holog[:, 420:]
    # holog = holog[:, :-420]
    # hologramReconstruction(holog,pp,dist,wlen)
    #Estraggo la matrice delle parti immaginarie e la matrice delle parti reali
    imagMatrix = np.imag(holo)
    realMatrix = np.real(holo)

    # Comprimo matrice immaginaria con zfpy
    compressed_data_imag = zfpy.compress_numpy(imagMatrix, rate=rate)
    # Comprimo matrice reale con zfpy
    compressed_data_real = zfpy.compress_numpy(realMatrix,rate=rate)

    with open(dict_name + filename + '/immaginaria_C.bin', 'wb') as f:
        f.write(compressed_data_imag)
    with open(dict_name + filename + '/reale_C.bin', 'wb') as f:
        f.write(compressed_data_real)

def zfp_decompression(filename, pp, wlen, dist):
    print('ZFP DECOMPRESSION ALGORITHM')
    with open(dict_name + filename + '/immaginaria_C.bin', 'rb') as f:
        compressed_data_imag = f.read()
    with open(dict_name + filename + '/reale_C.bin', 'rb') as f:
        compressed_data_real = f.read()

    decompressed_array_imag = zfpy.decompress_numpy(compressed_data_imag)
    decompressed_array_real = zfpy.decompress_numpy(compressed_data_real)

    # confirm lossy compression/decompression
    # np.testing.assert_allclose(realMatrix, decompressed_array_real, atol=1e-6)
    # confirm lossy compression/decompression
    # np.testing.assert_allclose(imagMatrix, decompressed_array_imag, atol=1e-6)

    #Ricostruzione della matrice
    complexMatrix = getComplex(decompressed_array_real, decompressed_array_imag)
    # holog = complexMatrix
    # holog = holog[:, 420:]
    # holog = holog[:, :-420]
    # hologramReconstruction(holog,pp,dist,wlen)
    total_size_HOL_NC = os.path.getsize(dict_name + filename + '/matrix_HOLO.npz')
    _ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_NC)
    print('NON COMPRESSA: ', total_size_HOL_P_formatted)

    total_size_HOL_C = os.path.getsize(dict_name + filename + '/immaginaria_C.bin') + os.path.getsize(dict_name + filename + '/reale_C.bin')
    _ , total_size_HOL_P_formatted = sizeof_fmt(total_size_HOL_C)
    print('COMPRESSA: ', total_size_HOL_P_formatted)

    rate(total_size_HOL_NC, total_size_HOL_C)













