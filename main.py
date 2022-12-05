import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math as math
import scipy.fft
import confresnel1D
import fourierfresnel1D
import fpzip
from givensphaserot import givenphaserot

#Per scalare il plot dell'ologramma ricostruito
plt.rcParams["figure.figsize"] = (60,60)

#Queste due funzioni servono per ricostruire l'ologramma.
def intfresnel2D(x, fw, pp, z, wlen, algo):
    # aggiungere un assert per controllare se l'array x non contiene zero
    assert (np.size(x) % 2) == 0, "La dimensione dell'ologramma deve essere pari"
    assert np.size(x, 0) == np.size(x, 1), "L'ologramma deve essere quadrato (da risolvere nelle prossime versioni)"

    def fw_fresnel(r):
        return np.round(fpropfun(r, pp, fz, wlen))

    def bw_fresnel(r):
        return np.round(bpropfun(r, pp, bz, wlen))

    def fwdfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, True)

    def revfourierfresnel1D(x, pp, z, wlen):
        return fourierfresnel1D.fourierfresnel1D(x, pp, z, wlen, False)

    def apply_transform():
        if (fw):
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

    if algo == "conv":
        fpropfun = confresnel1D.convfresnel1D  # convfresnel1D(x,pp,z,wlen)
        bpropfun = confresnel1D.convfresnel1D  # convfresnel1D(x,pp,z,wlen)
        fz = z
        bz = -z
    elif algo == "four":
        fpropfun = fwdfourierfresnel1D
        fz = z
        bpropfun = revfourierfresnel1D
        bz = z
    else:
        print("algoritmo sconosciuto")

    for i in [-1, 1]:
        if fw:
            x = np.rot90(x, i)
        x = apply_transform()
        if not fw:
            x = np.rot90(x, i)
    return x


def fourier_phase_multiply(r, fw, pp, z, wlen):
    n = np.size(r, 0)
    xx = np.power(np.matrix(np.arange(-n / 2, n / 2)), 2)
    ppout = wlen * np.abs(z) / n / pp

    temp = (math.pi * np.power(ppout, 2)) /((wlen * np.abs(z)))
    p=np.multiply(temp, (xx + xx.transpose()))+ (2 * np.pi * z )/ wlen
    return givenphaserot(r,p, fw)


def main():
    f = scipy.io.loadmat('Hol_2D_dice.mat')  # aprire il file .mat
    #f = scipy.io.loadmat('exampleholo.mat')  # aprire il file .mat

    #Dice Parameters
    pp = 8e-6  # pixel pitch
    pp = np.matrix(pp)
    wlen = 632.8e-9  # wavelenght
    wlen = np.matrix(wlen)
    dist = 9e-1  # propogation depth
    dist = np.matrix(dist)

    #Letter Parameter
    #f = scipy.io.loadmat('exampleholo.mat')  # aprire il file .mat
    #pp = 8e-6  # pixel pitch
    #pp = np.matrix(pp)
    #wlen = 532e-9  # wavelenght
    #wlen = np.matrix(wlen)
    #dist = 5e-1  # propogation depth
    #dist = np.matrix(dist)

    #Venus Parameters
    #pp = 8e-6  # pixel pitch
    #pp = np.matrix(pp)
    #wlen = 632.8e-9  # wavelenght
    #wlen = np.matrix(wlen)
    #dist = 5e-1  # propogation depth
    #dist = np.matrix(dist)

    #Cat Parameters
    #pp = 2e-6  # pixel pitch
    #pp = np.matrix(pp)
    #wlen = 632.8e-9  # wavelenght
    #wlen = np.matrix(wlen)
    #dist = 1e-1  # propogation depth
    #dist = np.matrix(dist)


    #holo è la matrice di numeri complessi
    holo = np.matrix(f['Hol'])

    #Prova con fpzip
    #data = np.imag(holo)
    #compressed_bytes = fzip.compress(data,precision=0,order='C')
    #print(compressed_bytes)


    #holo = np.matrix(f['holo'])

    #Per gli ologrammi 1920*1080
    holo = holo[:,420:]
    holo = holo[:,:-420]

    #np.save("matriceComplessaCsingle.npy", np.csingle(holo))
    #np.save("matriceImmaginaria.npy",np.imag(holo))
    #np.save("matriceReale.npy",np.real(holo))


    #Per visualizzare la parte reale ed immaginaria dell'ologramma
    #plt.imshow(np.imag(holo), cmap="gray")
    #plt.show()  # mostro l'ologramma pre - compressione

    #plt.imshow(np.real(holo), cmap="gray")
    #plt.show()

    #Scegliere l'implementazione di Fresnel
    #algo='conv'
    algo = 'four'

    #Ricostruzione dell'ologramma tramite diffrazione di Fresnel
    t = intfresnel2D(np.csingle(holo), False, pp, dist, wlen, algo)
    t = fourier_phase_multiply(t, False, pp, dist, wlen)
    plt.imshow(np.imag(t), cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
    # prova()
