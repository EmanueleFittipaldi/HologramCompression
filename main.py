import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.fft


# Per scalare il plot dell'ologramma ricostruito
plt.rcParams["figure.figsize"] = (60, 60)


def main():
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

    # Prova di ricostruzione ologramma dalle due jpeg salvate

    # img = cv.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_reale.jpg', cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread('/Users/emanuelefittipaldi/PycharmProjects/HologramCompression/matrice_immaginaria.jpg', cv.IMREAD_GRAYSCALE)
    # ricostruzioneOlogramma(getComplex(img,img2), pp, dist, wlen)

    # Creazione delle immagini della matrice reale ed immaginaria

    # matriceReale = np.real(holo)
    # matriceImmaginaria = np.imag(holo)
    # matplotlib.image.imsave('matrice_reale.jpg', matriceReale, cmap='gray')
    # matplotlib.image.imsave('matrice_immaginaria.jpg', matriceImmaginaria, cmap='gray')

if __name__ == '__main__':
    main()
