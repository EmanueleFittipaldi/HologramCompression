import cv2

def bitPlaneSlicing (imgPath):
    # Carichiamo l'immagine in scala di grigi
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    # Creiamo un array di 8 elementi con i valori 1, 2, 4, 8, 16, 32, 64, 128
    masks = [1, 2, 4, 8, 16, 32, 64, 128]

    # Iteriamo su ogni maschera
    for mask in masks:
        # Estraggo il bitplane corrispondente alla maschera
        bitplane = cv2.bitwise_and(img, mask)

        # Mostriamo l'immagine estratta
        cv2.imshow('Bitplane {}'.format(mask), bitplane)

    # Aspettiamo che l'utente premi un tasto per chiudere le finestre
    cv2.waitKey(0)
    cv2.destroyAllWindows()