# HologramCompression

## Abstract
In this paper the problem of compressing computer generated holograms (CGH) is explored using four different methods. 
- The first method used is image-based compression, in which the matrix of complex numbers representing the hologram is decomposed into a matrix containing the real part and a matrix containing the imaginary part. These two arrays are interpreted as images and compressed using a standard image compression algorithm. 
- The second method used is compression based on the Zfp library, which uses a compression algorithm specially designed for scientific data in floating point format. 
- The third method is compression based on the application of SVD (Singular Value Decomposition), which analyzes the hologram matrix and uses only the most important singular values to reconstruct it, thus eliminating unnecessary data. 
- Finally, the fourth method used is wavelet-based compression, which uses a wave-based transformation algorithm to analyze and compress the hologram data.

The results show that the SVD and Wavelet methods are the best, with compression ranging from 97% to 99%, with Wavelet being slightly more effective. Furthermore, the paper provides a comparative evaluation of the various compression methods and offers indications for further research in this field. In particular, it might be interesting to explore the combined use of multiple compression methods to achieve greater efficiency in compressing computer-generated holograms.

## To run this Project:
- Clone the repository from Pycharm
- Install all the libraries you need
- Create a launch configuration for each different compression method, so that you can run them individually.
- Run to verify that everything works

## Contribution
The project is OpenSource, we are open to any kind of contribution
