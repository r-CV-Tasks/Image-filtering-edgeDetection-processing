# Image Filtering & Edge Detection Techniques
<h1 style="text-align: center;"> Image Processing Techniques</h1>

In this Repository we present a variety of Image processing Techniques implemented from scratch using Numpy and Pure Python. Each Category of Algorithms is presented in its tab in the UI which we will discover next. 

## Implementations Added:

1. Noise Functions (Simulation of Different Noise Types): Uniform, Gaussian and Salt & Pepper.
2. Edge Detection Techniques:  Prewitt, Sobel and Roberts.
3. Image Histogram Equalization and Normalization.
4. Local and Global Thresholding 
5. Transformation to Gray Scale
6. Frequency Domain Filters: Low Pass and High Pass Filters

In addition to histogram and distribution curve drawing for the loaded image and the option to mix 2 input images.

## Results:

Our UI present a tab for each category of the implemented algorithms. We first load our image and apply the selected algorithm.

1. Noise Addition : 

   1. 1. Uniform Noise

         ![image-20210405160220887](./src/1.png)

      2. Gaussian Noise

         ![image-20210405160825328](./src/2.png)

      3. Salt & Pepper

         ![image-20210405160909867](./src/3.png)

2. Noise Filtration: 

   1. Average Filter (Applied on Gaussian Noise)

      ![image-20210405161102172](./src/4.png)

   2. Gaussian Filter (Applied on Gaussian Noise)

      ![image-20210405161220897](./src/5.png)

   3. Median Filter (Applied on a Salt & Pepper Noisy Image)

      ![image-20210405161600194](./src/6.png)

3. Edge Detection Techniques: 

   1. 1. Sobel

         ![sobel](./src/sobel.png)

      2. Prewitt

         ![prewitt](./src/prewitt.png)

      3. Roberts

         ![roberts](./src/roberts.png)

   You can apply different SNR ratios and choose the Sigma of Each Algorithm implemented from the sliders added on the left, each cell is marked with its contents and the application of the change in the sliders is instant.

   ![image-20210405163353372](./src/ui.png)

4. Histogram Equalization with input and output histograms

   ![image-20210405163750243](./src/eq.png)

5. Local and Global Thresholding

   

   ![image-20210405163848013](./src/local.png)

![image-20210405163920015](./src/global.png)

6. Gray Scale Transformation 

   ![image-20210405164031568](./src/grayscale.png)

7. Frequency Domain Mixing 

   ![image-20210405164130406](./src/hybrid.png)