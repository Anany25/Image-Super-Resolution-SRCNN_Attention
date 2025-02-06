# Super Resolution of Images using Deep Learning Techniques

## Project Details

This project explores the potential of deep learning techniques to enhance the resolution of low-quality images, making them sharper and more detailed. The objective is to apply advanced algorithms to improve image clarity in applications requiring high precision.

## Problem Statement

The project addresses the challenge posed by the abundance of poor-quality images due to various factors like suboptimal shooting conditions and compression losses. Traditional enhancement techniques often result in unsatisfactory resolution and clarity, necessitating the need for superior methods such as those based on deep learning.

## Dataset

The DIV2K dataset, comprising 2,000 high-resolution images, is used for training and testing our models. This dataset is selected for its diversity and relevance to real-world scenarios.

## Methodology

Our approach involves several stages:
1. **Bicubic interpolation** - Used as a preliminary upscaling method.
2. **Unsharp filtering** - Applied after bicubic interpolation to enhance image details.
3. **Super-Resolution Convolutional Neural Network (SRCNN)** - Employs a deep learning model to perform end-to-end image resolution enhancement.
4. **Enhanced SRCNN with Spatial Attention** - Integrates a spatial attention mechanism into SRCNN to focus on important areas within the image, improving detail reproduction.

## Results

The project's effectiveness is measured using Peak Signal-to-Noise Ratio (PSNR), with the enhanced SRCNN model demonstrating significant improvements over traditional methods and standard SRCNN.

## Conclusion

The integration of deep learning and attention mechanisms into super-resolution techniques significantly enhances image quality, surpassing traditional methods.

## References

For detailed academic references and methodologies, please refer to the literature listed in the project report.
