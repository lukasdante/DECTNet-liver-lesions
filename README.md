# DECTNet
### DECTNet: Dual Encoder Network Combined Convolution and Transformer Architecture for Medical Image Segmentation

This is originally forked from the PyTorch implementation of DECTNet by LBL0704/DECTNet.

#### Implementation
This forked repository extends the use of DECTNet in multi-class focal liver lesion (FLL) image segmentation. The pretrained model of heart segmentation is used to initialize the weights and biases of the neural network. We improved the documentation by adding type-hinting and docstrings for each Python script. The pretrained model Codes/FLLs/multi_class.pt is duplicated from MMs/CaTNet_Weights/Best_Weights.pt.