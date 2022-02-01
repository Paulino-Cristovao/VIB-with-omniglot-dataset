# VIB - imprinted-weights
This work introduces the variational Information Bottleneck (VIB) loss function for imprinted models. We find that this objective function gives us relevant information for generalizing on novel classes. We intend to show that cross entropy learning objective alone is not enough for Few-shot learning.

We trained the imprinted model with VIB. The dataset used is omniglot. We tested on digits [MNIST] as novel classes.
Save the data in "data_omniglot" and for mnist "data". We used a csv files.
We give credit to https://github.com/YU1ut. 
Parts of the code is taking from his repository.

## Requirements
- Python 3.5+
- PyTorch 0.4.1
- torchvision
- pandas (0.23.4)
- progress
- matplotlib
- numpy

## References
- [1]: H. Qi, M. Brown and D. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.
- [2]: A. A. Alemi et al. "Deep Variational Information Bottleneck" ICLR, 2017
