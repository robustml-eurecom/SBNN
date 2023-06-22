# SBNN
This is the pytorch implementation of the paper "A binary domain generalization for sparsifying binary neural networks", published in ECML PKDD 2023. Authors: R. Schiavone, F. Galati and M. A. Zuluaga.

![High level view](SBNN_convolution.jpg)

In this paper for sparsifying binary network 
 - We resort to entropy to reach sparsity
 - When using entropy the goal is to optimize the network to be largely skewed to one of the two possible weight values, i.e. a very low entropy.
 - This leads to a significant asymmetry in the distribution of the weight values
 - Representing this asymmetry using symmetric values used by standard binary networks is suboptimal since these use symmetric values
 - Thus we propose a more general binary domain ($\alpha$, $\beta$) that allows the weight values of the network to adapt to the asymmetry. With this we can capture that information and achieve a better representation.
