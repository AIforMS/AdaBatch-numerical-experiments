AdaBatch

# Work
Our job is to reproduce this paper：《AdaBatch: adaptive batch sizes for training deep neural networks》
In our work, we use adabatch technology to train the neural network architecture only in Alexnet, Resnet-20 and VGG on cifar-10 and cifar-100 datasets. We have done a single nividia Tesla T4 GPU and three GPUs parallel training experiments.

# AdaBatch-numerical-experiments
Numerical experiments for 《AdaBatch: adaptive batch sizes for training deep neural networks》

========

Training deep neural networks with Stochastic Gradient Descent, or its
variants, requires careful choice of both learning rate and batch size.
While smaller batch sizes generally converge in fewer training epochs,
larger batch sizes offer more parallelism and hence better computational
efficiency. We have developed a new training approach that, rather than
statically choosing a single batch size for all epochs, adaptively
increases the batch size during the training process. 

Our method delivers the convergence rate of small batch sizes while
achieving performance similar to large batch sizes. We analyse our
approach using the standard AlexNet, ResNet, and VGG networks operating
on the popular CIFAR-10, CIFAR-100, and ImageNet datasets. Our results
demonstrate that learning with adaptive batch sizes can improve
performance by factors of up to 6.25 on 4 NVIDIA Tesla P100 GPUs while
changing accuracy by less than 1% relative to training with fixed batch
sizes.

Details can be found in our companion paper:

> A. Devarakonda, M. Naumov and M. Garland, "AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks", Technical Report, [ArXiv:1712.02029](https://arxiv.org/abs/1712.02029), December 2017. 


Implementation
--------------

**CIFAR**.  Our implementation of AdaBatch for the CIFAR-10 and
CIFAR-100 datasets is contained in:
**1.Single GPU**

    adabatch_cifar.py
  
**2.Multi GPU**
three nodes：
    cifar_dis.py --world-size 3 --rank 0
    cifar_dis.py --world-size 3 --rank 1
    cifar_dis.py --world-size 3 --rank 2

References
----------

A. Devarakonda, M. Naumov and M. Garland, "AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks", Technical Report, [ArXiv:1712.02029](https://arxiv.org/abs/1712.02029), December 2017. 
