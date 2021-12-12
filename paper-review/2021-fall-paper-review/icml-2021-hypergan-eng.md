---
description: Ratzlaff et al. / HyperGAN - A Generative Model for Diverse, Performant Neural Networks / ICML 2019
---

#  HyperGAN \[Eng\]

## \(Start your manuscript from here\)


한국어로 쓰인 리뷰를 읽으려면 [**여기**](icml-2021-hypergan-kor.md)를 누르세요.

##  1. Problem definition

HyperGAN is a generative model for learning a distribution of neural network parameters. Specifically, weights of convolutional filters are generated with latent and mixer layers.

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.22%20PM.png)

## 2. Motivation & Related work

It is well known that it is possible to train deep neural networks from different random initializations. Also, ensembles of deep networks have been further studied that they have better performance and robustness. In Bayesian deep learning, learning posterior distributions over network parameters is a significant interest, and dropout is commonly used for Bayesian approximation. MC dropout was proposed as a simple way to estimate model uncertainty. However applying dropout to every layer may lead to underfitting of the data and it integrates over the space of models reachable from only a single initialization. 

As another interesting direction, hypernetworks are neural networks which output parameters for a target neural network. Hypernetwork and the target network together form a single model which is trained jointly. However, prior hypernetworks relied on normalizing flow to produce posteriors, which limited their scalability. 

This work explore an approach which generates all the parameters of a neural network in a single pass, without assuming any fixed noise models or functional form of the generating function. Instead of using flow-based models, authors utilize GANs. This method results in models more diverse than training with multiple random starts(ensembles) or past variational methods.


### Idea

The idea of HyperGAN is to utilize a GAN-type approach to directly model weights. However, this would require a large set of trained model parameters as training data. So the authors take another approach, they directly optimize the target supervised learning objective. This method can be more flexible than using normalized flows, and also computationally efficient because parameters of each layer is generated in parallel. Also compared to ensemble model, which has to train many models, it is both computationally and memory efficient.

## 3. Method

Above figure in introduction section shows the HyperGAN architecture.
Distinct from the standard GAN, authors propose a *Mixer* Q which is a fully-connected network that maps s ~ *S* to a mixed latent space Z. The mixer is motivated by the observation that weight parameters between network layers must be strongly correlated as the output of one layer needs to be the input to the next one. So it produces *Nd* - dimensional mixed latent vector in mixed latent space *Q*(z|s), which is all correlated. Latent vector is partitioned to *N* layer embeddings, each being a *d*-dimensional vector. Finally, *N* parallel generators produce parameters for each of the N layers. This mehtod is also memory efficient since the extremely high dimensional space of the weight parameters are now separately connected to multiple latent vectors, instead of fully-connected to the latent space.

Now the new model is evalutated on the training set and generated parameters are optimized with respect to loss *L*:

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.36%20PM.png)

However, it is possible that codes sampled from *Q*(z|s) may collapse to the maximum likelihood estimate (MLE). To prevent this, authors add an adversarial constraint on the mixed latent space and require it to not deviate too much from a high entropy prior *P*. So this leads to the HyperGAN objective:

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.44%20PM.png)

*D* could be any distance function between two distributions in practice. Here, discriminator network together with adversarial loss is used to approximate the distance function.


![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%208.39.57%20PM.png)

Since it is difficult to learn a discriminator in the high dimensional space and as there is no structure in those parameters(unlike images), regularizing in the latent space works well.



## 4. Experiment & Result

### Experimental setup

- classification performance on MNIST and CIFAR-10
- learning variance of a simple 1D dataset
- Anomaly detection of out-of-distribution examples
  - Model trained on MNIST / tested with notMNIST
  - Model trained on CIFAR-10 5 classes / tested on rest of 5 classes

- baselines
  - APD(Wang et al., 2018), MNF(Louizos & Welling, 2016), MC Dropout(Gal & Ghahramani, 2016)


### Result

#### Classification result

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.18.17%20PM.png)

#### Anomaly detection result

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.18.38%20PM.png)

### Ablation Study

First, removing regularization term *D*(Q(s), *P*) from the objective reduces the network diversity. Authors measure L2 norm of 100 weight samples and divide their standard deviation by the mean. Also, authors examine that the diversity decreases over time, so they suggest early stopping of the training. Next the authors remove the mixer *Q*. While the accuracy is retained, diversity suffers significantly. Authors hypothesize that without the mixer, a valid optimization is difficult to find. With the mixer, the built-in correlation between the parameters of different layers may have made optimization easier.

## 5. Conclusion

In conclusion, HyperGAN is a great mehtod to build an ensemble models that are highly robust and reliable. It has strength in that it is able to generate parameters with GAN method, without mode collapse using mixer network and regularization terms. However, has weakness that the work was only built with small target networks with small datasets like MNIST and CIFAR10, performing a simple classification tasks. It would be more interesting if the work can be done on large networks like ResNets, training with larger datasets.

### Take home message \(오늘의 교훈\)

Hypernetworks can be trained using GAN to build bayesian neural networks.

## Author / Reviewer information

### Author

**형준하 (Junha Hyung)**
* KAIST graduate school of AI (M.S.)
* Research Area: Computer Vision
* sharpeeee@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

[[1]](https://arxiv.org/abs/1609.09106)Ha, D., Dai, A. M., and Le, Q. V. Hypernetworks. CoRR

[[2]](http://bayesiandeeplearning.org/2018/papers/121.pdf)Henning, C., von Oswald, J., Sacramento, J., Surace, S. C., Pfister, J.P., and Grewe, B. F. Approximating the predic- tive distribution via adversarially-trained hypernetworks

[[3]](https://arxiv.org/abs/1710.04759)Krueger, D., Huang, C.W., Islam, R., Turner, R., Lacoste, A., and Courville, A. Bayesian Hypernetworks

[[4]](https://arxiv.org/abs/1802.09419)Lorraine, J. and Duvenaud, D. Stochastic hyperparameter optimization through hypernetworks. CoRR

[[5]](https://arxiv.org/abs/1711.01297)Pawlowski, N., Brock, A., Lee, M. C., Rajchl, M., and Glocker, B. Implicit weight uncertainty in neural networks

