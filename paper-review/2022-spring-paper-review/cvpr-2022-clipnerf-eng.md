---
description: Can Wang et al. / CLIP-NeRF; Text-and-Image Driven Manipulation of Neural Radiance Fields / CVPR 2022
---

# CLIP-NeRF \[Eng\]
한국어로 쓰인 리뷰를 읽으려면 [**여기**](./cvpr-2022-clipnerf-kor.md)를 누르세요.



## 1. Introduction
Recently, [Neural Radiance Fields(NeRF, ECCV'20)](https://arxiv.org/abs/2003.08934) has received great attention in the field of view synthesis. NeRF is a fully-connected neural network that can generate novel views of complex 3D scenes by being trained based on a partial set of 2D images. Meanwhile, [Contrastive Language-Image Pre-Training(CLIP, ICML'21)](https://arxiv.org/abs/2103.00020) suggested a multi-modal neural network, which maps both text and image to the same latent space, and learned the similary between text and image using large-capacity (text, image) pairs. This paper made it possible to manipulate images using text in many subsequent studies. In this article, I'm going to review a paper [CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(CVPR'22)](https://arxiv.org/abs/2112.05139). This paper combines the methodology of NeRF and CLIP and suggests a way to manipulate synthesized views of NeRF with only prompt texts or exemplar images.

### Problem Definition
The problem to solve in this paper is to implement how to manipulate NeRF with a text prompt or a single exemplar image. Specifically, the authors has changed the shape of the NeRF output by applying conditional shape deformation to the positional encoding of NeRF, and the color of the NeRF output by appling deferred appearance conditioning to the volumetric rendering stage of NeRF. Also, they combined the NeRF model with pre-trained CLIP model and proposed an overall architecture that can manipulate the shape or color of NeRF output by just manipulating the shape code and the appearance code with prompt text or exemplar image.



## 2. Motivation
Since CLIP-NeRF introduces a method of manipulating the NeRF output by combining NeRF and CLIP methodologies, I will introduce NeRF and CLIP first, and then let you know the details of CLIP-NeRF.

### Related work
#### NeRF

![Figure 1. NeRF Overview](../../.gitbook/assets/2022spring/49/nerf_overview.png)

View synthesis is a way to train how to take pictures of 3D objects or scenes from different view points and generate pictures of the objects from any new view point. NeRF, which uses volume rendering to carry out neural implicit presentation, enables high-quality view synthesis. NeRF is a fully-connected neural network which takes the specific position $$(x, y, z)$$ of the 3D scene and view point $$(\theta, \phi)$$ and returns the color $$c=(r, g, b)$$, and the volume density $$\sigma$$, which represents the degree of light reflection at that position. Since volume density is a unique characteristic that is determined by the type of substance present in a particular position, it should have a view point-independent value. However, the color emitted from each position may vary depending on the view point. The phenomenon that the color of an object changes according to the view point is called the non-Lambertian effect.

![Figure 2. Non-Lambertian Effect](../../.gitbook/assets/2022spring/49/non_lambertian_effect.png)

If we train NeRF properly, we can obtain the color and the volume density for all points in a 3D scene at a specific view point. After achieving this, 2D rendering can be created using the classical volume rendering method.

##### classical volume rendering
If we shoot the camera ray in the direction $$\mathbf{d}$$ at a particular point $$\mathbf{o}$$, then the trajectory of this ray can be represented by the equation of a straight line $$\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$$. If the range of $$ t $$ where the trajectory of this camera ray meets the 3D scene is $$ [t_n, t_f] $$, then the color $$ C(\mathbf{r}) $$ of the 3D scene observed at $$ \mathbf{o} $$ is expressed as follows:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f}{T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})}dt,

\\~\text{where}~ T(t) = \exp\Big(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))ds\Big).
$$

An intuitive interpretation of this is that we can get colors from $$\mathbf{r}(t_n)$$ to $$\mathbf{r}(t_f)$$ of the 3D scene at view point $$\mathbf{d}$$ from NeRF, and can obtain the final color by integrating them. At this point, $$T(t)\sigma(\mathbf{r}(t),\mathbf{d})$$ multiplied before $$\mathbf{r}(t)$$ acts as weight. If there are many opaque objects in front of the object in the current location, the amount of the object in the current location contributes to the final color will be reduced. $$T(t)$$ is a reflection of this and represents the volume density accumulated to date. If the volume density accumulated to date is large, then $$\int_{t_n}^{t}\sigma(\mathbf{r}(s))ds$$ will be smaller, and thus $$T(t)$$ will be smaller. Eventually, the amount of contribution the current location will make to the final color will be reduced.

The amount contributed to the final color will also be proportional to the volume density $$\sigma(\mathbf{r}(t))$$ at a particular point. Multiplied by these two elements, $$T(t)\sigma(\mathbf{r}(t))$$ becomes the weight at a particular point. The principle of NeRF view synthesis is that an accumulated color of specific pixel of 2D view images can be calculated by shooting a camera ray in a specific direction, and the whole 2D image can be calculated by shooting camera rays repeatedly in multiple directions.

##### hierarchical volume sampling
NeRF calculates the above integral in a numerical way through a sampling method. Specifically, $$[t_n, t_f]$$ is divided into $$N$$ uniform intervals and sampling is done from the uniform distribution in each interval to estimate the color and volume density $$\hat{C}_c(\mathbf{r})$$ of coarse network. The fine network $$\hat{C}_f(\mathbf{r})$$, which estimates the color and volume density, is learned by performing inverse transform sampling proportional to the volume density of each interval calculated from the coarse network. This hierarchical volume sampling enables importance sampling where there are many samples in areas that are heavily involved in final color calculation.

##### architecture
The specific architecture of NeRF is as follows. NeRF $$F_{\theta}$$ is a MLP based deep neural network. We pass the 3D coordinate through the 8 fully-connected layers(ReLU activation, 256 channels per layer) to obtain the volume density and 256-dimensional feature vector. Then we concatenate the returned feature vector and view point, and pass them through the two layers behind to obtain the final color. To make the volume density independent of view point, we can see that the view point $$\mathbf{d}$$ was added after obtaining the volume density $$\sigma$$ value from the network architecture.

![Figure 3. NeRF Architecture](../../.gitbook/assets/2022spring/49/nerf_architecture.png)

##### positional encoding
The authors of NeRF confirmed that giving location information and view points directly into NeRF is not suitable for expressing the fast-changing part of an object in a 3D scene. To solve this problem, they introduced a method of mapping location information and view point to higher dimensional space using high frequency function. The authors used a positional encoding method similar to that in the transformer. This means that $$F_{\Theta}$$ is represented by $$F_{\theta}' \circ \gamma$$ where $$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \cdots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$$ and they independently applied $$\gamma$$ to normalized position $$\mathbf{x} = (x, y, z)$$ and normalized viewing director unit vector $$\mathbf{d} = (\sin\theta \cos\phi, \sin\theta\sin\phi, \cos\theta)$$. They achieved the improvement of performance by this way.

##### loss function
Training is done using photographs taken from various view points in a single 3D scene. The loss function is as follows.

$$
\mathcal{L} = \sum\limits_{\mathbf{r} \in \mathcal{R}} \Big[ ||\hat{C}_c(\mathbf{r}) - C(\mathbf{r}) ||_2^2 + ||\hat{C}_f(\mathbf{r}) - C(\mathbf{r}) ||_2^2 \Big]
$$

This is pixel-wise $$L_2$$ loss, which allows the coarse network and fine network to generate scenes that are close to the actual scene taken at each view point.

#### CLIP

![Figure 4. CLIP Overview](../../.gitbook/assets/2022spring/49/clip_overview.png)

Contrastive Language-Image Pretraining(CLIP) is a proposed model to overcome the limitations of traditional image classification models: 1) large amounts of labeled data are required for training and 2) individual training is required for specific task. CLIP first collects many (text, image) pairs that exist on web pages like Wikipedia without requiring separate labeling process. Then, pre-training is performed to predict which text is attached to which image using 400 million (text, image) pairs collected. Pre-trained CLIP can predict the similarity between text and images as cosine similarity.

Given $$\{(T_n, I_n) \}_{n=1}^{N}$$ of (text, image) pairs collected within mini-batch, text $$T_n$$ and image $$I_n$$ are passed through text encoder $$f_t$$(in transformer format) and image encoder $$f_i$$(in ResNet or vision transformer format), and then embedded to feature space $$f_t(T_n) \in \mathbb{R}^{d_t}$$, $$f_i(I_n) \in \mathbb{R}^{d_i}$$, respectively.

The embedding vectors are multiplied by linear projection matrix $$W_t \in \mathbb{R}^{d_e \times d_t}$$, $$W_i \in \mathbb{R}^{d_e \times d_i}$$, respectively and we get final embedding vectors of text and image in same embedding space. The pairwise cosine similarity matrix $$S_{nm}(1 \le n, m \le N)$$ between text and images within the mini-batch is as follows:

$$
S_{nm} = \frac{\langle W_t f_t(T_n), W_i f_i(I_m) \rangle}{|| W_t f_t(T_n) ||_2 \cdot || W_i f_i(I_m) ||_2} \times \exp(t)
$$

Here, $$t$$ is a learnable temperature parameter. After that, we proceed with training using the following symmetric cross-entropy loss.

$$
\mathcal{L} = \frac{1}{2} \Big[ \sum_{n=1}^{N} \frac{\exp(S_{nn})}{\sum_{m=1}^{N} \exp(S_{nm})} + \sum_{n=1}^{N} \frac{\exp(S_{nn})}{\sum_{m=1}^{N} \exp(S_{mn})} \Big]
$$

This is the loss for the pairwise cosine similarity matrix $$S_{nm}$$ to maximize the cosine similarity of already correlated (text, image) pairs, $$S_{nn}(1 \le n \le N)$$. In the figure above, (1) Contrastive Pre-training corresponds to this process. CLIP trained in this way allows us to calculate the similarity between text and image as cosine similarity, so we can perform zero-shot image classification. For example, given a particular image, each category label can be given as an input text to determine which category has the highest similarity, and the category with the highest cosine similarity can be predicted as the category of the image category. No matter what category label comes in, cosine similarity can be calculated, so there is no need to train CLIP by task. In the figure above, part (2) Create dataset classifier from label text, (3) Use for zero-shot prediction corresponds to this process.

#### NeRF Editing
NeRF, which can generate high-quality views from 3D scenes received great attention and led to many subsequent studies. Subsequent studies included DietNeRF, GRAF, which extend NeRF, such as applying NeRF to dynamic scenes rather than one fixed scene, or applying it to relighting or generative models. These follow-up studies have led to advances, but there is a problem that it is difficult to intuitively manipulate the NeRF output because it still consists of millions of network parameters.

In order to address this problem, EditNeRF defined conditional NeRF, a structure that separates the 3D objects encoded with NeRF into shape code and color appearance code. By manipulating two latent codes, the user can be able to control the shape and color of the object. However, EditNeRF was only able to manipulate or erase certain parts of the object, and the speed was very slow. Compared to EditNeRF, CLIP-NeRF has the advantage of 1) being able to manipulate the overall shape of an object, 2) training two latent mappers to improve the reference speed, and 3) being able to easily manipulate the results of the NeRF using short prompt texts or exemplar images.

#### CLIP-Driven Iamge Generation and Manipulation
CLIP, as described above, calculates the cosine similarity score of similar text and image in the shared latent space. With the help of the CLIP model, several methods like StyleCLIP, and DiffusionCLIP have been proposed which can manipulate images using text. These methods can manipulate images only using text, while in this study, the results of NeRF can be manipulated by both text and image.

### Idea
In this paper, the authors introduced an intuitive way to manipulate NeRF with simple text prompt or single reference image. This study was conducted in a disentangled conditional NeRF structure that disentangles the latent space of the shape code that can manipulate the shape of objects and the appearance code that can control the color of objects.

In addition, they trained two code mappers using the CLIP model, which was a way to map CLIP features into a latent space to manipulate shape code and appearance code. In other words, when a prompt text or an exemplar image is given, it is used to change the shape or color of an object! After receiving the text prompt or exemplar image as condition, the feature vector was extracted using the pre-trained CLIP model, and it was put into the code mappers to create a local displacement on the late space of NeRF and therefore manipulate the shape code and the appearance code. For training, CLIP-based loss was designed to improve the consistency between input text(or image) and output rendering, and to enable high-resolution NeRF manipulation. In addition, they also proposed an inversion method of extracting shape code, appearance code, and view point from real image.

The main contribution of this paper is as follows:
- For the first time, text-and-image driven manipulation for NeRF has been proposed so that the users can manipulate 3D content with text or image.
- They proposed a disentangled conditional NeRF architecture that manipulates the shape of the object with the shape code and the color with the appearance code.
- They proposed a feedforward code mapper that performs faster than the optimization-based deformation method.
- They propose an inverse optimization method, which is a method of extracting shape code, appearance code, and view point from the image.



## 3. Method

The authors in the paper introduce the method part in following order: the general definition of conditional NeRF $$\rightarrow$$ disentangled conditional NeRF $$\rightarrow$$ CLIP-Driven Manipulation $$\rightarrow$$ Training Strategy $$\rightarrow$$ Disentangled Conditional NeRF. I think this is a reasonable way, so I will explain the method part in the same order.

### General Definition of Conditional NeRF
Based on original NeRF, Conditional NeRF can create objects by changing shapes and colors within a particular category by manipulating latent vectors that control shapes and colors, rather than creating just one 3D view. This is similar to conditional GAN, which can generate the desired number within MNIST dataset by giving digit label as condition. Conditional NeRF is a continuous volumetric function $$\mathcal{F}_{\Theta}:(x,y,z, \phi, \theta, z_s, z_a) \rightarrow (r, g, b, \sigma)$$, which not only receives a specific location $$\mathbf{x}(x, y, z)$$ and a view point $$v(\theta, \phi)$$ of a 3D scene, but also receives a shape code $$z_s$$ and an appearance code $$z_a$$ to specify the shape and the color of the scene, and returns the emitted color $$c = (r, g, b)$$ and volume density $$\sigma$$ at that point. Trivial formulation $$\mathcal{F}_{\theta}'(\cdot)$$ of conditional NeRF, which simply concatenates shape code to an existing location and concatenates appearance code to an existing view point, is shown below.

$$
\mathcal{F}_{\theta}'(\mathbf{x}, v, z_s, z_a) : (\Gamma(\mathbf{x}) \oplus z_s, \Gamma(v) \oplus z_a) \rightarrow (c, \sigma)
$$

In this case, $$\oplus$$ is the concatenation operator, $$\Gamma(\bold{p}) = \{ \gamma(p) | p \in \bold{p} \}$$ is the sinusoidal positional encoding which maps $$x$$, $$y$$, $$z$$ in $$\bold{p}$$ to high dimensional space. $$\gamma(\cdot): \mathbb{R} \rightarrow \mathbb{R}^{2m}$$ is defined as follows:
$$
\gamma(p)_k = \begin{cases}
\sin(2^k \pi p) & \text{if k is even}\\
\cos(2^k \pi p) & \text{if k is odd}\\
\end{cases}
$$
where $$k \in \{ 0, \cdots, 2m-1 \}$$, and $$m$$ is a hyperparameter.

### Disentangled Conditional NeRF
Conditional NeRF improves the architecture of NeRF, allowing to create a scene by manipulating its shape and color. Aforementioned trivial conditional NeRF $$\mathcal{F}_{\theta}'(\cdot)$$ has a problem of interference between the shape code and the color code. For example, manipulating the shape code to change shape also leads to the change of the color. This is because the shape code and the appearance code are not completely disentangled. To address this problem, this work proposes a disentangled conditional NeRF structure. This allows individual manipulation of shape and color. In order to implement the disentangled conditional NeRF, conditional shape deformation and deferred appearance conditioning were proposed. This corresponds to the Disentangled Conditional NeRF portion of the CLIP-NeRF figure above.

#### Conditional Shape Deformation
In trivial conditional NeRF, the shape code is directly concatenated to positional encoding. However, in disentangled conditional NeRF, the input position was slightly changed using the shape code. In this way, the shape code and the color code are completely disentangled. To do this, they introduce the shape deformation network $$\mathcal{T} : (\mathbf{x}, z_s) \rightarrow \Delta \mathbf{x}$$, which maps $$\mathbf{x}$$ and $$z_s$$ to a displacement vector $$\Delta \mathbf{x} \in \mathbb{R}^{3 \times 2m}$$ in the positional encoding $$\Gamma(\mathbf{x})$$. This slightly changes each element of positional encoding as much as a displacement vector by mapping it to. The deformed positional encoding is $$\Gamma^*(\mathbf{p}, z_s) = \{ \gamma(p)_k + \tanh(\Delta p_k) | p \in \mathbf{p}, \Delta p \in \mathcal{T}(p, z_s) \}$$. At this time, $p$ is scalar, $$\Delta p \in \mathbb{R}^{2m}$$, and $\tanh$ limits the displacement range to $$[-1, 1]$$ in order to prevent the encoding from changing too much. In summary, the shape code is not simply concatenated to the positional encoding, but the positional encoding is transformed through $$\mathcal{T}$$, which tells how much the positional encoding should change when a position and a shape code are given.

#### Deferred Appearance Conditioning
In order to make the volume density have a value independent of the view point, the view point $$\mathbf{d}$$ was given to the neural network after the volume density was calculated in NeRF. Likewise, if the appearance code is given to the neural network after obtaining the volume density, the appearance code cannot affect the volume density. In this way, the appearance code can manipulate the color without affecting the appearance at all.

Overall, the disentangled conditional NeRF $$\mathcal{F}_{\theta}(\cdot)$$ is defined as
$$
\mathcal{F}_{\theta}(\bold{x}, v, z_s, z_a) : (\Gamma^*(\bold{x}, z_s), \Gamma(v) \oplus z_a) \rightarrow (c, \sigma)
$$
For convenience, let $$\mathcal{F}_\theta (v, z_s, z_a) = \{ \mathcal{F}_\theta(\bold{x}, v, z_s, z_a) | \bold{x} \in \mathbf{R} \}$$ be a 2D rendered image of whole 3D view from a view point $$v$$.

### CLIP-Driven Manipulation
Using the above disentangled conditional NeRF as a baseline generator and linking it with CLIP, we can manipulate the output of NeRF based on the text. For example, given an input text prompt $$\mathbf{t}$$ and initial shape/appearance code $$z_s' / z_a'$$, if we train shape mapper $\mathcal{M}_s$ and appearance mapper $\mathcal{M}_a$ which maps the input text prompt $$\mathbf{t}$$ to the displacement of shape code and appearance code $$\mathcal{M}_s(\hat{\mathcal{E}}_{t}(\mathbf{t})) / \mathcal{M}_a(\hat{\mathcal{E}}_{t}(\mathbf{t}))$$, two codes can be appropriately manipulated to $$z_s / z_a$$, and therefore the shape and the color of the disentangled conditional NeRF output can also be manipulated appropriately.
$$
z_s = \mathcal{M}_s(\hat{\mathcal{E}}_{t}(\mathbf{t})) + z_s'\\
z_a = \mathcal{M}_a(\hat{\mathcal{E}}_{t}(\mathbf{t})) + z_a'
$$

In this case, $$\hat{\mathcal{E}}_t(\cdot)$$ is the pre-trained CLIP text encoder, and the shape mapper and appearance mapper map CLIP embeddings to displacement vectors of shape code and appearance code, respectively. Same method can be applied to pre-trained CLIP image encoder $$\hat{\mathcal{E}}_i(\cdot)$$. Through this, we can change the existing shape code and appearance code as above through a text prompt or exemplar image.

To learn the shape mapper and the appearance mapper, we need to change the shape code and the appearance code and then calculate the embedding similarity between the rendered image patch and the input text to maximize it. For this, the cross-modal CLIP distance function $$D_{\text{CLIP}}(\cdot, \cdot) = 1 - \langle \hat{\mathcal{E}}_i(\mathbf{I}), \hat{\mathcal{E}}_t(\mathbf{t}) \rangle$$ is defined. Here, $$\hat{\mathcal{E}}_i$$ and $$\hat{\mathcal{E}}_t$$ are pre-trained CLIP image and text encoders, $$\mathbf{I}$$ and $$\mathbf{t}$$ are the image and text to compute similarity, and $$\langle \cdot, \cdot \rangle$$ is the cosine similarity operator. This corresponds to the CLIP-Driven Manipulation part of the CLIP-NeRF figure above.

### Training Strategy
CLIP-NeRF is trained in two stages for stability. First, disentangled conditional NeRF is trained to work well without considering about the compatibility with CLIP. Next, the shape mapper and the appearance mapper are trained so that the given text or image in CLIP can transform the shape code and the appearance code well to obtain a natural NeRF result with high cosine silimarity with the given text or image. Hat above the symbol means that the symbol is fixed during training.

#### Training Disentangled Conditional NeRF
The disentangled conditional NeRF generator $$\mathcal{F}_{\theta}$$ uses a non-saturating GAN loss $$f(x) = -\log(1 + \exp(-x))$$, and it is trained with the discriminator $$\mathcal{D}$$. The generator and the discriminator are trained while competing with each other through the adversarial training process.

Assuming that real images $$\mathbf{I}$$ constitutes training data distribution $$d$$, we sample a shape code $$z_s$$, an appearance code $$z_a$$, and a camera pose $$v$$ from $$\mathcal{Z}_s$$, $$\mathcal{Z}_a$$, $$\mathcal{Z}_v$$, respectively. $$\mathcal{Z}_s$$, $$\mathcal{Z}_a$$ are normal distribution, and $$\mathcal{Z}_v$$ is the uniform distribution in the northern hemisphere of the camera coordinate system. The training loss is as follows

$$
\mathcal{L}_{\text{GAN}} = \mathbb{E}_{z_s \sim \mathcal{Z}_s, z_a \sim \mathcal{Z}_a, v \sim \mathcal{Z}_v}[f(\mathcal{D(\mathcal{F}_{\theta}(v, z_s, z_a)))}] + \\ \mathbb{E}_{\mathbf{I} \sim d}[f(-\mathcal{D}(\mathbf{I}) + \lambda_r || \nabla \mathcal{D}(\mathbf{I}) ||)^2].
$$

The disentangled conditional NeRF generator is trained to fool the discriminator to produce a 2D rendering similar to the training data distribution as much as possible by maximizing the loss above, and the discriminator is trained to determine the generated 2D rendering is fake, and the actual 2D images in the training data distribution are real by minimizing the loss above. $$\lambda_r$$ is the weight of the regularization term for the stability of the discriminator. This corresponds to the Training Strategy part of the CLIP-NeRF figure above.

#### Training CLIP Manipulation Mappers
Pre-traiend and fixed NeRF generator $$\mathcal{F}_{\theta}$$, CLIP text encoder and image encoder $$\{\hat{\mathcal{E}}_t, \hat{\mathcal{E}}_i \}$$, and discriminator $$\mathcal{D}$$ from previous training step are used to train shape mapper $$\mathcal{M}_s$$ and appearance mapper $$\mathcal{M}_a$$. As in training disentangled conditional NeRF, shape code $$z_s$$, appearance code $$z_a$$, camera pose $$v$$ are randomly sampled from $$\mathcal{Z}_s$$, $$\mathcal{Z}_a$$, $$\mathcal{Z}_v$$, respectively. In addition, text prompt $$\mathbf{t}$$ is randomly sampled from the pre-defined text library $$\mathbf{T}$$. The following losses using the CLIP distance function $$D_{\text{CLIP}}(\cdot, \cdot)$$ are used during training.

$$
\mathcal{L}_{\text{shape}} = f(\hat{\mathcal{D}}(\hat{\mathcal{F}}_{\theta}(v, \mathcal{M_s(\hat{\mathcal{E}}_t(\mathbf{t})) + z_s, z_a}))) + \\ \lambda_c D_{\text{CLIP}}(\hat{\mathcal{F}}_{\theta}(v, \mathcal{M}_s(\hat{\mathcal{E}}_t(\mathbf{t})) + z_s, z_a), \mathbf{t})
$$

$$
\mathcal{L}_{\text{appear}} = f(\hat{\mathcal{D}}(\hat{\mathcal{F}}_{\theta}(v, z_s, \mathcal{M_a(\hat{\mathcal{E}}_t(\mathbf{t})) + z_a}))) + \\ \lambda_c D_{\text{CLIP}}(\hat{\mathcal{F}}_{\theta}(v, z_s, \mathcal{M}_s(\hat{\mathcal{E}}_t(\mathbf{t}))+ z_a), \mathbf{t})
$$

The first term in each loss function is for fooling the discriminator so that the image generated after manipulation is similar to the actual image belonging to the training data distribution, and the second term is for maximizing cosine similarity of the generated image with the text prompt given in the CLIP embedding space.

### Inverse Manipulation
The above manipulation pipeline can only be performed using given shape code and appearance code. It is because disentangled conditional NeRF can output a 3D view when it receives shape code, appearance code, and view point as the input only. When there is only an input image $$\mathbf{I}_r$$, shape code, appearance code, and view point must be inversely estimated to directly manipulate the input image with prompt text or exemplar image. An iterative method using the EM(Expectation-Maximization) algorithm is used for this. EM algorithm optimizes the shape code $$z_s$$, the appearance code $$z_a$$, and the view point $$v$$ for the input image $$\mathbf{I}_r$$.

First, $$z_s$$ and $$z_a$$ are fixed at each step of the algorithm, and $$v$$ is optimized using the loss below. This is the process of finding optimal view point $$v$$ for a given input image.

$$
\mathcal{L}_v = || \hat{\mathcal{F}}_{\theta}(v, \hat{z}_s, \hat{z}_a) - \mathbf{I}_r ||_2 + \lambda_v D_{\text{CLIP}}(\hat{\mathcal{F}}_{\theta}(v, \hat{z}_s, \hat{z}_a), \mathbf{I}_r)
$$

Next, $$v$$ and $$z_a$$ are fixed at each step of the algorithm, and $$z_s$$ is optimized using the loss below. This is the process of finding optimal shape code $$z_s$$ for a given input image.

$$
\mathcal{L}_s = || \hat{\mathcal{F}}_{\theta}(\hat{v}, z_s + \lambda_n z_n, \hat{z}_a) - \mathbf{I}_r ||_2 + \lambda_s D_{\text{CLIP}}(\hat{\mathcal{F}}_\theta(\hat{v}, z_s + \lambda_n z_n, \hat{z}_a), \mathbf{I}_r)
$$

Finally, $$v$$ and $$z_s$$ are fixed at each step of the algorithm, and $$z_a$$ is optimized using the loss below. This is the process of finding optimal appearance code $$z_a$$ for a given input image.

$$
\mathcal{L}_{\text{appear}} = f(\hat{\mathcal{D}}(\hat{\mathcal{F}}_{\theta}(v, z_s, \mathcal{M_a(\hat{\mathcal{E}}_t(\mathbf{t})) + z_a}))) + \\ \lambda_c D_{\text{CLIP}}(\hat{\mathcal{F}}_{\theta}(v, z_s, \mathcal{M}_s(\hat{\mathcal{E}}_t(\mathbf{t}))+ z_a), \mathbf{t})
$$

$$z_n$$ was introduced to find the starting point of optimization as a random standard Gaussian noise vector extracted from each iteration step. During optimization, the size of $$z_n$$ gradually decreases from 1 to 0.



## 4. Experiment & Result
### Experimental setup
#### Dataset
- Photographs is composed of 150,000 chair photos with a resolution of $128 \times 128$.
- Carla is composed of 10,000 car photos with a resolution of $256 \times 256$.
- The view point information was not used in training because the training was performed adversarially.

#### Baselines
- The baseline model directly compared to CLIP-NeRF was EditNeRF.

#### Implementation Details
- Disentangled conditional NeRF is an 8-layer MLP with ReLU activation with input dimension of 64 and hidden unit of 256.
- The shape deformation network is a 4-layer MLP with ReLU activation with a hidden unit of 256. Both shape code $$z_s$$ and appearance code $$z_a$$ are 128-dimensional vectors.
- Both shape mapper and appearance mapper are 2-layer MLP with ReLU activation. The channel size is 128 (input) &rarr; 256 (hidden) &rarr; 128 (output) for both mappers.
- They used PatchGAN as a discriminator.
- They used adam optimizer with an initial learning rate of $10^{-4}$ and halved the learning rate for every 50K steps.
- The weight for the loss terms are $\lambda_{r} = 0.5$, $\lambda_v = 0.1$, and $\lambda_s = \lambda_a = 0.2$.

#### Evaluation Metrics
- Several quantitative metrics such as FID score, editing time, and user study results were provided to evaluate the performance of CLIP-NeRF.
- In addition, various images actually generated by CLIP-NeRF, such as clip-driven manipulation, real image manipulation, and ablation study result, were given to provide qualitative evaluation.

### Result

#### CLIP-Driven Manipulation
As shown in the two results below, they have confirmed that the outputs of NeRF can be manipulated with the desired shape or color when prompt text(ex: long car, red car) or exemplar image(ex : sports car image, table chair image) is given.

![Figure 6. Text-driven Editing Results of CLIP-NeRF](../../.gitbook/assets/2022spring/49/text_driven_editing_results.png)

![Figure 7. Image-driven Editing Results of CLIP-NeRF](../../.gitbook/assets/2022spring/49/exemplar_driven_editing_results.png)

#### Real Image Manipulation
They conducted an experiment to see if disentangled conditional NeRF is generalized well to unseen real image. Using the inversion method, they mapped a single real image to shape code and appearance code and used them to create an inverted image. Although the inverted image did not perfectly match the real image, the performance of the manipulation through text and image did not deteriorate.

![Figure 8. Inversion Method Results of CLIP-NeRF](../../.gitbook/assets/2022spring/49/real_images_results.png)

#### Comparison with EditNeRF
CLIP-NeRF requires fewer views than EditNeRF. Also, EditNeRF requires view point information, while CLIP-NeRF does not require it. For EditNeRF, the user should choose a color or draw a coarse scribble in the local region. However, for CLIP-NeRF, the user only need to provide a text prompt, or an exemplar image, which makes it much easier.

![Figure 9. Qualitative Comparison with EditNeRF](../../.gitbook/assets/2022spring/49/compare_to_editnerf.png)

They also compared the FID scores of EditNeRF and CLIP-NeRF. EditNeRF showed a lower FID score before manipulating the chair dataset because it saw 40 views per chair instance, but after the deformation, the FID became larger. In the case of car dataset, EditNeRF also saw only one view by car instance, so the FID was clearly large for EditNeRF.

![Figure 10. Quantitative Comparison with EditNeRF](../../.gitbook/assets/2022spring/49/compare_to_editnerf_fid.png)

Only certain parts of an object could be manipulated by EditNeRF and few layers in EditNeRF needed to be fine-tuned to manipulate the result. Also, EditNeRF did not perform properly for unseen view. On the other hand, CLIP-NeRF could make a big difference in overall shape and also performed well on unseen view. The reference speed of CLIP-NeRF was also faster than EditNeRF.

![Figure 11. Editing Time Comparison with EditNeRF](../../.gitbook/assets/2022spring/49/compare_to_editnerf_time.png)

#### Ablation Study
When training without using the disentangled conditional NeRF structure and conditional shape deformation network, the color changed together by changing the shape. This was because the vectors representing shape and color were not completely disentangled.

![Figure 12. Ablation Study](../../.gitbook/assets/2022spring/49/ablation_study.png)

#### User Study
Each user was asked 20 questions. Each question has a common source image, five randomly extracted prompt texts(or exemplar images), and manipulated images using this common source image and these five extracted prompt texts(or exemplar images). The correct answer rate was reported by matching which prompt text(or exemplar image) generated which manipulated image.

![Figure 13. User Study](../../.gitbook/assets/2022spring/49/user_study_results.png)



## 5. Conclusion
In this study, the authors proposed a text-and-image driven manipulation method for NeRF, which allows users to flexibly manipulate 3D content, by providing text prompts or exemplar images. To this end, they designed the disentangled conditional NeRF and introduced the CLIP-based shape and appearance code mapper. Finally, they proposed an inversion method which can obtain the shape and the appearance code from the real image and manipulate the real image from them.

However, there is still a problem that fine-granted, and out-of-domain shapes cannot manipulated as shown in the example below. This is because they are out of the training data of pre-trained CLIP model.

![Figure 14. Limitations of CLIP-NeRF](../../.gitbook/assets/2022spring/49/limitations.png)

In my opinion, there is no guarantee that conditional shape deformation does not change the color. Also, manipulation will be difficult for complex 3D images, and we cannot decide which part of the complex 3D image to change. Besides, it can be a problem that the number of people who participated in the user study is too small.

It was a fancy paper that suggested a novel method by combining the results of NeRF and CLIP. Rather than having original methods, it seems that the existing methods were properly mixed. In particular, disentangled conditional NeRF and CLIP-driven image generation and manipulation were key methods of this paper, but there have been existing studies using these methods.

My personal guess is that since the problem they want to solve is similar to that of EditNeRF, the reviewers required comparison between CLIP-NeRF and EditNeRF. Therefore, they reported FID, inference time, and the scope of each method thoroughly. It would have been very difficult to achieve high performance only with adversarial training by utilizing the GAN method. CLIP-NeRF is actually trained without information about view point, and ths is also a great achievement. Extensive experiments of CLIP-NeRF is also a key point for the acceptance. Thank you for reading such a long article, and if you have any questions, please contact me using the contact information below :)

### Take home message \(오늘의 교훈\)
> CLIP-NeRF, NeRF, and CLIP are all great papers. I want to write great papers, too.
>
> Let's study the existing studies steadily and use them in the right place.
>
> Extensive experiments increase the probability of acceptance of papers.


## Author / Reviewer information

### Author

**김창훈 (Changhun Kim)**

* Master's student @ [MLILAB](https://mli.kaist.ac.kr), [KAIST Graduate School of AI](https://gsai.kaist.ac.kr)
* Contact information : <ssohot1@kaist.ac.kr>, [GitHub](https://github.com/drumpt), [Blog](https://drumpt.github.io)
* Research interests : Speech Processing, Generative Models, Graph Neural Networks, Bayesian Deep Learning

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
1. Korean name \(English name\): Affiliation / Contact information
1. ...

## Reference & Additional materials

1. [CLIP-NeRF Paper](https://arxiv.org/abs/2112.05139)
1. [CLIP-NeRF Implementation Examples](https://cassiepython.github.io/clipnerf/)
1. [CLIP-NeRF GitHub Repository](https://github.com/cassiePython/CLIPNeRF)
1. [NeRF Paper](https://arxiv.org/abs/2003.08934)
1. [CLIP Paper](https://arxiv.org/abs/2103.00020)
1. [PatchGAN Paper](https://arxiv.org/abs/1611.07004v3)
1. [Transformer Paper](https://arxiv.org/abs/1706.03762)