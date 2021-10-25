## DI-FGSM : DIFGSM.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchgeometry as tgm
from attack import Attack


class DIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, di_pad_amount=31, di_prob=0.5):
        super().__init__("DIFGSM", model)
        self.eps = eps # Maximum change in one pixel for total step (range 0-255)
        self.steps = steps # number of di-fgsm steps
        self.alpha = alpha # Maximum change in one pixel for one step (range 0-255)
        self.di_pad_amount = di_pad_amount # Maximum value that can be padded
        self.di_prob = di_prob # Probability of deciding whether to apply DI transform or not
        self._supported_mode = ['default', 'targeted'] # deciding targeted attack or not

    def diverse_input(self, x_adv):
        x_di = x_adv # size : [24,3,32,32]
        h, w = x_di.shape[2], x_di.shape[3] # original image size, h: 32, w: 32
        # random value that be padded
        pad_max = self.di_pad_amount - int(torch.rand(1) * self.di_pad_amount) # pad_max : 2
        # random value that be padded left
        pad_left = int(torch.rand(1) * pad_max) # pad_left : 1
        # random value that be padded right
        pad_right = pad_max - pad_left # pad_right : 1
        # random value that be padded top
        pad_top = int(torch.rand(1) * pad_max) # pad_top : 1
        # random value that be padded bottom
        pad_bottom = pad_max - pad_top  # pad_bottom : 1

        # four vertices of the original image
        # tensor([[[ 0.,  0.], [31.,  0.], [31., 31.], [ 0., 31.]]])
        points_src = torch.FloatTensor([[
            [0, 0], [w - 1, 0], [w - 1 + 0, h - 1 + 0], [0, h - 1 + 0],
        ]])

        # four vertices of the image after DI transform
        # tensor([[[ 1.,  1.], [30.,  1.], [30., 30.], [ 1., 30.]]])
        points_dst = torch.FloatTensor([[
            [pad_left, pad_top], [w - pad_right - 1, pad_top],
            [w - pad_right - 1, h - pad_bottom - 1], [pad_left, h - pad_bottom - 1],
        ]])

        # Matrix used in the transformation process
        # tensor([[[0.9355, 0.0000, 1.0000], [0.0000, 0.9355, 1.0000], [0.0000, 0.0000, 1.0000]]])
        M = tgm.get_perspective_transform(points_src, points_dst)

        # The image is resized and padded so that the vertices of the original image go to the new vertices.
        x_di = tgm.warp_perspective(x_di, torch.cat(x_di.shape[0] * [M]).cuda(), dsize=(w, h)).cuda()
        x_di = transforms.Resize((w, h), interpolation=InterpolationMode.NEAREST)(x_di)

        # If the random value is less than or equal to di_prob, di conversion does not occur.
        cond = torch.rand(x_adv.shape[0]) < self.di_prob

        cond = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_di = torch.where(cond.cuda(), x_di, x_adv)
        return x_di

    def forward(self, images, labels):
        """
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted: # targeted attack case, get target label
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss() # use Cross-Entropy loss for classification
        adv_images = images.clone().detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.diverse_input(adv_images)) # after DI transform image

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels) # targeted attack case, use -loss function
            else:
                cost = loss(outputs, labels) # else, (untargeted attack case), use +loss function

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)

            adv_images = adv_images.detach() + self.alpha*grad.sign() # I-fgsm step
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps) # limiting changes beyond epsilon
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images