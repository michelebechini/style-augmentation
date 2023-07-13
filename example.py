from styleaug import StyleAugmentor

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# PyTorch Tensor <-> PIL Image transforms:
toTensor = ToTensor()
toPIL = ToPILImage()

# load image:
# im = Image.open('tango.jpg')
im = cv2.imread('mug.png')
im_torch = toTensor(im).unsqueeze(0) # 1 x 3 x 256 x 256
im_torch = im_torch.to('cuda:0' if torch.cuda.is_available() else 'cpu')

# create style augmentor:
augmentor = StyleAugmentor()

# randomize style:
im_restyled = augmentor(im_torch)

# get restyled img as numpy array with HxWxC channels order
img_out = im_restyled.squeeze().permute(1, 2, 0).cpu().numpy()

# display:
plt.imshow(img_out)
plt.show()