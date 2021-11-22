'''
dataloaders +
model summary +
plotting +
image transforms +
gradcam
misclassification code +
tensorboard related stuff
advanced training policies
'''
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

def mean_std_cifar10(dataset):

  imgs = [item[0] for item in dataset]
  labels = [item[1] for item in dataset]

  imgs = torch.stack(imgs, dim=0).numpy()

  mean_r = imgs[:,0,:,:].mean()
  mean_g = imgs[:,1,:,:].mean()
  mean_b = imgs[:,2,:,:].mean()
  mu = [mean_r,mean_g,mean_b]
  print("Mean:", mu)

  std_r = imgs[:,0,:,:].std()
  std_g = imgs[:,1,:,:].std()
  std_b = imgs[:,2,:,:].std()
  sigma = [std_r,std_g,std_b]
  print("Std:", sigma)

  return mu, sigma

def augmentation(data, mu, sigma):

  if data == 'Train':
    transform = A.Compose([A.RandomCrop(32, padding=4),
                           A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=np.mean(mu)),
                           A.Rotate(limit=5),
                           A.Normalize(mean=mu, std=sigma),
                           ToTensorV2()])
  else:
    transform = A.Compose([A.Normalize(mean=mu, std=sigma),
                           ToTensorV2()])

  return transform

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_grid(image, label, predictions=None):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))
    if predictions:
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title("Label: %s, Pred: %s" %(classes[label[index].cpu()],classes[predictions[index].cpu().argmax()))
                ax[i, j].imshow(np.transpose(image[index].cpu(), (1, 2, 0)))
    else:
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title("Label: %s" %(classes[label[index]]))
                ax[i, j].imshow(np.transpose(image[index], (1, 2, 0)))

def device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda

def data_loader(dataset, device, use_cuda):
    return torch.utils.data.DataLoader(dataset, batch_size=args(batch_size, device, use_cuda).batch_size, shuffle=True, **args(batch_size, device, use_cuda).kwargs)

def summary(model, device, input_size):
  print(summary(model.to(device), input_size=input_size))

def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]
