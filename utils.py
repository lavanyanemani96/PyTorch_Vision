'''
dataloaders +
model summary +
plotting +
image transforms +
gradcam +
misclassification code +
tensorboard related stuff
advanced training policies
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    transform = A.Compose([A.PadIfNeeded(min_height=36,
                                        min_width=36,
                                        border_mode=cv2.BORDER_CONSTANT,
                                        value=np.mean(mu)),
                           A.RandomCrop(32, 32),
                           A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=np.mean(mu)),
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
                ax[i, j].set_title('Label: %s, Pred: %s' %(classes[label[index].cpu()],classes[predictions[index].cpu()].argmax()))
                ax[i, j].imshow(np.transpose(image[index].cpu(), (1, 2, 0)))
    else:
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title("Label: %s" %(classes[label[index]]))
                ax[i, j].imshow(np.transpose(image[index], (1, 2, 0)))

def superimpose(heatmap, image):

    image = np.transpose(image.cpu(), (1, 2, 0))
    heatmap = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_image = (heatmap * 0.4) + 255*image.numpy()

    return superimposed_image/superimposed_image.max()



def device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, use_cuda

def show_summary(model, device, input_size):
  print(summary(model.to(device), input_size=input_size))

def plot_loss_accuracy(results):
    train_losses, test_losses, train_acc, test_acc = results
    fig, axs = plt.subplots(2,2,figsize=(15,10))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    axs[1, 0].plot(train_accuracy)
    axs[1, 0].set_title("Training Accuracy")

    axs[1, 1].plot(test_accuracy)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]

def gradcam(model, results, test_images, device):

    results[torch.arange(len(results)),
            results.argmax(dim=1)].backward(torch.ones_like(results.argmax(dim=1)))

    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    activations = model.get_activations(test_images.to(device)).detach()

    for j in range(activations.shape[0]):
        for i in range(512):
            activations[j, i, :, :] *= pooled_gradients[j, i]

    heatmaps = torch.mean(activations, dim=1).squeeze()
    heatmaps = np.maximum(heatmaps.cpu(), 0)
    heatmaps /= torch.max(heatmaps)

    return heatmaps
