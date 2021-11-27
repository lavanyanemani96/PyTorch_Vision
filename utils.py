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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def mean_std_cifar10(dataset):

  train_data = dataset.data
  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  return mean, std

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

def augmentation_custom_resnet(data, mu, sigma, pad=4):

  if data == 'Train':
    transform = A.Compose([A.PadIfNeeded(min_height=32+pad,
                                        min_width=32+pad,
                                        border_mode=cv2.BORDER_CONSTANT,
                                        value=np.mean(mu)),
                            A.RandomCrop(32, 32),
                            A.HorizontalFlip(p=0.5),
                            A.Cutout(max_h_size=8, max_w_size=8),
                            A.Normalize(mean=mu, std=sigma),
                            ToTensorV2(),
  else:
    transform = A.Compose([A.Normalize(mean=mu, std=sigma),
                           ToTensorV2()])

  return transform

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_grid(image, label, UnNorm=None, predictions=[]):

    nrows = 2
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))
    if len(predictions):
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title('Label: %s, \nPred: %s' %(classes[label[index].cpu()],classes[predictions[index].cpu().argmax()]))
                ax[i, j].imshow(np.transpose(UnNorm(image[index].cpu()), (1, 2, 0)))
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

def show_summary(model, device, input_size):
  print(summary(model.to(device), input_size=input_size))

def plot_loss_accuracy(results):
    train_losses, test_losses, train_acc, test_acc = results
    fig, axs = plt.subplots(2,2,figsize=(15,10))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

def classfication_result(predictions, labels, device, b=True):
    # for misclassified images, b = False
    return torch.where((predictions.argmax(dim=1) == labels) == b)[0]

def grad_cam(model, use_cuda, test_images, test_targets):

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=use_cuda)
    image_cam = cam(input_tensor=test_images, target_category=test_targets)

    return image_cam

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + 5*img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_grid_cam(test_images, cam_images, label, predictions, UnNorm=None):

    nrows = 2
    ncols = 5

    test_images = [np.transpose(UnNorm(test_images[i]).cpu().detach().numpy(), (1,2,0)) for i in range(len(test_images))]
    cam_images = [cam_images[i].cpu().detach().numpy() for i in range(len(cam_images))]

    super_impose = [show_cam_on_image(test_images[i], cam_images[i], use_rgb=True) for i in range(len(test_images))]

    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 4))
    if len(predictions):
        for i in range(nrows):
            for j in range(ncols):
                index = i * ncols + j
                ax[i, j].axis("off")
                ax[i, j].set_title('Label: %s, \nPred: %s' %(classes[label[index].cpu()],classes[predictions[index].cpu().argmax()]))
                ax[i, j].imshow(super_impose[index])
