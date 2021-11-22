'''
training and test loops +
data split between test and train +
epochs +
batch size +
which optimizer to run +
do we run a scheduler +
'''

from tqdm import tqdm
import torch
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):

    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
      super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
      image, label = self.data[index], self.targets[index]

      if self.transform is not None:
        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, label

def dataset(transforms, train_bool):
    return Cifar10SearchDataset(root='./data', train=train_bool, download=True, transform=transforms)

class args():
    def __init__(self, batch_size = 128, device = 'cpu',use_cuda = False):
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 4, 'pin_memory': True} if self.use_cuda else {}

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, criterion, optimizer, epoch):

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0

  for batch_idx, (data, target) in enumerate(pbar):

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    y_pred = model(data)
    loss = criterion(y_pred, target)
    train_loss += loss
    loss.backward()
    optimizer.step()

    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_loss /= len(train_loader.dataset)
  train_losses.append(loss)
  train_acc.append(100*correct/processed)


def test(model, device, test_loader, criterion):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

class train_args():
    def __init__(self, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optimizer = device
        self.scheduler = scheduler

def train_model(model, train_args, train_loader, test_loader, EPOCHS):

    criterion, optimizer, scheduler = train_args
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, criterion , optimizer, epoch)
        test(model, device, test_loader, criterion)

        if scheduler:
            scheduler.step()

    return [train_losses, test_losses, train_acc, test_acc]
