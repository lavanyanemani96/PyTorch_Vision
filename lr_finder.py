import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def lr_exp(i, lr):
    return lr + np.exp(i**1.4/1e3)/1e5

def lr_iterations(criterion, optimizer, model, train_loader):

    lrs, lr_finder_losses, iterations = [], [], []

    i = 0
    for epoch in range(4):
        model.train()
        pbar = tqdm(train_loader)

        for _, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            y_pred = model(data)
            loss = criterion(y_pred, target)
            lr_finder_losses.append(loss)

            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            iterations.append(i)

            loss.backward()
            optimizer.step()

            optimizer.param_groups[0]["lr"] = lr_exp(i, lr)
            i = i+1

    plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, lrs)
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(lrs, lr_finder_losses)
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.grid()

    plt.show()
