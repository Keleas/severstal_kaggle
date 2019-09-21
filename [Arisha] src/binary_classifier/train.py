import itertools
import math
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import seaborn as sns
from collections import Counter
import itertools

num_epochs = 100
num_classes = 2


def train(train_loader, model, optimizer, criterion):
    losses = []
    current_loss, current_acc = 0, 0
    model.train()
    for (image, label) in tqdm(train_loader):
        image = image.cuda()
        label = label.cuda()
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            image.requires_grad = True #?
            outputs = model(image)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            # print('gradients =', [x.grad.data for x in model.parameters()])
            optimizer.step()
            # print('weights after backpropagation = ', list(model.parameters()))
        current_loss += loss.item() * image.size(0)
        current_acc += torch.sum(predictions == label.data)
        losses.append(loss.item())
    total_loss = current_loss / len(train_loader.dataset)
    total_acc = 100 * current_acc.double() / len(train_loader.dataset)
    print('TRAIN: Loss: {}, Accuracy: {}%'.format(total_loss, total_acc))


def validation(val_loader, model, criterion):
    model.eval()
    losses, all_predicted, all_labels = [], [], []
    total, correct = 0, 0
    with torch.set_grad_enabled(False):
        for (image, label) in tqdm(val_loader):
            image = image.cuda()
            label = label.cuda()
            outputs = model(image)
            loss = criterion(outputs, label)
            losses.append(loss.item())
            total += label.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.append(predicted.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            correct += (predicted == label).sum().item()
    mean_loss = np.mean(losses)
    print('VALIDATION: Loss: {}, Accuracy: {}%'.format(mean_loss, (correct/total) * 100))
    return mean_loss, list(itertools.chain.from_iterable(all_predicted)), list(itertools.chain.from_iterable(all_labels))


def training(train_loader, val_loader, model):
    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
    for epoch in range(num_epochs):
        print("Epoch:[{}/{}]".format(epoch + 1, num_epochs))
        train(train_loader, model, optimizer, criterion)
        val_loss, predicted, labels = validation(val_loader, model, criterion)
        # print('gradients =', [x.grad.data for x in model.parameters()])
        # confusion_matrix(predicted, labels)
        scheduler.step(val_loss)
    torch.save(model.state_dict(), 'G:\\SteelDetection\\new_model.pth')


def confusion_matrix(predicted, labels):
    print(predicted)
    print(labels)
    all = len(labels)
    data = (dict(Counter(sorted(zip(predicted, labels)))))
    mas = np.zeros((num_classes, num_classes))
    mas_prob = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if (i,j) in data.keys():
                mas[i,j] = data[(i,j)]
                mas_prob[i,j] = data[(i,j)] / all
    plt.figure()
    # plt.subplot(211)
    ax = sns.heatmap(mas, annot=True)
    # plt.ylabel('predicted')
    # plt.xlabel('label')
    # plt.subplot(212)
    # ax = sns.heatmap(mas_prob, annot=True)
    plt.ylabel('predicted')
    plt.xlabel('label')
    plt.show()
