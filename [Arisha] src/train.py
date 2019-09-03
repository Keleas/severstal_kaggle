import itertools
import math
import numpy as np
import torch
from tqdm import tqdm

num_epochs = 10
patience_valid = 20
channel_num = 4
loss_to_save = 1


def do_kaggle_metric(predict,truth, threshold=0.5):

    N = len(predict)
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)

    predict = predict>threshold
    truth   = truth>0.5
    intersection = truth & predict
    union        = truth | predict
    iou = intersection.sum(1)/(union.sum(1)+1e-8)

    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (truth.sum(1)==0)
    is_empty_predict = (predict.sum(1)==0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou>=t

        tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou> t)
        fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou<=t)
        fn  = (~is_empty_truth)  & ( is_empty_predict)
        fp_empty = ( is_empty_truth)  & (~is_empty_predict)
        tn_empty = ( is_empty_truth)  & ( is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
        precision.append(p)

    result = np.array(result).transpose(1,2,0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold


def dice_loss(probability, truth, threshold=0.5):
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.
    with torch.no_grad():
        for i in range(batch_size):
            for j in range(channel_num):
                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)
                mean_dice_channel += channel_dice/(batch_size * channel_num)
    return mean_dice_channel


def dice_single_channel(probability, truth, threshold, eps = 1E-9):
    p = (probability.view(-1) > threshold).float()
    t = (truth.view(-1) > 0.5).float()
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice

def train(train_loader, model,  scheduler, total_step, optimizer):
    correct, total, running_loss = 0, 0, 0
    accuracy, predicted = [], []
    for i, (image, label) in tqdm(enumerate(train_loader)):
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        # loss = criterion(outputs, label)
        loss = torch.autograd.Variable(dice_loss(outputs, label), requires_grad=True)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total += label.size(0)
        predicted.append(outputs.detach().cpu().numpy())

        # correct += (outputs.detach().cpu().numpy() == label.detach().cpu().numpy()).sum().item()
        # print('label:{}, prediction:{}'.format(label, predicted))
        # running_loss += loss.item()*image.size(0)
        running_loss += loss
        if (i + 1) % 100 == 0:
            print('TRAIN: Step [{}/{}], Loss: {:.4f}'
                  .format(i + 1, total_step, running_loss / 100))
            running_loss = 0
            # accuracy.append(correct / total * 100)
    return accuracy


def validation(val_loader, model, all_valid_loss, loss_counter):
    all_predicted, all_labels, predicted, true = [], [], [], []
    correct, total = 0, 0
    cum_loss = 0
    for (images, labels) in tqdm(val_loader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        # loss = criterion(outputs, labels)
        loss = dice_loss(outputs, labels)
        predicted.append(outputs.detach().cpu().numpy())
        true.append(labels.detach().cpu().numpy())
        correct += (outputs.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum().item()
        total += labels.size(0)
        # cum_loss += loss.item() * images.size(0)
        cum_loss += loss
        all_predicted.append(predicted)
        all_labels.append(labels)
    predicts = np.concatenate(predicted).squeeze()
    truths = np.concatenate(true).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()
    print('VALIDATION: Loss: {:4f}, Accuracy: {:.2f} %'.format(cum_loss / total, precision.mean()))
    # all_valid_loss = np.append(all_valid_loss, loss.item())
    if len(all_valid_loss) >= 2 and all_valid_loss[-2] < all_valid_loss[len(all_valid_loss)-1]:
        loss_counter += 1
    return ((list(itertools.chain.from_iterable(all_predicted)))), ((list(itertools.chain.from_iterable(all_labels)))), all_valid_loss


def test(test_loader, model):
    all_predicted, all_labels, predicted = [], [], []
    with torch.no_grad():
        correct = 0
        total = 0
        for (images, labels) in tqdm(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            total += labels.size(0)
            predicted.append(outputs.detach().cpu().numpy())

            correct += (outputs.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum().item()
            all_predicted.append(predicted)
            all_labels.append(labels)
        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))
    return ((list(itertools.chain.from_iterable(all_predicted)))), ((list(itertools.chain.from_iterable(all_labels))))


def training(train_loader, val_loader, test_loader, model):
    val_predicted, val_labels, all_valid_loss = [], [], []
    train_accuracy = []
    loss_counter, lr, steps = 0, 1., 10
    all_valid_loss = np.array(all_valid_loss)
    train_accuracy = np.array(train_accuracy)

    for epoch in tqdm(range(num_epochs)):
        print("Epoch:[{}/{}]".format(epoch + 1, num_epochs))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        lr = scheduler.get_lr()[-1]

        train_accuracy = np.append(train_accuracy,
            train(train_loader, model, scheduler, total_step=len(train_loader), optimizer=optimizer))

        predicted, label, all_valid_loss = validation(val_loader, model, all_valid_loss, loss_counter)
        scheduler.step()
        val_predicted.append(predicted)
        val_labels.append(label)
        # train_accuracy= np.array((list(itertools.chain.from_iterable(train_accuracy))))
        # if all_valid_loss[len(all_valid_loss)-1] == loss_to_save:
        #     torch.save(model.state_dict(),
        #                'G:\\SteelDetection\\model.pth')
        # if loss_counter > patience_valid:
        #     break


    val_predicted, val_labels = ((list(itertools.chain.from_iterable(val_predicted)))), (
    (list(itertools.chain.from_iterable(val_labels))))
    test_predicted, test_labels = test(test_loader, model)
    return  train_accuracy, val_predicted, val_labels, test_predicted, test_labels
