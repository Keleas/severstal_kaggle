import itertools
import math
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
num_epochs = 100
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

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

def train(train_loader, model,  scheduler, total_step, optimizer, criterion):
    correct, total, running_loss = 0, 0, 0
    accuracy, predicted = [], []
    losses = []
    for (image, label) in tqdm(train_loader):
        image, label = image.cuda(), label.cuda()

        with torch.set_grad_enabled(True):
            outputs = model(image)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, label)
            # loss = (torch.autograd.Variable(dice_loss(outputs, label), requires_grad=True))
            losses.append(loss.item())
            # loss = nn.BCEWithLogitsLoss()(outputs, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print('Train loss:', np.mean(losses))
        # print('Training loss:', dice_loss(outputs, label))
        # total += label.size(0)
        # predicted.append(outputs.detach().cpu().numpy())

        # correct += (outputs.detach().cpu().numpy() == label.detach().cpu().numpy()).sum().item()
        # print('label:{}, prediction:{}'.format(label, predicted))
        # running_loss += loss.item()*image.size(0)
        # running_loss += loss.item()

        # accuracy.append(correct / total * 100)
    return accuracy, running_loss


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def validation(val_loader, model, all_valid_loss, loss_counter, criterion):
    model.eval()
    all_predicted, all_labels, predicted, true = [], [], [], []
    correct, total = 0, 0
    cum_loss = 0
    dice_best_score = []
    iou_best_score = []
    for (images, labels) in tqdm(val_loader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        # loss = criterion(outputs, labels)
        # loss = dice_loss(outputs, labels)
        prediction = outputs.detach().cpu()#.numpy()
        target = labels.detach().cpu()#.numpy()
        # dice = dice_loss(prediction, target)
        dice, _, _, _, _ = metric(prediction, target)
        dice_best_score.append(dice)
        # correct += (outputs.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum().item()
        total += labels.size(0)
        # print('Validation loss:', loss)
        # cum_loss += loss.item() * images.size(0)
        # cum_loss += loss
        # all_predicted.append(predicted)
        # all_labels.append(labels)
        prediction, target = prediction.numpy(), target.numpy()

        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        # iou_score = compute_iou_batch(prediction, target, classes=[1])
        iou_best_score.append(iou_score)



        # fig = plt.figure()
        # for type in range(4):
        #     plt.subplot(211)
        #     plt.imshow(target[0, type, :, :])
        #     plt.subplot(212)
        #     plt.imshow(prediction[0, type, :, :])
        #     plt.savefig('D:\\My_Data\\_Desktop\\check\\'+ str(i)+'_'+str(type) + '.png')
        # plt.close(fig)

    # predicts = np.concatenate(predicted).squeeze()
    # truths = np.concatenate(true).squeeze()
    # precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)

    # precision = precision.mean()
    print('VALIDATION: Dice: {}, Iou: {}'.format(np.mean(dice_best_score), np.mean(iou_best_score)))
    # all_valid_loss = np.append(all_valid_loss, loss.item())
    # if len(all_valid_loss) >= 2 and all_valid_loss[-2] < all_valid_loss[len(all_valid_loss)-1]:
    #     loss_counter += 1
    # return ((list(itertools.chain.from_iterable(all_predicted)))), ((list(itertools.chain.from_iterable(all_labels)))), all_valid_loss
    return all_predicted, all_labels, all_valid_loss, cum_loss / total

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
            # predicted.append(outputs.detach().cpu().numpy())
            correct += (outputs.detach().cpu().numpy() == labels.detach().cpu().numpy()).sum().item()
            # all_predicted.append(predicted)
            # all_labels.append(labels)
        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))
    # return ((list(itertools.chain.from_iterable(all_predicted)))), ((list(itertools.chain.from_iterable(all_labels))))
    return all_predicted, all_labels


def training(train_loader, val_loader, test_loader, model):
    val_predicted, val_labels, all_valid_loss = [], [], []
    train_accuracy = []
    test_predicted, test_labels = [], []
    loss_counter, lr, steps = 0, 10, 10
    all_valid_loss = np.array(all_valid_loss)
    # train_accuracy = np.array(train_accuracy)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    criterion = nn.SmoothL1Loss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        print("Epoch:[{}/{}]".format(epoch + 1, num_epochs))
        # lr = scheduler.get_lr()[-1]

        # train_accuracy = np.append(train_accuracy,
        #     train(train_loader, model, scheduler, total_step=len(train_loader), optimizer=optimizer))
        model.train()
        train(train_loader, model, scheduler, total_step=len(train_loader), optimizer=optimizer, criterion=criterion)
        # predicted, label, all_valid_loss = validation(val_loader, model, all_valid_loss, loss_counter)
        _, _, _, val_loss = validation(val_loader, model, all_valid_loss, loss_counter, criterion)
        scheduler.step(val_loss)
        # val_predicted.append(predicted)
        # val_labels.append(label)

        # train_accuracy= np.array((list(itertools.chain.from_iterable(train_accuracy))))
        # if all_valid_loss[len(all_valid_loss)-1] == loss_to_save:
        #     torch.save(model.state_dict(),
        #                'G:\\SteelDetection\\model.pth')
        # if loss_counter > patience_valid:
        #     break
        torch.cuda.empty_cache()


    # val_predicted, val_labels = ((list(itertools.chain.from_iterable(val_predicted)))), (
    # (list(itertools.chain.from_iterable(val_labels))))
    # test_predicted, test_labels = test(test_loader, model)
    return  train_accuracy, val_predicted, val_labels, test_predicted, test_labels
