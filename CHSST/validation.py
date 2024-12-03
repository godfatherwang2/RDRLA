def iou(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            # if there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / union)
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


from matplotlib import pyplot as plt
import numpy as np


def drawcurve_for_final(x, loss, vloss, accu, saveTo=None, suptitle=''):
    loss = np.array(loss)
    vloss = np.array(vloss)
    plt.clf()
    plt.figure(figsize=(6, 8), dpi=400)
    plt.suptitle(suptitle, fontsize='x-large', fontweight='bold')
    ex = x[0] + len(loss)
    ey = max(loss.max(), vloss.max())

    plt.subplot(2, 1, 1)
    plt.grid(True, which='both')
    plt.title('Loss')
    plt.plot(range(x[0], x[0] + len(loss)), loss)
    plt.plot(x, vloss)
    plt.legend(['train', 'val'], loc='lower left')
    plt.axis([x[0], ex, 0, ey])

    plt.subplot(2, 1, 2)
    plt.grid(True, which='both')
    plt.title('Accuracy (%)')
    plt.plot(x, accu)
    plt.legend(['Target', 'Rank10', 'Rank30', 'Rank50'], loc='upper left')

    if saveTo:
        plt.savefig(saveTo)
    else:
        plt.show()


def drawcurve_for_fcn(x, data1, data2, losses, saveTo=None):
    data1 = np.array(data1)
    data2 = np.array(data2)
    plt.clf()
    plt.figure(figsize=(6, 12), dpi=400)

    plt.subplot(3, 1, 1)
    plt.title('Losses')
    plt.plot(range(len(losses)), losses)

    sx = min(x);
    ex = max(x)
    plt.subplot(3, 1, 2)
    plt.title('IoU Score')
    plt.plot(x, data1)
    sy = data1.min();
    ey = data1.max()
    plt.axis([sx, ex, sy, ey])

    plt.subplot(3, 1, 3)
    plt.title('Pixel Accuracy')
    plt.plot(x, data2)
    sy = data2.min();
    ey = data2.max()
    plt.axis([sx, ex, sy, ey])


def drawcurve(x, loss, vloss, saveTo=None):
    loss = np.array(loss)
    vloss = np.array(vloss)
    plt.clf()
    plt.figure(figsize=(6, 10), dpi=400)
    ex = len(loss)
    ey = max(loss.max(), vloss.max())

    plt.subplot(3, 1, 1)
    plt.title('Loss')
    plt.plot(range(len(loss)), loss[:, 0])
    plt.plot(x, vloss[:, 0])
    plt.legend(['train', 'val'], loc='upper right')
    plt.axis([0, ex, 0, ey])

    plt.subplot(3, 1, 2)
    plt.title('Regression Loss')
    plt.plot(range(len(loss)), loss[:, 1])
    plt.plot(x, vloss[:, 1])
    plt.legend(['train', 'val'], loc='upper right')
    plt.axis([0, ex, 0, ey])

    plt.subplot(3, 1, 3)
    plt.title('Classification Loss')
    plt.plot(range(len(loss)), loss[:, 2])
    plt.plot(x, vloss[:, 2])
    plt.legend(['train', 'val'], loc='upper right')
    plt.axis([0, ex, 0, ey])

    if saveTo:
        plt.savefig(saveTo)
    else:
        plt.show()

