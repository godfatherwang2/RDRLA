from logging import logProcesses
import os
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from CHSST.models.toptransformer.basemodel import Topformernet
from CHSST.models.toptransformer.seaformer import Seaformernet
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from CHSST.utils import Arguments,timer
from CHSST.dataset import TopFormerDataset_with_augment
from CHSST.validation import iou, pixel_acc,drawcurve_for_fcn

args = Arguments(
    dataset_list='ALL.json',
    dataset_path='/mnt/sda2/PalmRec/',
    background_path='/mnt/sda2/backgroundimgs',
    lr=0.1,
    model="Sea",
    load=False,
    start_epoch=6,
    use_cuda=True,
    num_classes=2,
    record_path='TrainRec/',
    image_size=(448, 448),
    batch_size=32,
    num_workers=8,
    opt_momentum=0.9,
    opt_weight_decay=1e-3,
    sch_step_size=10,
    sch_gamma=0.5,
    num_epochs=81,
    debug_steps=100,
    validation_epochs=1
)

if not os.path.exists(args.record_path):
    os.makedirs(args.record_path)

if torch.cuda.is_available() and args.use_cuda:
    DEVICE = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    print('Use Cude.')
else:
    DEVICE = torch.device('cpu')
    print('Use CPU.')

IU_scores = []
pixel_scores = []
losses = []
axis_x = []


def main():
    print('Prepare training datasets.')
    train_data = TopFormerDataset_with_augment(listfile=args.dataset_list, root=args.dataset_path, if_vit=True,
                                               bright=True, pure_background=True,
                                               backgroundroot=args.background_path, phase="train",
                                               trainsize=args.image_size, crop=True)

    print("trainnum:", train_data.__len__())
    val_data = TopFormerDataset_with_augment(listfile=args.dataset_list, root=args.dataset_path, if_vit=True,
                                             bright=True, pure_background=True,
                                             backgroundroot=args.background_path, phase="val",
                                             trainsize=args.image_size)

    print("valnum:", val_data.__len__())
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.num_workers)

    print('Build Network.')

    if args.model == "Top":
        if args.load == True:
            fcn_model = torch.load("TrainRec/EP5-iou0.98234-pacc0.992178.pth").cuda()
            start_epoch = args.start_epoch
        else:
            fcn_model = Topformernet()
            if args.use_cuda:
                fcn_model = fcn_model.cuda()
            start_epoch = 0

    elif args.model == "Sea":
        if args.load == True:
            fcn_model = torch.load("TrainRec/EP5-iou0.99764-pacc0.99325.pth").cuda()
            start_epoch = args.start_epoch
        else:
            fcn_model = Seaformernet()
            if args.use_cuda:
                fcn_model = fcn_model.cuda()
            start_epoch = 0
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=args.lr, momentum=args.opt_momentum,
                          weight_decay=args.opt_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    '''
    optimizer = optim.RMSprop(fcn_model.parameters(
    ), lr=args.lr, momentum=args.opt_momentum, weight_decay=args.opt_weight_decay)

    #lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.sch_gamma)
    '''
    for epoch in range(start_epoch, args.num_epochs):
        timer.start()
        loss, ious, pixel_accs = train(fcn_model, optimizer, scheduler, criterion, train_loader, epoch)
        losses.append(loss)
        print("epoch:", epoch + 1, "train_iou:", ious, "train_acc:", pixel_accs)
        model_path = os.path.join(args.record_path, 'Latest.pth')
        torch.save(fcn_model, model_path)
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            ious, pixel_accs = val(fcn_model, criterion, val_loader, epoch)
            print("Validated epoch:%d, pix_acc:%f, meanIoU:%f, IoUs:%s" % (
                epoch, pixel_accs, np.nanmean(ious), str(ious)))
            # if epoch%5 == 0 or epoch == args.num_epochs-1:
            model_path = os.path.join(args.record_path, 'EP%d-iou%f-pacc%f.pth' % (epoch, np.nanmean(ious), pixel_accs))
            torch.save(fcn_model, model_path)
            drawcurve_for_fcn(axis_x, IU_scores, pixel_scores, losses,
                              os.path.join(args.record_path, 'fig.png'))
        print('Finish train epoch:%d, loss:%f' % (epoch, loss))
        print('Time elapses:%f' % timer.end())


def train(net, optimizer, scheduler, criterion, train_loader, epoch):
    net.train()
    total_ious = []
    pixel_accs = []

    total_loss = 0

    for ind, batch in enumerate(tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        if args.use_cuda:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['Y'])
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if ind % args.debug_steps == 0:
            print("Epoch{}, step{}, loss: {}".format(epoch, ind, loss.item()))
            scheduler.step()
        op = outputs.data.cpu().numpy()
        N, _, h, w = op.shape
        pred = op.transpose(0, 2, 3, 1).reshape(-1, args.num_classes).argmax(axis=1).reshape(N, h, w)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t, args.num_classes))
            pixel_accs.append(pixel_acc(p, t))

        total_loss += loss.item()
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    IU_scores.append(ious)
    pixel_scores.append(pixel_accs)
    axis_x.append(epoch)
    return total_loss / ind, ious, pixel_accs

def val(net, criterion, val_loader, epoch):
    net.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if args.use_cuda:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = net(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, args.num_classes).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t, args.num_classes))
            pixel_accs.append(pixel_acc(p, t))
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    IU_scores.append(ious)
    pixel_scores.append(pixel_accs)
    axis_x.append(epoch)
    return ious, pixel_accs


if __name__ == '__main__':
    main()


