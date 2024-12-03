import RandAugment
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn import metrics
from scipy import interpolate
from scipy import optimize
import pandas as pd
import tqdm
from torchvision import transforms
from torchvision import transforms as T
from datasets import Dataset_closeset,Dataset_closeset_session
from model.RLANN import RLANN
from model.ACPLoss import ACPLoss
from torch.utils.data import DataLoader
import time
import os
import json
from itertools import chain
from torch import nn


def filter_long(l_list):
    step = len(l_list) // 1000
    sampled_values = l_list[::step]
    if len(sampled_values) < 1000:
        sampled_values = l_list
    return sampled_values

class NormSingleROI(object):
    def __init__(self, outchannels=3):
        self.outchannels = outchannels
    def __call__(self, tensor):
        c, h, w = tensor.size()
        tensor = tensor.view(c,h * w)
        idx = tensor > 0
        t = tensor[idx]
        m = t.mean()
        s = t.std()
        t = t.sub_(m)
        t = t.div_(s+1e-6)
        tensor[idx] = t
        tensor = tensor.view(c, h, w)
        return tensor

def train(loader,net,head,criterion,optimizer, device, epoch=-1,flag = 0):
    net.train()
    head.train()
    if head != None:
        head.train()
    running_loss = 0.0
    avg_loss = 0
    i = 0
    running_c_loss = 0
    for i, data in enumerate(tqdm.tqdm(loader)):
        #print(".", end="", flush=True)
        images, labels,_ = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        features = net(images)
        if flag == 0:
            outputs,raw,C_loss = head(features,labels)
            loss = criterion(outputs, labels) + C_loss
            running_c_loss += C_loss.item()
        elif flag == 1:
            outputs,raw = head(features,labels)
            loss = criterion(outputs, labels)
        elif flag == 2:
            outputs = head(features,labels)
            loss = criterion(outputs, labels)
        elif flag == 3:
            outputs = head(features, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    #print(".", flush=True)
    avg_loss = running_loss / i
    avg_c_loss = running_c_loss/i
    print(
        f"Epoch: {epoch}, Step: {i}, " +
        f"Average Loss: {avg_loss:.4f}",
        f"Center Loss: {avg_c_loss:.4f}",
        flush=True
    )
    return avg_loss

def test(loader,net,head,criterion,device="cuda",top_accu=range(1,51,1),flag = 0):
    net.eval()
    head.eval()
    running_loss = 0.0
    num = 0
    accu = [0]*len(top_accu)
    for _, data in enumerate(loader):
        images, labels,_ = data
        images = images.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            features = net(images)
            if flag == 0:
                outputs,raw,C_loss = head(features, labels)
                loss = criterion(outputs, labels) + C_loss
            elif flag == 1:
                outputs, raw = head(features,labels)
                loss = criterion(outputs, labels)
            elif flag == 2:
                raw = head(features)
                loss = criterion(raw,labels)
            for ind, num_top in enumerate(top_accu):
                _, preds = torch.topk(raw.data,num_top,1)
                accu[ind] += (labels.repeat(num_top, 1).T==preds).sum().cpu().numpy()/raw.shape[0]

        running_loss += loss.item()
    accu = [x/num*100 for x in accu]
    return running_loss/num, accu
def feature_extraction(net,data_loader_test):
    net.eval()
    device = "cuda"
# feature extraction:
    featDB_test = []
    iddb_test = []
    iname_test = []
    with torch.no_grad():
        for batch_id, (data, target,name) in enumerate(data_loader_test):
            data = data.to(device)
            target = target.to(device)
            outs = net(data)
            codes = outs.cpu().detach().numpy()
            y = target.cpu().detach().numpy()
            if batch_id == 0:
                featDB_test = codes
                iddb_test = y
                iname_test = list(name)
            else:
                featDB_test = np.concatenate((featDB_test,codes), axis=0)
                iddb_test = np.concatenate((iddb_test,y))
                iname_test = iname_test+list(name)

    print('featDB_test.shape: ', featDB_test.shape)
    return featDB_test,iddb_test,iname_test
def get_acc_withCMC(featdbtest,labeltest,featDB_train,iddb_train):
    ntest = featdbtest.shape[0]
    test_cmc = np.zeros(50,dtype=float)
    for i in range(ntest):
        each_test_feat = featdbtest[i]
        label = labeltest[i]
        dis = cosine(each_test_feat.reshape(1, -1),featDB_train)
        match_label = iddb_train[np.argsort(dis)].reshape(-1)[::-1]
        matchres =  [*pd.unique(match_label)]
        for idx in range(50):
            test_cmc[idx] += label in matchres[:idx+1]
    cmc = 100 * (test_cmc/float(ntest))
    acc = cmc[0]
    return acc,cmc

def find_EER(fpr,tpr):
    eer = optimize.brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    #thresh = interpolate.interp1d(fpr,thresholds)(eer)
    return eer*100,list(fpr),list(tpr)

def get_eer(featdbtest,labeltest,featDB_train,iddb_train,ddh = False):
    # rank-1 acc
    # dataset EER of the test set (the gallery set is not used)
    s = []
    l = []
    n = featdbtest.shape[0]
    for i in range(n):
        feat = featdbtest[i]
        label = labeltest[i]
        dis = list(cosine(feat.reshape(1,-1),featDB_train).reshape(-1))
        matching_pair = list((iddb_train == label).reshape(-1))
        s += list(dis)
        l += matching_pair
    fpr, tpr, thresholds = metrics.roc_curve(l,s, pos_label=1, sample_weight=None,drop_intermediate=True)
    eer,fpr,tpr = find_EER(fpr,tpr)
    return eer,list(fpr),list(tpr)

def eval(net,galloader,valloader):
    net.eval()
    featDB_test, iddb_test,iname_test = feature_extraction(net,valloader)
    featDB_train, iddb_train,_ = feature_extraction(net,galloader)
    acc,cmc = get_acc_withCMC(featDB_test,iddb_test,featDB_train,iddb_train)
    eer,fpr,tpr = get_eer(featDB_test, iddb_test, featDB_train, iddb_train)
    return acc,eer,cmc,fpr,tpr

def training(net,head,
             data_pth,model_save_pth,rst_pth,rst_dir,
             total_epoch = 40,load_last = False,lr = 1e-4,val_epoch = 1,size=(128,128),
             milestones = [8,16,32,40,30,200],flag = 0,session_dataset = False):
    if load_last==True:
        with open(model_save_pth, 'rb') as file:
            loaded_params = torch.load(file)
        sub_dict = loaded_params["backbone"]
        net.load_state_dict(sub_dict)
        sub_dict = loaded_params["head"]
        head.load_state_dict(sub_dict)
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir)
    device = "cuda"
    imgsize = size
    batch_size = 32
    num_workers = 8
    total_epoch = total_epoch
    initial_lr = lr
    milestones = milestones
    validation_epochs = val_epoch
    net = net.to(device)
    try:
        learning_rates = [1e-5, 1e-4]
        param_groups = [
            {'params': net.spt_Layer.parameters(), 'lr': learning_rates[0]},
            {'params':list(chain(net.vgg_p4.parameters(),net.extra.parameters(),head.parameters())) , 'lr': learning_rates[1]},
        ]
        optimizer = torch.optim.Adam(param_groups)
    except:
        if head != None:
            head = head.to(device)
            params = list(net.parameters())+list(head.parameters())
        else:
            params = list(net.parameters())
        optimizer = torch.optim.Adam(params=params, lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)
    val_transform = T.Compose([
        T.Resize(imgsize),
        T.ToTensor(),
        NormSingleROI(outchannels=3)
    ])
    train_transform = T.Compose([
        T.Resize(imgsize),
        T.RandomChoice(transforms=[
            #T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
            RandAugment.RandAugment(3,15)
        ]),
        T.ToTensor(),
        NormSingleROI(outchannels=3)
    ])
    train_val_transform = transforms.Compose([
        transforms.Resize(imgsize),
        transforms.ToTensor(),
        NormSingleROI(outchannels=3),
    ])
    if session_dataset:
        Ds = Dataset_closeset_session
    else:
        Ds = Dataset_closeset
    train_dataset = Ds(data_pth,train_transform,True,False)
    print(train_dataset.counter,flush=True)
    train_val_dataset = Ds(data_pth,val_transform,True,True)
    val_dataset = Ds(data_pth,train_val_transform,False,True)
    print(f'--TrainSize:{len(train_dataset)}: ValSize:{len(val_dataset)}',flush=True)
    num_classes = len(train_dataset.labels)
    criterion = nn.CrossEntropyLoss().to(device)
    print(f'num_classes:{num_classes}',flush=True)
    train_loader = DataLoader(train_dataset,batch_size,num_workers=num_workers, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size,num_workers=num_workers, shuffle=False)
    train_val_loader = DataLoader(train_val_dataset,batch_size,num_workers=num_workers, shuffle=False)
    losses = []
    val_losses = []
    val_accu = []
    val_epoch = []
    highest_acc = 0
    lowest_loss = 100
    lowest_eer = 100
    best_cmc = []
    for epoch in range(total_epoch):
        epoch_start_time = time.time()
        loss = train(train_loader,net,head,criterion,optimizer,device=device, epoch=epoch,flag = flag)
        scheduler.step()
        print("time: {:2.2f}s".format(time.time() - epoch_start_time),
        flush=True)
        losses.append(loss)
        print("lr rate :{}".format(optimizer.param_groups[0]['lr']),
        flush=True)
        try:
            print("lr rate :{}".format(optimizer.param_groups[1]['lr']))
        except:
            pass
        if (epoch+1) % validation_epochs == 0:
            val_epoch.append(epoch+1)
            print("lr rate :{}".format(optimizer.param_groups[0]['lr']))
            accu, eer,cmc, fpr, tpr = eval(net,train_val_loader,val_loader)
            fpr = filter_long(fpr)
            tpr = filter_long(tpr)
            val_accu.append(accu)
            print(
                f"Epoch: {epoch}, " +
                f"Accuracy: {accu:.4f}, " +
                f"EER: {eer:.4f}, "
            )
            if (accu > highest_acc):
                highest_acc = accu
                best_cmc = cmc
            if (eer < lowest_eer):
                lowest_eer = eer
                best_tpr = tpr
                best_fpr = fpr

            rst_dic = {'acc': highest_acc, 'eer': lowest_eer, 'cmc': list(best_cmc), 'fpr': best_fpr,'tpr': best_tpr}
            with open(rst_pth, 'w') as f:
                json.dump(rst_dic, f)
            print("rst_saved")

if __name__ == '__main__':
    Session = {"HIT": False, "IITD": False, "MPD": True, "BJTU": False}
    Classes_Num = {"HIT": 324, "IITD": 460, "MPD": 400, "BJTU": 296}
    Data_Path = {"HIT": "/mnt/wx/data/ROIs/HIT_Ours"
                ,"IITD": "/mnt/wx/data/ROIs/IITD_Ours/",
                 "MPD": "/mnt/wx/data/ROIs/MPD_Ours",
                 "BJTU": "/mnt/wx/data/ROIs/BJTU_Ours",}

    DataSet_name = "HIT"
    Method_Name = "RLANN"
    net = RLANN().cuda()
    dict = torch.load("pretrained.pkl")
    net.load_state_dict(dict)
    head = ACPLoss(512, Classes_Num[DataSet_name]).cuda()
    data_pth = Data_Path[DataSet_name]
    if not os.path.exists(os.path.join("SavedWeights", DataSet_name)):
        os.makedirs(os.path.join("SavedWeights", DataSet_name))
    model_save_path = os.path.join("SavedWeights", DataSet_name, Method_Name + ".pkl")
    rst_dir = os.path.join("Result", "CloseSet", DataSet_name)
    if not os.path.exists(rst_dir):
        os.makedirs(rst_dir)
    rst_pth = os.path.join("Result", "CloseSet", DataSet_name, Method_Name + ".json")
    training(net, head, data_pth, model_save_path, rst_pth, rst_dir, flag=0, total_epoch=120, val_epoch=5,
             session_dataset=Session[DataSet_name])