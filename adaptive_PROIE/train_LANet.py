import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from adaptive_PROIE.LANet import LAnet
from dataset import Rotate_Angle_Dataset,Rotate_Angle_Visualize_Dataset
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import warnings
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def tensor_to_cv2(imgt):
    imgt = imgt.permute(1,2,0)
    imgt = imgt.detach().cpu().numpy()
    imgt = ((imgt)*255).astype(np.uint8)
    return imgt

def process_and_save_one_iter(raw_ipt,thetas,start_idx,floder):
    for idx,img in enumerate(raw_ipt):
        tensor_img = img
        tensor_theta = thetas[idx]
        img_cv2 = tensor_to_cv2(tensor_img)
        theta = tensor_theta.numpy()
        angle_degrees = np.degrees(theta*np.pi)
        angle_degrees = float(angle_degrees)
        img_gray = img_cv2[:,:,0]
        hmap = img_cv2[:,:,2]
        max_index = np.argmax(hmap)
        max_coords = np.unravel_index(max_index,hmap.shape)
        center = (int(max_coords[1]), int(max_coords[0]))
        img_bgr = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(img_bgr, center,2, (0, 255, 0), -1)
        img_rotate = img_bgr.copy()
        rotation_matrix = cv2.getRotationMatrix2D(center,angle_degrees,scale=1.0)
        rotated_image = cv2.warpAffine(img_rotate,rotation_matrix, (img_rotate.shape[1],img_rotate.shape[0]))
        horizontally_concatenated = cv2.hconcat([img_bgr,rotated_image])
        i_name = str(idx+start_idx)+".jpg"
        cv2.imwrite(os.path.join(floder,i_name),horizontally_concatenated)



def visualize(net,visualize_loader,epoch,visualize_base_pth):
    print("start_visualize")
    net.eval()
    idx = 0
    for i, data in enumerate(tqdm.tqdm(visualize_loader)):
        ipts = data.cuda()
        with torch.no_grad():
            theta = net(ipts)
        idx += i
        folder = os.path.join(visualize_base_pth,str(epoch))
        if not os.path.exists(folder):
            os.makedirs(folder)
        process_and_save_one_iter(ipts.detach().cpu(),theta.detach().cpu(),idx,folder)
    print("epoch",epoch,"visualized")

def main():
    net = LAnet().cuda()
    load = True
    model_pth = "/tmp/NTUv2detection/Theta_predict/train_rst/models/LANet_v1.pkl"
    net.cuda()
    initial_lr = 2.5e-4
    total_epoch = 40
    val_epoch = 10
    milestones = [10,20,30,40]
    model_save_file = "train_rst/models"
    if load == True:
        with open(model_pth, 'rb') as file:
            loaded_params = torch.load(file)
        sub_dict = loaded_params["LANet"]
        net.load_state_dict(sub_dict)

    data_pth = "/tmp/NTUv2detection/KPT_detect2/data/V1/kptdetect_v1"
    visualize_data_pth = "Datas/visualize_data"
    visualize_save_pth = "visualize/rsts"
    json_list_pth = "/tmp/NTUv2detection/KPT_detect2/data/V1/kptloc_v1.json"
    train_data = Rotate_Angle_Dataset(data_pth,json_list_pth,(56,56))
    visualize_data = Rotate_Angle_Visualize_Dataset(visualize_data_pth)
    print("start")
    print('train num:',len(train_data.file_name_list))
    train_bs = 64
    train_Loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
    visualize_Loader = DataLoader(dataset=visualize_data, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
    criterion = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    for epoch in range(total_epoch):
        if (epoch) % val_epoch == 0:
            visualize(net, visualize_Loader, epoch + 1, visualize_save_pth)
        epoch_start_time = time.time()
        scheduler.step()
        print("lr rate :{}".format(optimizer.param_groups[0]['lr']))
        net.train(mode=True)
        loss_sigma = 0.0
        for i, data in enumerate(tqdm.tqdm(train_Loader)):
            inputs, theta_lb = data
            inputs, theta_lb = inputs.cuda(),theta_lb.cuda().float()
            optimizer.zero_grad()
            theta = net(inputs)
            loss_all = 0
            loss_all += criterion(theta,theta_lb)
            loss_all.backward()
            optimizer.step()
            loss_sigma += loss_all.item()
        loss_avg = loss_sigma/(i+1)
        print("Training: Epoch[{:0>3}/{:0>3}]".format(epoch + 1, total_epoch),"avg_Loss:",loss_avg)
        print('time:',time.time()-epoch_start_time)
        ckp = {}
        ckp["LANet"] = net.state_dict()
        torch.save(ckp, os.path.join(model_save_file, 'LANet_v1.pkl'))
        print("model_saved")
    print("finish_train")

if __name__ == '__main__':
    main()