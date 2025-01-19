import torch
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from models import LAnet
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import warnings
import math
import tqdm
import warnings
from utills import *
warnings.filterwarnings("ignore", category=UserWarning)

def generate_heatmap(keypoint_location, heatmap_size, variance):
    x, y = keypoint_location
    x_range = torch.arange(0, heatmap_size[1], 1)
    y_range = torch.arange(0, heatmap_size[0], 1)
    X, Y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((X, Y), dim=2)
    heatmap = torch.exp(-(torch.sum((pos - torch.tensor([x, y]))**2, dim=2)) / (2.0 * variance**2))
    return heatmap

def center_and_pad_image(input_img_cv2,kpts):
    height, width, _ = input_img_cv2.shape
    new_size = int(max(width, height))
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
    kpts[:, 0] += x_offset
    kpts[:, 1] += y_offset
    return padded_image, kpts

class ThetaPreDetector(object):
    def __init__(self,):
        self.net = LAnet().cuda()
        save_pth = r"LANet_v1.pkl"
        with open(save_pth, 'rb') as file:
            loaded_params = torch.load(file)
        sub_dict = loaded_params["LANet"]
        self.net.load_state_dict(sub_dict)
        self.net.cuda().eval()
    def forward(self,img):
        rst =self.net(img)
        return rst

def generate_net_ipt(root,imgn):
    img = cv2.imread(os.path.join(root,imgn))
    mask = np.where((img == 0), 0, 255)[:, :, 2].astype(np.uint8)
    center, r = find_circle_inform_hard(mask)
    h, w, _ = img.shape
    kpts = np.array([center])
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i
            m_area = area
    contour = contours[m].reshape(-1, 2)
    xs, ys = np.int0(np.min(contour, 0))
    xe, ye = np.int0(np.max(contour, 0))
    rotated_img = img[ys:ye, xs:xe]
    kpts[:, 0] -= xs
    kpts[:, 1] -= ys
    rotated_img, rotated_points = center_and_pad_image(rotated_img, kpts)
    center = rotated_points[0]
    h, w, _ = rotated_img.shape
    raw_img = rotated_img.copy()
    center_changed = np.array([int(center[1] * 56 / w), int(center[0] * 56 / h)])
    center_hmap = generate_heatmap(center_changed, (56, 56), 2)
    img = cv2.resize(rotated_img,(56,56))
    img = np.transpose(img, (2, 0, 1)) / 255.
    img = torch.from_numpy(img.copy()).float()
    img[2] = center_hmap
    return img,raw_img,(int(center[1]), int(center[0]))

def get_inter_square(img,Rotate_theta):
    h,w,_ = img.shape
    center = (w//2,h//2)
    width = math.sqrt(2)*(w/2)/2
    left = center[0] - int(width)
    right = center[0] + int(width)
    top = center[1] - int(width)
    bottom = center[1] + int(width)
    mat = cv2.getRotationMatrix2D(center,Rotate_theta,scale=1)
    rotated_img = cv2.warpAffine(img,mat,(w,h))
    square_roi = rotated_img[top:bottom,left:right]
    return square_roi,rotated_img

detector = ThetaPreDetector()

def process_single_img_ipt(save_dir_visualize,save_dir_square,root_dir,imgn):
    tensor_for_ipt,raw_img,r_center = generate_net_ipt(root_dir,imgn)
    tensor_for_ipt = tensor_for_ipt.unsqueeze(0).cuda()
    theta = detector.forward(tensor_for_ipt).detach().cpu()[0]
    theta = theta.numpy()
    angle_degrees = np.degrees(theta * np.pi)
    angle_degrees = float(angle_degrees)
    rotation_matrix = cv2.getRotationMatrix2D(r_center,angle_degrees, scale=1.0)
    rawimg = cv2.warpAffine(raw_img, rotation_matrix,(raw_img.shape[1], raw_img.shape[0]))
    rawimg = padding_img(rawimg,20)
    raw_mask = np.where((rawimg <= 10),0,255)[:, :, 2].astype(np.uint8)
    visualize_img = rawimg.copy()
    final_circle, visualize_circle = circle_better(rawimg,visualize_img,raw_mask,rate=1.1)
    imgn = imgn.split(".")[0]
    h,w, _ = visualize_circle.shape
    visualize_save_iname = imgn + ".jpg"
    visualize_circle = cv2.resize(visualize_circle, (int(200 * w / h), 200))
    cv2.imwrite(os.path.join(save_dir_visualize,visualize_save_iname),visualize_circle)
    try:
        for angle in range(-30, 30, 3):
            square_roi,circle_roi = get_inter_square(final_circle, angle)
            save_iname = imgn + "_" + str(angle) + ".jpg"
            square_roi_128 = cv2.resize(square_roi,(128,128))
            cv2.imwrite(os.path.join(save_dir_square, save_iname), square_roi_128)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    root_dir = r"/tmp/NTUv2detection/Theta_predict/Datas/HIT"#
    save_dir_visualize = "Datas/ROIS/MPD/visualize"
    save_dir_square = "Datas/ROIS/HIT"
    mask_save_dir = "Datas/ROIS/HIT/visualize_mask"
    imgns = os.listdir(root_dir)
    total_len = len(imgns)
    for imgn in tqdm.tqdm(imgns):
        process_single_img_ipt(save_dir_visualize,save_dir_square,root_dir,imgn)

