import os
import json
from torch.utils.data.dataset import Dataset
import torch
import cv2
import numpy as np
import random
import csv
from PIL import Image,ImageOps

def see_dist_map(label):
    dist_map = cv2.distanceTransform(label, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return  dist_map

def find_circle_inform_easy(mask):
    dist_map = see_dist_map(mask)
    _, max_val, _, center = cv2.minMaxLoc(dist_map)
    return center,max_val

def find_circle_inform_hard(mask):
    dist_map = see_dist_map(mask)
    _, max_val, _, center_flag = cv2.minMaxLoc(dist_map)
    good_val = max_val * 0.9
    good_region = dist_map.copy()
    good_region[good_region < good_val] = 0
    good_region[good_region >= good_val] == 255
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    center_x, center_y = center
    good_region = good_region.astype(np.uint8)
    inner_dist_map = cv2.distanceTransform(good_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = inner_dist_map[center_y, center_x]
    cv2.circle(good_region, center, int(radius), 0, -1)
    black_image = np.zeros_like(good_region)
    black_image[:center_y] = good_region[:center_y]
    black_image[black_image > 0] = 255
    contours, _ = cv2.findContours(black_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 用mask找到轮廓点组
    m = 0;
    m_area = 0
    for i in range(len(contours)):  # 检测出最大的轮廓
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m]  # .reshape(-1, 2)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx, cy)
    r = int(dist_map[cy][cx])
    return center, r

def cal_theta(rotated_points):
    points = np.array([rotated_points[1],rotated_points[2]])
    pt = points
    mean_pt = np.mean(pt, axis=0)
    mean_pt = (int(mean_pt[0]), int(mean_pt[1]))
    center = rotated_points[0]
    lenb = np.linalg.norm(np.array(center) - np.array(mean_pt))
    sintheta = (mean_pt[0] - center[0]) / lenb
    radians = np.arcsin(sintheta)
    return radians

def center_and_pad_image(input_img_cv2,kpts):
    height, width, _ = input_img_cv2.shape
    new_size = int(max(width, height))  # + 2 * border_width
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
    kpts[:, 0] += x_offset
    kpts[:, 1] += y_offset
    return padded_image, kpts

def generate_heatmap(keypoint_location, heatmap_size, variance):
    x, y = keypoint_location
    x_range = torch.arange(0, heatmap_size[1], 1)
    y_range = torch.arange(0, heatmap_size[0], 1)
    X, Y = torch.meshgrid(x_range, y_range)
    pos = torch.stack((X, Y), dim=2)
    # Calculate 2D Gaussian distribution
    heatmap = torch.exp(-(torch.sum((pos - torch.tensor([x, y]))**2, dim=2)) / (2.0 * variance**2))
    return heatmap

class Rotate_Angle_Dataset:
    def __init__(self, file_pth, json_list_pth, train_size=(56, 56)):
        self.file_pth = file_pth
        self.train_size = train_size
        with open(json_list_pth, "r") as f:
            data = json.load(f)
        self.data = data
        self.file_name_list = list(self.data.keys())
        self.augment = True

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, item):
        i_name = self.file_name_list[item]
        kpt_loc = self.data[i_name]
        img = cv2.imread(os.path.join(self.file_pth,i_name))
        try:
            h, w, _ = img.shape
        except Exception as e:
            print(e)
            print(i_name)
        mask = np.where((img == 0), 0, 255)[:, :, 2].astype(np.uint8)
        method = np.random.rand(1)
        if method < 0.5:
            center, r = find_circle_inform_easy(mask)
        else:
            center, r = find_circle_inform_hard(mask)
        h, w, _ = img.shape
        kpts = np.array([center, kpt_loc[0], kpt_loc[1]])
        rotate_theta = random.randint(-70, 70)  #
        img = cv2.resize(img, (4 * w, 4 * h))
        r_c = (2 * w, 2 * h)
        scale = 0.25
        kpts = 4 * (kpts)
        mat = cv2.getRotationMatrix2D(r_c, rotate_theta, scale=scale)  # random.uniform(0.8,1)
        rotated_img = cv2.warpAffine(img, mat, (4 * w, 4 * h))
        rotated_points = []
        for point in kpts:
            npt = np.array([point[0], point[1], 1])
            rotated_point = mat.dot(npt)
            rotated_points.append(rotated_point)
        rotated_points = np.array(rotated_points)
        mask = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask,0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        m = 0
        m_area = 0
        for i in range(len(contours)):  # 检测出最大的轮廓
            area = cv2.contourArea(contours[i])
            if area > m_area:
                m = i
                m_area = area
        contour = contours[m].reshape(-1, 2)
        xs, ys = np.int0(np.min(contour, 0))
        xe, ye = np.int0(np.max(contour, 0))
        rotated_img = rotated_img[ys:ye, xs:xe]
        rotated_points[:, 0] -= xs
        rotated_points[:, 1] -= ys
        radians = cal_theta(rotated_points)
        h, w, _ = rotated_img.shape
        top = 0
        bottom = int(rotated_points[0][1] - r)
        if bottom>top:
            cut_y = random.randint(top, bottom)
        else:
            cut_y = 0
        if int(rotated_points[0][0] + r)<w:
            cut_right = random.randint(int(rotated_points[0][0] + r), w)
        else:
            cut_right = w
        if int(rotated_points[0][0] - r) >1:
            cut_left = random.randint(0, int(rotated_points[0][0] - r))
        else:
            cut_left = 0

        rotated_img = rotated_img[cut_y:, cut_left:cut_right]
        rotated_points[:, 0] -= cut_left
        rotated_points[:, 1] -= cut_y
        rotated_img, rotated_points = center_and_pad_image(rotated_img, rotated_points)
        center = rotated_points[0]
        h, w, _ = rotated_img.shape
        center_changed = np.array([int(center[1] * 56 / w), int(center[0] * 56 / h)])
        center_hmap = generate_heatmap(center_changed, (56, 56), 2)
        mapped_radians = radians / np.pi
        radians = torch.tensor([mapped_radians])
        img = cv2.resize(rotated_img, self.train_size)
        img = np.transpose(img, (2, 0, 1)) / 255.
        img = torch.from_numpy(img.copy()).float()
        img[2] = center_hmap
        return img, radians


class Rotate_Angle_Visualize_Dataset:
    def __init__(self, file_pth, train_size=(56, 56)):
        self.file_pth = file_pth
        self.train_size = train_size
        self.file_name_list = os.listdir(self.file_pth)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, item):
        i_name = self.file_name_list[item]
        img = cv2.imread(os.path.join(self.file_pth,i_name))
        try:
            h, w, _ = img.shape
        except Exception as e:
            print(e)
            print(i_name)
        mask = np.where((img == 0), 0, 255)[:, :, 2].astype(np.uint8)
        center, r = find_circle_inform_hard(mask)
        h, w, _ = img.shape
        kpts = np.array([center])
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        m = 0
        m_area = 0
        for i in range(len(contours)):  # 检测出最大的轮廓
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
        center_changed = np.array([int(center[1] * 56 / w), int(center[0] * 56 / h)])
        center_hmap = generate_heatmap(center_changed, (56, 56), 2)
        img = cv2.resize(rotated_img, self.train_size)
        img = np.transpose(img, (2, 0, 1)) / 255.
        img = torch.from_numpy(img.copy()).float()
        img[2] = center_hmap
        return img
