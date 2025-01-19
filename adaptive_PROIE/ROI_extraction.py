import cv2
import numpy as np
import torch
from adaptive_PROIE.LANet import LAnet
import math
import os

save_dir = ''

def center_and_pad_image(input_img_cv2):
    height, width, _ = input_img_cv2.shape
    new_size = int(max(width, height))
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
    return padded_image

def tensor_to_cv2(imgt):
    imgt = imgt.permute(1,2,0)
    imgt = imgt.detach().cpu().numpy()
    imgt = ((imgt)*255).astype(np.uint8)
    return imgt#kpts

class KptDetector(object):
    def __init__(self,):
        self.net = LAnet().cuda()
        save_pth = r"contact_the_autor_for_modelweight"
        with open(save_pth, 'rb') as file:
            loaded_params = torch.load(file)
        sub_dict = loaded_params["LANet"]
        self.net.load_state_dict(sub_dict)
        self.net.cuda().eval()
    def forward(self,img):
        rst =self.net(img)
        return rst
def padding_img(img,padd):
    K = padd
    height, width, channels = img.shape
    new_height = height + 2 * K
    new_width = width + 2 * K
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    new_image[K:K + height, K:K + width, :] = img
    return new_image

def see_dist_map(label):
    h, w = label.shape
    contours, _ = cv2.findContours(label.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m].reshape(-1, 2)
    xs, ys = np.min(contour, 0)
    xe, ye = np.max(contour, 0)
    mid_h = (ys + ye) / 2
    label = np.zeros((label.shape[0], w)).astype('uint8')
    mask = cv2.fillPoly(label, [contours[m]], 255)
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return  dist_map,mask,mid_h

def get_center(mask):
    dist_map,mask,mid_h = see_dist_map(mask)
    _,max_val,_,center_flag = cv2.minMaxLoc(dist_map)
    if center_flag[1] < mid_h:
        dist_map = cv2.rotate(dist_map, cv2.ROTATE_180)
    good_val = max_val * 0.85
    good_region = dist_map.copy()
    good_region[good_region < good_val] = 0
    good_region[good_region >= good_val] == 255
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    center_x,center_y = center
    good_region = good_region.astype(np.uint8)
    inner_dist_map = cv2.distanceTransform(good_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = inner_dist_map[center_y,center_x]
    cv2.circle(good_region,center, int(radius), 0, -1)
    black_image = np.zeros_like(good_region)
    black_image[:center_y] = good_region[:center_y]
    black_image[black_image>0] = 255
    contours, _ = cv2.findContours(black_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m]
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx,cy)
    return center

def circle_better(img,mask,rate=1):
    dist_map,mask,mid_h = see_dist_map(mask)
    _,max_val,_,center_flag = cv2.minMaxLoc(dist_map)
    if center_flag[1] < mid_h:
        img = cv2.rotate(img, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        dist_map = cv2.rotate(dist_map, cv2.ROTATE_180)
    good_val = max_val * 0.85
    good_region = dist_map.copy()
    good_region[good_region < good_val] = 0
    good_region[good_region >= good_val] == 255
    _, radius, _, center = cv2.minMaxLoc(dist_map)
    center_x,center_y = center
    good_region = good_region.astype(np.uint8)
    inner_dist_map = cv2.distanceTransform(good_region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radius = inner_dist_map[center_y,center_x]
    cv2.circle(good_region,center, int(radius), 0, -1)
    black_image = np.zeros_like(good_region)
    black_image[:center_y] = good_region[:center_y]
    black_image[black_image>0] = 255
    contours, _ = cv2.findContours(black_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m]
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    center = (cx,cy)
    circle_map = np.zeros_like(img).astype(np.uint8)
    r = int(dist_map[cy][cx]) * rate
    left = center[0] - int(r)
    right = center[0] + int(r)
    top = center[1] - int(r)
    bottom = center[1] + int(r)
    cv2.circle(circle_map, center, int(r), (255, 255, 255), -1)
    img = cv2.bitwise_and(img,np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1))
    final_circle = cv2.bitwise_and(img, circle_map)
    pad = int((r - int(dist_map[cy][cx]))/2)
    final_circle = padding_img(final_circle,pad)
    final_circle = final_circle[pad+top+1:pad+bottom+1,pad+left+1:pad+right+1]
    return final_circle
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

detector = KptDetector()

def process_single_img_ipt(imgn):
    img = cv2.imread(os.path.join(root_dir,imgn))
    h, w, _ = img.shape
    img_for_detect = img.copy()
    mask = np.where((img[:,:,2]<20),0,255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    counter = contours[m].reshape(-1, 2)

    xs, ys = np.int0(np.min(counter, 0))
    xe, ye = np.int0(np.max(counter, 0))

    img = img[ys:ye, xs:xe]
    img_for_detect = img_for_detect[ys:ye, xs:xe]

    raw_img = center_and_pad_image(img)
    raw_img_detect = center_and_pad_image(img_for_detect)

    gray1_detect = cv2.cvtColor(raw_img_detect.copy(), cv2.COLOR_BGR2GRAY)
    gray1_detect = np.expand_dims(gray1_detect, axis=-1)
    gray1_detect = np.repeat(gray1_detect, 3, axis=-1)
    imgh, imgw, _ = gray1_detect.shape
    train_size = (56,56)
    img = cv2.resize(gray1_detect,train_size)
    img = np.transpose(img, (2, 0, 1)) / 255.
    img = torch.from_numpy(img.copy()).float()
    img = img.unsqueeze(0).cuda()
    rst=detector.forward(img)
    img_cv2 = tensor_to_cv2(img[0].cpu())
    points = rst.detach().cpu().reshape(-1, 2)
    points = (points + 1) * (imgh/2)
    raw_mask = np.where((raw_img[:,:,2]<20),0,255).astype(np.uint8)
    pt = points.numpy()
    mean_pt = np.mean(pt, axis=0)
    mean_pt = (int(mean_pt[0]),int(mean_pt[1]))
    center = get_center(raw_mask)
    lenb = np.linalg.norm(np.array(center) - np.array(mean_pt))
    sintheta =  (mean_pt[0] - center[0])/lenb
    radians = np.arcsin(sintheta)
    angle_degrees = np.degrees(radians)
    height, width = raw_img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center,angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(raw_img, rotation_matrix, (width, height))
    rotated_mask = cv2.warpAffine(raw_mask, rotation_matrix, (width, height))
    final_circle, visualize_circle = circle_better(rotated_image,rotated_mask,rate=1.1)
    imgn = imgn.split(".")[0]
    h, w, _ = visualize_circle.shape
    try:
        for angle in range(-30, 30, 3):
            square_roi,circle_roi = get_inter_square(final_circle, angle)
            save_iname = imgn + "_" + str(angle) + ".jpg"
            square_roi_128 = cv2.resize(square_roi,(128,128))
            cv2.imwrite(os.path.join(save_dir, save_iname), square_roi_128)
    except Exception as e:
        print(e)

if __name__ == '__main__':

    root_dir = r"/tmp/data/HIT"
    imgns = os.listdir(root_dir)
    cnt = 0
    total_len = len(imgns)
    for imgn in imgns:
        cnt += 1
        print(cnt)
        if cnt%100 == 0:
            print(cnt)
        process_single_img_ipt(imgn)




