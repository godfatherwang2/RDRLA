import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import numpy as np
import random
from matplotlib import pyplot as plt
import cv2
import json
import os
import torch
import re
#from torch._six import container_abcs, string_classes, int_classes

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

angles = [x for x in range(0,40,360)]

class FCNDataset(Dataset):
    def __init__(self, listfile,root='', trainsize=(384,320),phase='train', n_class=2, crop=False, flip_rate=0.5,rotate = False):
        with open(listfile, "r") as f:
            self.data=json.load(f)
        self.n_class=n_class
        self.flip_rate=flip_rate
        self.crop=crop
        self.root=root
        self.rotate = False
        self.trainsize=None
        if phase == "train":
            self.trainsize=trainsize
            self.data=self.data[1::4]+self.data[2::4]+self.data[3::4]
            if rotate == True:
                self.rotate = True
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.trainsize=trainsize
            self.data=self.data[::4]
        elif phase == "test":
            self.flip_rate = 0.
            self.crop = False
            self.trainsize=None
            self.data =self.data[::4]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data[idx][0]
        img        = cv2.imread(self.root+img_name+'_0.png')
        label_name = self.data[idx][1]
        label      = np.load(self.root+label_name+'_0.npy')
        try:
            h, w, _ = img.shape
        except:
            print(self.root + img_name + '_0.png')

        if self.rotate == True:
            angle = angles[random.randint(0,len(angles)-1)]
            img,label = rotate(img,label,angle=angle)

        if self.trainsize is None:
            w=int(w/32)*32
            h=int(h/32)*32
            img=cv2.resize(img,(w,h))
            label=cv2.resize(label.astype('float'),(w,h))
            label=label>0.5
        else:
            if self.crop and h>self.trainsize[0] and w > self.trainsize[1]:
                top   = random.randint(0, h - self.trainsize[0])
                left  = random.randint(0, w - self.trainsize[1])
                img   = img[top:top + self.trainsize[0], left:left + self.trainsize[1]]
                label = label[top:top + self.trainsize[0], left:left + self.trainsize[1]]
            else:
                img=cv2.resize(img,(self.trainsize[1],self.trainsize[0]))
                label=cv2.resize(label.astype('float'),(self.trainsize[1],self.trainsize[0]))
                label=label>0.5


        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        img = np.transpose(img, (2, 0, 1)) / 255.


        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        sample = {'X': img, 'Y': target, 'l': label}  #target(2,288,192)#label(288,192)

        return sample

class FCNDataset_with_augment(Dataset):
    def __init__(self, listfile,root='',backgroundroot = '' ,trainsize=(384,320),phase='train', n_class=2, if_vit = False,crop=False, flip_rate=0.5,bright = False,pure_background = False,use_Cr = False,use_R=False,rotate = False,show = False):
        with open(listfile, "r") as f:
            self.data = json.load(f)
        self.n_class = n_class
        self.flip_rate = flip_rate
        self.crop = crop
        self.show = show
        self.bright = bright
        self.angles = [x for x in range(-180, 180, 5)]
        self.root = root
        self.backgroundroot = backgroundroot
        self.background_list = os.listdir(backgroundroot)
        self.pureimgroot = "/data1/wx/palm/detection/tools/pureimgs"
        self.pureimg_list = os.listdir(self.pureimgroot)
        self.pure = pure_background
        self.rotate = rotate
        self.trainsize = None
        self.phase = phase
        self.use_Cr = use_Cr
        self.use_R = use_R
        self.if_vit = if_vit
        true_list= []
        for item in self.data:
            img_name = item[0]
            img_pth = self.root + img_name + '_0.png'
            if os.path.exists(img_pth):
                true_list.append(item)
        self.data = true_list

        if phase == "train":
            self.trainsize = trainsize
            #self.data = self.data[1::4] + self.data[2::4] + self.data[3::4]
            if rotate == True:
                self.rotate = True
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.trainsize = trainsize
            self.data = self.data[::4]
        elif phase == "test":
            self.flip_rate = 0.
            self.crop = False
            self.trainsize = None
            self.data = self.data[::4]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        try:
            if img_name[-6:] == "_0.png":
                img = cv2.imread(self.root + img_name)
            else:
                img = cv2.imread(self.root + img_name + '_0.png')
            label_name = self.data[idx][1]
            if label_name[-6:] == "_0.npy":
                label = np.load(self.root + label_name)
            else:
                label = np.load(self.root + label_name + '_0.npy')
            if self.phase == "train":
                try:
                    h, w, _ = img.shape
                except:
                    print(self.root + img_name + '_0.png')
                mask = np.where(label==True,255,label).astype(np.float32)
                mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB).astype(np.uint8)
                if img.shape != mask.shape:
                    h,w,_ = img.shape
                    img = img[:,:int(w/2)]
                if self.bright:
                    if_bright = random.randint(0,2)
                    if if_bright == 1:
                        img,_ = bright_augment(img,0.3,True)
                palm = cv2.bitwise_and(img,mask)
                angle = self.angles[random.randint(0,len(self.angles)-1)]
                if self.pure == False:
                    background = self.background_list[random.randint(0,len(self.background_list)-1)]
                    background = cv2.imread(os.path.join(self.backgroundroot,background))
                    if self.bright:
                        background, _ = bright_augment(background, 0.3)
                else:
                    if_pure = random.randint(0,10)
                    if if_pure < 2:
                        background = self.pureimg_list[random.randint(0,len(self.pureimg_list)-1)]
                        background = cv2.imread(os.path.join(self.pureimgroot, background))
                    else:
                        background = self.background_list[random.randint(0, len(self.background_list) - 1)]
                        background = cv2.imread(os.path.join(self.backgroundroot, background))
                        if self.bright:
                            background,_ = bright_augment(background,0.3)
                if_rotate = random.randint(0,4)
                if if_rotate != 0:
                    r_palm,r_mask = rotate(img=palm,gt=mask,angle=angle)
                else:
                    r_palm = palm
                    r_mask = mask
                #plt.subplot(211),plt.imshow(r_mask)
                #plt.subplot(212),plt.imshow(r_palm)
                img,label = add_background(r_palm,r_mask,background)
                #print(r_palm.shape,r_mask.shape)
                #palm =
                if self.trainsize is None:
                    w = int(w / 32) * 32
                    h = int(h / 32) * 32
                    img = cv2.resize(img, (w, h))
                    label = cv2.resize(label.astype('float'), (w, h))
                    label = label > 0.5
                else:
                    '''
                    if self.crop and h > self.trainsize[0] and w > self.trainsize[1]:
                        top = random.randint(0, h - self.trainsize[0])
                        left = random.randint(0, w - self.trainsize[1])
                        img = img[top:top + self.trainsize[0], left:left + self.trainsize[1]]
                        label = label[top:top + self.trainsize[0], left:left + self.trainsize[1]]
                    '''
                    if self.crop:
                        h,w,_ = img.shape
                        croprange = 0.15
                        top = int(random.uniform(0,croprange)*h)
                        bottom = int((1-random.uniform(0,croprange))*h)
                        left = int(random.uniform(0,croprange)*w)
                        right = int((1-random.uniform(0,croprange))*w)
                        img = img[top:bottom,left:right]
                        label = label[top:bottom,left:right]
                        img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
                        label = cv2.resize(label.astype('float'), (self.trainsize[1], self.trainsize[0]))
                        label = label > 0.5
                    else:
                        img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
                        label = cv2.resize(label.astype('float'), (self.trainsize[1], self.trainsize[0]))
                        label = label > 0.5
                if random.random() < self.flip_rate:
                    img = np.fliplr(img)
                    label = np.fliplr(label)
            else:
                mask = np.where(label == True, 255, label).astype(np.float32)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                if img.shape != mask.shape:
                    h,w,_ = img.shape
                    img = img[:,:int(w/2)]
                palm = cv2.bitwise_and(img, mask)
                background = self.background_list[random.randint(0, len(self.background_list) - 1)]
                background = cv2.imread(os.path.join(self.backgroundroot, background))
                img, label = add_background(palm,mask,background)
                img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
                label = cv2.resize(label.astype('float'),(self.trainsize[1], self.trainsize[0]))
                label = label > 0.5

            if self.if_vit:
                label = cv2.resize(label.astype('float'),(self.trainsize[1]//8,self.trainsize[0]//8))
            if self.use_R:
                (B, G, R) = cv2.split(img)
                img = R
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif self.use_Cr:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # 把图像转换到YUV色域
                (y, cr, cb) = cv2.split(ycrcb)
                # cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 对cr通道分量进行高斯滤波
                img = cr
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if self.show == True:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                plt.subplot(211),plt.imshow(img)
                plt.subplot(212),plt.imshow(label)
                plt.show()

            img = np.transpose(img, (2, 0, 1)) / 255.
            # convert to tensor
            img = torch.from_numpy(img.copy()).float()
            label = torch.from_numpy(label.copy()).long()
            # create one-hot encoding
            h, w = label.size()
            target = torch.zeros(self.n_class, h, w)
            for c in range(self.n_class):
                target[c][label == c] = 1
            sample = {'X': img, 'Y': target, 'l': label}  # target(2,288,192)#label(288,192)
            return sample
        except Exception as e:
            print(e)
            print(img_name)
            return None

def bright_augment(image, brightness,nolighter = False):
    if nolighter:
        factor = 1.0 + random.uniform(-1.0*brightness, 0)
    else:
        factor = 1.0 + random.uniform(-1.0 * brightness,brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image, factor


def rotate_nopers(img, gt, angle=10):
    """
    angle: 旋转的角度
    """
    base_m1 = np.random.uniform(0, 0.0003)
    base_m2 = np.random.uniform(0, 0.0003)
    assert img.shape[:2] == gt.shape[:2]
    h, w = img.shape[:2]
    k = 1200/4*h
    base_m1 *= k
    base_m2 *= k
    img = cv2.resize(img,(4*w,4*h))
    gt = cv2.resize(gt,(4*w,4*h))
    center = (2*w, 2*h)
    scale = 0.25
    mat = cv2.getRotationMatrix2D(center, angle, scale=scale)#random.uniform(0.8,1)
    perspective_matrix = np.column_stack([mat, [base_m1,base_m2, 1]])
    rotated_img = cv2.warpPerspective(img,perspective_matrix,(4*w, 4*h))
    rotated_gt = cv2.warpPerspective(gt,perspective_matrix, (4*w, 4*h))
    final_gt = rotated_gt
    rotated_gt = cv2.cvtColor(rotated_gt,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    contours, _ = cv2.findContours(rotated_gt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):  # 检测出最大的轮廓
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m].reshape(-1,2)
    xs, ys = np.int0(np.min(contour, 0))
    xe, ye = np.int0(np.max(contour, 0))
    rotated_img = rotated_img[ys:ye,xs:xe]
    final_gt = final_gt[ys:ye,xs:xe]
    return rotated_img,final_gt

def rotate(img, gt, angle=60):
    base_m1 = np.random.uniform(0, 0.00015)
    base_m2 = np.random.uniform(0, 0.00015)
    assert img.shape[:2] == gt.shape[:2]
    h, w = img.shape[:2]
    k = 1200/(4*h)
    base_m1 *= k
    base_m2 *= k

    img = cv2.resize(img,(4*w,4*h))
    gt = cv2.resize(gt,(4*w,4*h))
    center = (2*w, 2*h)
    scale = 0.25
    mat = cv2.getRotationMatrix2D(center, angle, scale=scale)#random.uniform(0.8,1)
    perspective_matrix = np.vstack([mat,np.array([base_m1,base_m2, 1])])


    rotated_img = cv2.warpPerspective(img,perspective_matrix,(4*w, 4*h))
    rotated_gt = cv2.warpPerspective(gt,perspective_matrix, (4*w, 4*h))
    final_gt = rotated_gt
    rotated_gt = cv2.cvtColor(rotated_gt,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    contours, _ = cv2.findContours(rotated_gt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = 0;
    m_area = 0
    for i in range(len(contours)):  # 检测出最大的轮廓
        area = cv2.contourArea(contours[i])
        if area > m_area:
            m = i;
            m_area = area
    contour = contours[m].reshape(-1,2)
    xs, ys = np.int0(np.min(contour, 0))
    xe, ye = np.int0(np.max(contour, 0))
    rotated_img = rotated_img[ys:ye,xs:xe]
    final_gt = final_gt[ys:ye,xs:xe]
    return rotated_img,final_gt

def add_obj(obj,background,mask,startx,starty):
    bg = background.copy()
    h,w,_ = obj.shape
    bg[starty:starty+h,startx:startx+w,:] = bg[starty:starty+h,startx:startx+w,:]*~mask + (obj * mask)
    return bg

def add_background(img,g_mask,background):
    h,w,_ = img.shape
    background_size_rate = random.uniform(1,1.5)
    background_h = int(h*background_size_rate)
    background_w = int(w*background_size_rate)
    if (background_w == w) or (background_h == h):
        start_left_top_x = 0
        start_left_top_y = 0
    else:
        start_left_top_x = random.randint(0,background_w-w)
        start_left_top_y = random.randint(0,background_h-h)
    palm_img = img
    label = g_mask
    if min(h,w)>=300:
        label = np.where((label[:, :, 2] < 20), 0, 255).astype(np.uint8)
        label = cv2.cvtColor(label,cv2.COLOR_GRAY2BGR)

        kernel_size = (7,7)
        kernel = np.ones(kernel_size, np.uint8)
        label = cv2.erode(label, kernel, iterations=1)
    label_bool = label[:,:,:] != 0
    background = cv2.resize(background,(background_w,background_h))
    mask_boolean = np.zeros((background_h,background_w))
    mask_boolean[start_left_top_y:start_left_top_y+h,start_left_top_x:start_left_top_x+w] = (g_mask[:,:,0] != 0)
    gen_img = add_obj(palm_img,background,label_bool,start_left_top_x,start_left_top_y)
    return gen_img,mask_boolean

def cropping_together(img,g_mask,crop_rate = 0.6):
    k = np.random.uniform(crop_rate,1)
    height, width = img.shape[:2]
    # 计算裁剪后的新高度和宽度
    new_height = int(height * k)
    new_width = int(width * k)
    # 随机生成裁剪起始点
    start_row = np.random.randint(0, height - new_height + 1)
    start_col = np.random.randint(0, width - new_width + 1)
    cropped_img = img[start_row:start_row + new_height, start_col:start_col + new_width]
    cropped_mask = g_mask[start_row:start_row + new_height, start_col:start_col + new_width]
    return cropped_img,cropped_mask


def show_batch(batch):
    img_batch = batch['X']
    batch_size = len(img_batch)

    grid = utils.make_grid(img_batch)
    grid_mask = utils.make_grid(batch['Y'])
    plt.subplot(2,1,1)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
    plt.subplot(2,1,2)
    plt.imshow(grid_mask[1])
    plt.title('Batch from dataloader')


def random_crop_rectangle(image, aspect_ratio_range=(0.5, 2.0)):
    # 获取原始图像的高度和宽度
    height, width = image.shape[:2]
    # 计算 fixed_width 的范围
    min_fixed_width = int(0.3 * width)
    max_fixed_width = width
    # 随机生成一个宽度在一定范围内的值
    fixed_width = np.random.randint(min_fixed_width, max_fixed_width + 1)
    # 随机生成长宽比
    aspect_ratio = np.random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
    # 计算裁剪后的新高度
    new_height = int(fixed_width / aspect_ratio)
    # 确保新高度不超过原始高度
    new_height = min(new_height, height)
    # 随机生成裁剪起始点
    start_row = np.random.randint(0, height - new_height + 1)
    start_col = np.random.randint(0, width - fixed_width + 1)
    # 执行裁剪
    cropped_image = image[start_row:start_row + new_height, start_col:start_col + fixed_width]
    return cropped_image

def paired_random_perspective_transform(image, m31_range=(-0.2, 0.2), m32_range=(-0.1, 0.1)):
    # 生成随机的透视变换参数
    m31 = np.random.uniform(m31_range[0], m31_range[1])
    m32 = np.random.uniform(m32_range[0], m32_range[1])

    # 构建透视变换矩阵
    perspective_matrix = np.array([[1, 0, m31],
                                   [0, 1, m32],
                                   [0, 0, 1]])

    # 应用透视变换
    result_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))

    return result_image

def collate_fn(batch):
    '''
     collate_fn (callable, optional): merges a list of samples to form a mini-batch.
     该函数参考touch的default_collate函数，也是DataLoader的默认的校对方法，当batch中含有None等数据时，
     默认的default_collate校队方法会出现错误
     一种的解决方法是：
     判断batch中image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    :param batch:
    :return:
    '''
    r"""Puts each data field into a tensor with outer dimension batch size"""
    # 这里添加：判断image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
    if isinstance(batch, list):
        batch = [(image, image_id) for (image, image_id) in batch if image is not None]
    if batch==[]:
        return (None,None)

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return collate_fn([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    raise TypeError((error_msg_fmt.format(type(batch[0]))))

class TopFormerDataset_with_augment(Dataset):
    def __init__(self, listfile,root='',backgroundroot = '' ,trainsize=(448,448),phase='train', n_class=2, if_vit = False,crop=False, flip_rate=0.5,bright = False,pure_background = False,
                 use_Cr = False,use_R=False,rotate = False,show = False,gray = False):
        with open(listfile, "r") as f:
            self.data = json.load(f)
        self.gray = gray
        self.n_class = n_class
        self.flip_rate = flip_rate
        self.crop = crop
        self.show = show
        self.bright = bright
        self.angles = [x for x in range(-180, 180, 5)]
        self.root = root
        self.backgroundroot = backgroundroot
        self.background_list = os.listdir(backgroundroot)
        self.pureimgroot = "/mnt/sda1/pureimgs"
        self.pureimg_list = os.listdir(self.pureimgroot)
        self.pure = pure_background
        self.rotate = rotate
        self.trainsize = trainsize
        self.phase = phase
        self.use_Cr = use_Cr
        self.use_R = use_R
        self.if_vit = if_vit
        true_list= []
        for item in self.data:
            img_name = item[0]
            if img_name[-6:] != '_0.png':
                img_pth = self.root + img_name + '_0.png'
                if os.path.exists(img_pth):
                    true_list.append(item)
            else:
                img_pth = self.root + img_name
                if os.path.exists(img_pth):
                    true_list.append(item)
        self.data = true_list
        if phase == "train":
            self.trainsize = trainsize
            #self.data = self.data[1::4] + self.data[2::4] + self.data[3::4]
            if rotate == True:
                self.rotate = True
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False
            self.trainsize = trainsize
            self.data = self.data[::5]
        elif phase == "test":
            self.flip_rate = 0.
            self.crop = False
            self.trainsize = trainsize
            self.data = self.data[::5]

    def get_mean_and_std(self,image):
        b, g, r = cv2.split(image)
        non_zero_pixels = np.where((b > 0) & (g > 0) & (r > 0))
        # 提取非零像素的值
        non_zero_values_b = b[non_zero_pixels]
        non_zero_values_g = g[non_zero_pixels]
        non_zero_values_r = r[non_zero_pixels]
        # 计算均值和方差
        mean_value_b, mean_value_g, mean_value_r = np.mean(non_zero_values_b), np.mean(non_zero_values_g), np.mean(
            non_zero_values_r)
        std_value_b, std_value_g, std_value_r = np.std(non_zero_values_b), np.std(non_zero_values_g), np.std(
            non_zero_values_r)
        x_mean = np.hstack(np.around([mean_value_b, mean_value_g, mean_value_r], 2))
        x_std = np.hstack(np.around([std_value_b, std_value_g, std_value_r], 2))
        return x_mean, x_std

    def color_transfer(self,sc, dc):
        sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
        s_mean, s_std = self.get_mean_and_std(sc)
        dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
        t_mean, t_std = self.get_mean_and_std(dc)
        img_n = ((sc - s_mean) * (t_std / (s_std+1e-6))) + t_mean
        np.putmask(img_n, img_n > 255, 255)
        np.putmask(img_n, img_n < 0, 0)
        dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
        return dst

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        if img_name[-6:] == "_0.png":
            img = cv2.imread(self.root + img_name)
        else:
            img = cv2.imread(self.root + img_name + '_0.png')
        try:
            h, w, _ = img.shape
        except Exception as e:
            print(self.root + img_name,"Dont exists")
            print(e)
            return None
        label_name = self.data[idx][1]
        if label_name[-6:] == "_0.npy":
            label = np.load(self.root + label_name)
        else:
            label = np.load(self.root + label_name + '_0.npy')
        if self.phase == "train":
            h, w, _ = img.shape
            mask = np.where(label==True,255,label).astype(np.float32)
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB).astype(np.uint8)
            if img.shape != mask.shape:
                h,w,_ = img.shape
                img = img[:,:int(w/2)]
            if_Aug = random.randint(0, 5)
            if if_Aug == 0:
                if self.bright:
                    if_bright = random.randint(0,2)
                    if if_bright == 1:
                        img,_ = bright_augment(img,0.3,True)
                palm = cv2.bitwise_and(img,mask)
                angle = self.angles[random.randint(0,len(self.angles)-1)]
                if_pure = random.randint(0,10)
                if if_pure < 2:
                    background = self.pureimg_list[random.randint(0,len(self.pureimg_list)-1)]
                    background = cv2.imread(os.path.join(self.pureimgroot, background))
                else:
                    background = self.background_list[random.randint(0, len(self.background_list) - 1)]
                    background = cv2.imread(os.path.join(self.backgroundroot, background))
                background = random_crop_rectangle(background)
                background,_ = bright_augment(background,0.3)
                palm, _ = bright_augment(palm, 0.3)
                color_transfer_n = random.randint(0,4)
                if color_transfer_n == 0: #transfer_img_to_backround
                    palm = self.color_transfer(palm,background)
                elif color_transfer_n == 1:
                    background = self.color_transfer(background,palm)
                if_rotate = random.randint(0,4)
                if if_rotate != 0:
                    r_palm,r_mask = rotate(img=palm,gt=mask,angle=angle)
                else:
                    r_palm = palm
                    r_mask = mask
                #r_palm,r_mask = cropping_together(r_palm,r_mask)
                img,label = add_background(r_palm,r_mask,background)
                if self.trainsize is None:
                    w = int(w / 32) * 32
                    h = int(h / 32) * 32
                    img = cv2.resize(img, (w, h))
                    label = cv2.resize(label.astype('float'), (w, h))
                    label = label > 0.5
                else:
                    h, w, _ = img.shape
                    croprange = 0.3
                    top = int(random.uniform(0, croprange) * h)
                    bottom = int((1 - random.uniform(0, croprange)) * h)
                    left = int(random.uniform(0, croprange) * w)
                    right = int((1 - random.uniform(0, croprange)) * w)
                    img = img[top:bottom, left:right]
                    label = label[top:bottom, left:right]
                    img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
                    label = cv2.resize(label.astype('float'), (self.trainsize[1], self.trainsize[0]),interpolation=cv2.INTER_NEAREST)
                    label = label > 0.5
                if random.random() < self.flip_rate:
                    img = np.fliplr(img)
                    label = np.fliplr(label)
            else:
                mask = np.where(label == True, 255, label).astype(np.float32)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                if img.shape != mask.shape:
                    h, w, _ = img.shape
                    img = img[:, :int(w / 2)]
                palm = cv2.bitwise_and(img, mask)
                background = self.background_list[random.randint(0, len(self.background_list) - 1)]
                background = cv2.imread(os.path.join(self.backgroundroot, background))
                img, label = add_background(palm, mask, background)
                img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
                label = cv2.resize(label.astype('float'), (self.trainsize[1], self.trainsize[0]))
                label = label > 0.5
        else:
            mask = np.where(label == True, 255, label).astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            if img.shape != mask.shape:
                h,w,_ = img.shape
                img = img[:,:int(w/2)]
            palm = cv2.bitwise_and(img, mask)
            background = self.background_list[random.randint(0, len(self.background_list) - 1)]
            background = cv2.imread(os.path.join(self.backgroundroot, background))
            img, label = add_background(palm,mask,background)
            img = cv2.resize(img, (self.trainsize[1], self.trainsize[0]))
            label = cv2.resize(label.astype('float'),(self.trainsize[1], self.trainsize[0]))
            label = label > 0.5
        if self.if_vit:
            label = cv2.resize(label.astype('float'),(self.trainsize[1]//8,self.trainsize[0]//8),interpolation=cv2.INTER_NEAREST)
        if self.gray:
            img = cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        img = np.transpose(img, (2, 0, 1)) / 255.
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        sample = {'X': img, 'Y': target, 'l': label}  #target(2,288,192)#label(288,192)
        return sample

def tensor_to_cv2(imgt):
    imgt = imgt.permute(1,2,0)
    imgt = imgt.detach().cpu().numpy()
    imgt = ((imgt)*255).astype(np.uint8)
    return imgt#kpts



if __name__ == "__main__":

    dataset_list = '../ALL.json'
    dataset_path = '/PalmImgs'
    background_path = '/backgroundimgs'
    train_data = TopFormerDataset_with_augment(listfile = dataset_list, root=dataset_path,if_vit = True,show=False,
                        backgroundroot=background_path,phase="train",trainsize=(448,448),crop=True,rotate=True)
    for i in range(1,15000,100):
        data_pairs = train_data[i]
        img = data_pairs['X']
        img_cv2=tensor_to_cv2(img)
        label = data_pairs['l']
        label = label.unsqueeze(0).repeat(3,1,1)
        label_cv2 = tensor_to_cv2(label)
        cv2.imwrite("visualzie/"+str(i)+".jpg",img_cv2)
        cv2.imwrite("visualzie/"+str(i)+"_lb.jpg",label_cv2)





    #print(i, sample['X'].size(), sample['Y'].size())
    # FCNDataset_with_augment(listfile = dataset_list, root=dataset_path,if_vit = True,show=False,
    # backgroundroot=background_path,phase="train",trainsize=(448,448),crop=False,rotate=True)
    #dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

    #for i, batch in enumerate(dataloader):
        #print(i, batch['X'].size(), batch['Y'].size())
        # observe 4th batch
