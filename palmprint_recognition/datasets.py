import os
import numpy as np
from PIL import Image,ImageOps

class Dataset_closeset:
    def __init__(self, root, transform, if_train=True,filter_rotate = False):
        self.filter_rotate = filter_rotate
        self.transform = transform
        self.root = root
        self.train = if_train
        self.first_stage_item = []
        self.second_stage_item = []
        self.labels = {}
        self.label_total_num = {}
        self.label_state_split_num = {}
        self.items = []
        self.counter = 0
        self.file_name_list = os.listdir(root)
        self.first_stage_iname = []
        self.second_stage_iname = []
        self.img_name_set = set()
        self.search_dir()
        self.split_stage()

    def center_and_pad_image(self,input_img_cv2):
        height, width, _ = input_img_cv2.shape
        new_size = int(max(width,height))# + 2 * border_width
        x_offset = (new_size - width) // 2
        y_offset = (new_size - height) // 2
        padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
        return padded_image

    def split_stage(self):
        files = self.img_name_set
        for iname in files:
            label,_ = self.get_label(iname)
            if label not in self.label_state_split_num:
                self.label_state_split_num[label] = 1
            else:
                self.label_state_split_num[label] = self.label_state_split_num[label] + 1
            if self.label_state_split_num[label] <= int((self.label_total_num[label] + 1) / 2):
                self.first_stage_iname.append((str(iname)))
            else:
                self.second_stage_iname.append((str(iname)))

        for item in self.items:
            item_iname,theta = self.get_raw_name(item[1])
            if item_iname in self.first_stage_iname:
                if self.filter_rotate:
                    if theta == 0:
                        self.first_stage_item.append(item)
                else:
                    self.first_stage_item.append(item)
            elif item_iname in self.second_stage_iname:
                if self.filter_rotate:
                    if theta == 0:
                        self.second_stage_item.append(item)
                else:
                    self.second_stage_item.append(item)

    def search_dir(self):
        files = self.file_name_list
        for file in files:
            r_iname = self.get_raw_name(file)[0]
            self.img_name_set.add(r_iname)
            label,_ = self.get_label(file)
            if label not in self.labels:
                self.labels[label] = self.counter
                self.counter += 1
            self.items.append((self.labels[label], str(file)))


        for i in self.img_name_set:
            label,_ = self.get_label(i)
            if label not in self.label_total_num:
                self.label_total_num[label] = 1
            else:
                self.label_total_num[label] = self.label_total_num[label] + 1

    def get_label(self,file):
        _slist = file.split('_')
        if len(_slist) == 4:
            label = str(int(_slist[0])) + _slist[2][0]
            theta = int(_slist[-1].split(".")[0])
        elif len(_slist) == 3:
            label = str(int(_slist[0])) + _slist[2][0]
            theta = 0
        return label,theta

    def get_raw_name(self,file):
        _slist = file.split('_')
        if len(_slist) == 4:
            iname = "_".join(_slist[:3])
            theta = int(_slist[-1].split(".")[0])
        return iname,theta

    def _read_image(self, image_file):
        image_file = str(image_file)
        img = Image.open(image_file).convert('RGB')
        img = ImageOps.exif_transpose(img)
        return img

    def __len__(self):
        if self.train:
            return len(self.first_stage_item)
        else:
            return len(self.second_stage_item)

    def __getitem__(self, index):
        if self.train == True:
            data = self.first_stage_item[index]
        else:
            data = self.second_stage_item[index]
        image_path = os.path.join(self.root, data[1])
        label = np.array(data[0], dtype='int64')
        image = self._read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image,label,data[1]
class Dataset_closeset_session:
    def __init__(self, root, transform, if_train=True,filter_rotate = False):
        self.filter_rotate = filter_rotate
        self.transform = transform
        self.root = root
        self.train = if_train
        self.first_stage_item = []
        self.second_stage_item = []
        self.labels = {}
        self.label_total_num = {}
        self.label_state_split_num = {}
        self.items = []
        self.counter = 0
        self.file_name_list = os.listdir(root)
        self.search_dir()
        self.split_stage()

    def center_and_pad_image(self,input_img_cv2):
        height, width, _ = input_img_cv2.shape
        new_size = int(max(width,height))# + 2 * border_width
        x_offset = (new_size - width) // 2
        y_offset = (new_size - height) // 2
        padded_image = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = input_img_cv2
        return padded_image

    def split_stage(self):
        files = self.file_name_list
        for file in files:
            if len(file.split("_"))==3:
                label = self.get_label(file)
                if file.split("_")[1] == "F":
                    self.first_stage_item.append((self.labels[label], str(file)))
                elif file.split("_")[1] == "S":
                    self.second_stage_item.append((self.labels[label], str(file)))
            elif len(file.split("_"))==6:
                label = self.get_label(file)
                item_iname, theta = self.get_raw_name(file)
                if file.split("_")[1] == "2":
                    if self.filter_rotate:
                        if theta == 0:
                            self.first_stage_item.append((self.labels[label], str(file)))
                    else:
                        self.first_stage_item.append((self.labels[label], str(file)))
                elif file.split("_")[1] == "1" and int(file.split("_")[-1].split(".")[0]) == 0:
                    if self.filter_rotate:
                        if theta == 0:
                            self.second_stage_item.append((self.labels[label], str(file)))
                    else:
                        self.second_stage_item.append((self.labels[label], str(file)))



    def get_raw_name(self,file):
        _slist = file.split('_')
        if len(_slist) == 4:
            iname = "_".join(_slist[:3])
            theta = int(_slist[-1].split(".")[0])
        elif len(_slist) == 6:
            iname = "_".join(_slist[:5])
            theta = int(_slist[-1].split(".")[0])
        return iname,theta

    def search_dir(self):
        files = self.file_name_list
        for file in files:
            label = self.get_label(file)
            if label not in self.labels:
                self.labels[label] = self.counter
                self.counter += 1
            if label not in self.label_total_num:
                self.label_total_num[label] = 1
            else:
                self.label_total_num[label] = self.label_total_num[label] + 1
            self.items.append((self.labels[label],str(file)))

    def get_label(self,file):
        _slist = file.split('_')
        if len(_slist) == 4:
            label = str(int(_slist[0])) + _slist[2][0]
        elif len(_slist) == 3:
            label = str(int(_slist[0])) + _slist[2][0]
        elif len(_slist) == 6:
            label = str(int(_slist[0])) + _slist[3][0]
        elif len(_slist) == 7:
            label = str(int(_slist[0])) + _slist[3][0]
        return label

    def _read_image(self, image_file):
        image_file = str(image_file)
        img = Image.open(image_file).convert('RGB')
        img = ImageOps.exif_transpose(img)
        return img

    def __len__(self):
        if self.train:
            return len(self.first_stage_item)
        else:
            return len(self.second_stage_item)

    def __getitem__(self, index):
        if self.train == True:
            data = self.first_stage_item[index]
        else:
            data = self.second_stage_item[index]

        image_path = os.path.join(self.root, data[1])
        label = np.array(data[0], dtype='int64')
        image = self._read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image,label,data[1]

class Dataset_openset:
    def __init__(self,root,transform,mode = "train",session = False,filter_train = False,ifprint = False):
        self.filter_train = filter_train
        self.transform = transform
        self.root = root
        self.mode = mode
        self.test_items = []
        self.train_item = []
        self.gal_item = []
        self.val_item = []
        self.labels = {}
        self.label_total_num = {}
        self.label_state_split_num = {}
        self.counter = 0
        self.file_name_list = os.listdir(root)
        self.search_dir()
        if session:
            self.split_stage_session()
        else:
            self.split_stage()
        train_labels = list(set([item[0] for item in self.train_item]))
        total_train_class_num = len(train_labels)
        test_labels = list(set([item[0] for item in self.test_items]))
        total_test_class_num = len(test_labels)
        min_testnum = min(test_labels)
        self.gal_item = [(item[0]-min_testnum,item[1]) for item in self.gal_item]
        self.val_item = [(item[0]-min_testnum,item[1]) for item in self.val_item]
        if ifprint == True:
            print("train_class_num:",total_train_class_num,"test_class_num:",total_test_class_num)
            print("train_samples:",len(self.train_item))
            print("val_samples:", len(self.val_item))
            print("gal_samples:", len(self.gal_item))

    def get_label(self,file):
        _slist = file.split('_')
        if len(_slist) == 4:
            label = str(int(_slist[0])) + _slist[2][0]
            theta = int(_slist[-1].split(".")[0])
        elif len(_slist) == 6:
            label = str(int(_slist[0])) + _slist[3][0]
            theta = int(_slist[-1].split(".")[0])
        return label,theta

    def search_dir(self):
        files = self.file_name_list
        for file in files:
            label,theta = self.get_label(file)
            if label not in self.labels:
                self.labels[label] = self.counter
                self.counter += 1
        total_class = len(list(self.labels.keys()))
        #print("total:",total_class)
        if total_class%2 == 0:
            train_classes = total_class//2
        else:
            train_classes = (total_class//2)+1
        #print("train:",train_classes)

        for file in files:
            label,theta = self.get_label(file)
            label_num = self.labels[label]
            if label_num < train_classes:
                if self.filter_train == False:
                    self.train_item.append((self.labels[label], str(file)))
                else:
                    if theta == 0:
                        self.train_item.append((self.labels[label], str(file)))
            else:
                if theta == 0:
                    self.test_items.append((self.labels[label], str(file)))
                    if label not in self.label_total_num:
                        self.label_total_num[label] = 1
                    else:
                        self.label_total_num[label] = self.label_total_num[label] + 1

    def split_stage(self):
        files = self.test_items
        for item in files:
            iname = item[1]
            label, _ = self.get_label(iname)
            if label not in self.label_state_split_num:
                self.label_state_split_num[label] = 1
            else:
                self.label_state_split_num[label] = self.label_state_split_num[label] + 1
            if self.label_state_split_num[label] <= int((self.label_total_num[label] + 1) / 2):
                self.gal_item.append(item)
            else:
                self.val_item.append(item)

    def split_stage_session(self):
        files = self.test_items
        for item in files:
            iname = item[1]
            label,_ = self.get_label(iname)
            if iname.split("_")[1] == "2":
                self.gal_item.append(item)
            elif iname.split("_")[1] == "1":
                self.val_item.append(item)

    def _read_image(self, image_file):
        image_file = str(image_file)
        img = Image.open(image_file).convert('RGB')
        img = ImageOps.exif_transpose(img)
        return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_item)
        elif self.mode == 'gal':
            return len(self.gal_item)
        elif self.mode == 'val':
            return len(self.val_item)
    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.train_item[index]
        elif self.mode == 'gal':
            data = self.gal_item[index]
        elif self.mode == 'val':
            data = self.val_item[index]

        image_path = os.path.join(self.root,data[1])
        label = np.array(data[0], dtype='int64')
        image = self._read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image,label,data[1]