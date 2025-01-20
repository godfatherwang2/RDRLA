import numpy as np
import torch
import os
import cv2
import tqdm
from CHSST.models.toptransformer.basemodel import Topformernet

def process_one_img(rawpth,destpth,model,kpts,device="cuda"):
    img = cv2.imread(rawpth)
    sx = rawpth.split("_")[-1][0]
    h, w, _ = img.shape
    if sx == "L":
        if w>h:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img,1)
    else:
        if w>h:
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    imgIPT = img.copy() #cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    h, w, _ = img.shape
    inputimg = cv2.resize(imgIPT,(448,448))
    indata = np.transpose(inputimg, (2, 0, 1)) / 255.
    indata = torch.tensor(indata).float().unsqueeze(0)
    indata = indata.to(device)
    output = model(indata)
    res = output[0].data.cpu().numpy()
    res = np.expand_dims(res.argmax(axis=0), 2) * 255
    res = np.repeat(res,3,2)
    h, w, _ = img.shape
    o = np.zeros((h, w * 2, 3))
    o[:, 0:w, :] = img
    o[:, w:2 * w, :] = cv2.resize(res, (w, h), interpolation=cv2.INTER_NEAREST)
    return o

def segfile():
    sourcefile_list = [r"/data1/wx/palm/detection/data/MPD_RAW",]
    
    model = Topformernet().cuda()
    model.load_state_dict(torch.load(r"CHSST_checkpoints\EP7-iou0.951562-pacc0.977916.pth"))
    
    for sourcefile in sourcefile_list:
        out_file = r"/data1/wx/palm/detection/data/MPD/"
        all_imgs = os.listdir(sourcefile)
        needed_list = [i for i in all_imgs]#if int(i.split("_")[0])<=75]
        count = 0
        for img in tqdm.tqdm(needed_list):
            raw_pth = os.path.join(sourcefile, img)
            out_pth = os.path.join(out_file, img)
            o = process_one_img(rawpth=raw_pth, destpth=out_pth, model=model, kpts=None)
            h, w, _ = o.shape
            w = int(w / 2)
            p_img = o[:, :w].astype(np.uint8)
            raw_label = o[:, w:].astype(np.uint8)
            label = o[:, w:][:, :, 0].astype(np.uint8)
            contours, _ = cv2.findContours(label.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            m = 0;
            m_area = 0
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if area > m_area:
                    m = i;
                    m_area = area
            label = np.zeros((p_img.shape[0], w)).astype('uint8')
            label = cv2.fillPoly(label, [contours[m]], 255)
            label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
            xs, ys = np.int0(np.min(contours[m].reshape(-1, 2), 0))
            xe, ye = np.int0(np.max(contours[m].reshape(-1, 2), 0))
            palm = cv2.bitwise_and(p_img, label)
            palm = palm[ys:ye, xs:xe]
            cv2.imwrite(out_pth, palm)

if __name__ == '__main__':
    segfile()
