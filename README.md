# Joint Finger Valley Points-Free ROI Detection and Recurrent Layer Aggregation for Palmprint Recognition in Open Environment

This repository is a PyTorch implementation of FFARD (accepted by IEEE Transactions on Information Forensics and Security).
#### Abstract
Cooperative palmprint recognition, pivotal for civilian and commercial uses, stands as the most essential and broadly demanded branch in biometrics. These applications, often tied to financial transactions, require high accuracy in recognition. Currently, research in palmprint recognition primarily aims to enhance accuracy, with relatively few studies addressing the automatic and flexible palm region of interest (ROI) extraction (PROIE) suitable for complex scenes. Particularly, the intricate conditions of open environment, alongside the constraint of human finger skeletal extension limiting the visibility of Finger Valley Points (FVPs), render conventional FVPs-based PROIE methods ineffective. In response to this challenge, we propose an FVPs-Free Adaptive ROI Detection (FFARD) approach, which utilizes cross-dataset hand shape semantic transfer (CHSST) combined with the constrained palm inscribed circle search, delivering exceptional hand segmentation and precise PROIE. Furthermore, a Recurrent Layer Aggregation-based Neural Network (RLANN) is proposed to learn discriminative feature representation for high recognition accuracy in both open-set and closed-set modes. The Angular Center Proximity Loss (ACPLoss) is designed to enhance intra-class compactness and inter-class discrepancy between learned palmprint features. Overall, the combined FFARD and RLANN methods are proposed to address the challenges of palmprint recognition in open environment, collectively referred to as RDRLA. Experimental results on four palmprint benchmarks HIT-NIST-V1, IITD, MPD and BJTU\_PalmV2 show the superiority of the proposed method RDRLA over the state-of-the-art (SOTA) competitors.
#### Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{10795182,
  author={Chai, Tingting and Wang, Xin and Li, Ru and Jia, Wei and Wu, Xiangqian},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Joint Finger Valley Points-Free ROI Detection and Recurrent Layer Aggregation for Palmprint Recognition in Open Environment}, 
  year={2025},
  volume={20},
  number={},
  pages={421-435},
  keywords={Palmprint recognition;Feature extraction;Accuracy;Image segmentation;Training;Annotations;Representation learning;Manuals;Location awareness;Lighting;Palmprint recognition;palm ROI detection;recurrent layer aggregation;angular center proximity loss},
  doi={10.1109/TIFS.2024.3516539}}
```
#### DataSet
[ROI of HIT-NIST-V1 Palmprint DataSet](https://drive.google.com/drive/folders/11VJBl-gXTXItQZ8aoE6S2ybxdks-Qeh1?usp=sharing)

[ROI of BJTU\_PalmV2 Palmprint DataSet](https://drive.google.com/drive/folders/1HJFFg_sILdIzpcdWS0W3it2COJN0XhOp?usp=sharing)
1. **Non-commercial Use Only**: This dataset is intended for non-commercial purposes only, such as academic research. It is strictly prohibited to use the dataset for any commercial purposes without permission, including but not limited to the development of commercial products and the provision of commercial services.
2. **Citation Requirement**: If you use this dataset in your research, please make sure to cite this paper in your research outputs (such as academic papers, technical reports, etc.).
#### Requirements
Our codes were implemented by ```PyTorch 2.3.0``` and ```12.1``` CUDA version. If you wanna try our method, please first install necessary packages as follows:
```
pip install -r requirements.txt
```
#### Pretrained Model
The pretrained weights of RLANN for experimental comparison can be downloaded [here](https://drive.google.com/drive/folders/150fjxnRljHuS_kUfwS1QLgzCe4Xuy-dH?usp=sharing). You can use it for training on your own dataset following hyperparameter settings in palmprint_recognition/eval_script.py

#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is sa24221049@mail.ustc.edu.cn.
