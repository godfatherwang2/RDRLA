# Joint Finger Valley Points-Free ROI Detection and Recurrent Layer Aggregation for Palmprint Recognition in Open Environment
This repository is a PyTorch implementation of FFARD (accepted by IEEE Transactions on Information Forensics and Security).
#### Abstract
Cooperative palmprint recognition, pivotal for civilian and commercial uses, stands as the most essential and broadly demanded branch in biometrics. These applications, often tied to financial transactions, require high accuracy in recognition. Currently, research in palmprint recognition primarily aims to enhance accuracy, with relatively few studies addressing the automatic and flexible palm region of interest (ROI) extraction (PROIE) suitable for complex scenes. Particularly, the intricate conditions of open environment, alongside the constraint of human finger skeletal extension limiting the visibility of Finger Valley Points (FVPs), render conventional FVPs-based PROIE methods ineffective. In response to this challenge, we propose an FVPs-Free Adaptive ROI Detection (FFARD) approach, which utilizes cross-dataset hand shape semantic transfer (CHSST) combined with the constrained palm inscribed circle search, delivering exceptional hand segmentation and precise PROIE. Furthermore, a Recurrent Layer Aggregation-based Neural Network (RLANN) is proposed to learn discriminative feature representation for high recognition accuracy in both open-set and closed-set modes. The Angular Center Proximity Loss (ACPLoss) is designed to enhance intra-class compactness and inter-class discrepancy between learned palmprint features. Overall, the combined FFARD and RLANN methods are proposed to address the challenges of palmprint recognition in open environment, collectively referred to as RDRLA. Experimental results on four palmprint benchmarks HIT-NIST-V1, IITD, MPD and BJTU\_PalmV2 show the superiority of the proposed method RDRLA over the state-of-the-art (SOTA) competitors.
#### Citation
If our work is valuable to you, please cite our work:
```
```
#### Requirements
Our codes were implemented by ```PyTorch 2.3.0``` and ```12.1``` CUDA version. If you wanna try our method, please first install necessary packages as follows:
```
pip install requirements.txt
```
#### Pretrained Model
Readers can send an email to the authors of the paper to obtain the dataset mentioned in the text as well as the pretrained weights of RLANN for experimental comparison.
#### Training
After preparing well, you can directly run our training code as follows:
```
python palmprint_recognition/eval_script.py
```
#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

#### Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is sa24221049@mail.ustc.edu.cn.
