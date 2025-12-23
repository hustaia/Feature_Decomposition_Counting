This repository includes the official implementation of our illumination-robust feature decomposition approach for low-light crowd counting, presented in our paper:

**[An Illumination-robust Feature Decomposition Approach for Low-light Crowd Counting]()**

Pattern Recognition Letters, 2025

Jian Cheng, Chen Feng, Yang Xiao, Zhiguo Cao

Huazhong University of Science and Technology

### Overview
<p align="left">
  <img src="overview.PNG" width="850" title="Overview"/>
</p>

### Dataset
We present DarkCrowd, a dataset for evaluating the model’s crowd counting
performance in low-light environments. The dataset can be downloaded **[here](https://drive.google.com/file/d/1fgE9-6HXXE4G2aWa2MIkO2YeUSHRj-Pd/view?usp=drive_link)**.
<p align="left">
  <img src="dataset.PNG" width="850" title="Overview"/>
</p>

### Usage
Before training and testing, preprocess the dataset with
​```
python preprocess_dataset_DarkCrowd.py
​```

To train the model, run
​```
python train_BL.py
​```

To test the model, download the pretrained models (FD+BL: **[link](https://drive.google.com/file/d/1ZJjj5w1ZtSBtWJybJnfLnGW1mQP2FyoU/view?usp=drive_link)**, FD+MAN: **[link](https://drive.google.com/file/d/1bNau-DvNM-19UaeQBiXAtbYQBu4Lcfu2/view?usp=drive_link)**, FD+GramFormer: **[link](https://drive.google.com/file/d/1BxkBVA-mtPRzsfEiupI06bhddl6gGEH0/view?usp=drive_link)**) to the main folder, then run
​```
python test_BL.py
​```

### Permission
The code are only for non-commercial purposes. Copyrights reserved.

Contact: 
Jian Cheng (jian_cheng@hust.edu.cn) 

