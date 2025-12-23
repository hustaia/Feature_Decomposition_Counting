from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
from scipy.ndimage.filters import gaussian_filter


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def find_dis(point):
    if len(point) > 3:
        square = np.sum(point*point, axis=1)
        dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
        dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    else:
        dis = 128 * np.ones((len(point), 1))
    return dis

def calculate_density(rgb, points):
    density = np.zeros((rgb.shape[0],rgb.shape[1]))
    for i in range(0, len(points)):
        if int(points[i][1])<rgb.shape[0] and int(points[i][0])<rgb.shape[1]:
            density[int(points[i][1]),int(points[i][0])]=1
    density = gaussian_filter(density, 8)
    return density
    
def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    gt_path = im_path.replace('jpg', 'npy')
    points = np.load(gt_path)
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    density = calculate_density(im, points)
    return Image.fromarray(im), points, density



def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default="/data1/chengjian/DarkCrowd_final/",
                        help='original data directory')
    parser.add_argument('--data-dir', default='./data/DarkCrowd/',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048
    
    for phase in ['train', 'val', 'test_bright', 'test_dark']:
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        if phase == 'train':
            with open(os.path.join(args.origin_dir, "train_list.txt"), "r") as file:
                im_list = [os.path.join(args.origin_dir, "bright_images", line.strip()) for line in file.readlines()]
        if phase == 'val':
            with open(os.path.join(args.origin_dir, "val_list.txt"), "r") as file:
                im_list = [os.path.join(args.origin_dir, "degraded_dark_images", line.strip()) for line in file.readlines()]
        if phase == 'test_bright':
            with open(os.path.join(args.origin_dir, "test_bright_list.txt"), "r") as file:
                im_list = [os.path.join(args.origin_dir, "bright_images", line.strip()) for line in file.readlines()]
        if phase == 'test_dark':
            with open(os.path.join(args.origin_dir, "test_dark_list.txt"), "r") as file:
                im_list = [os.path.join(args.origin_dir, "degraded_dark_images", line.strip()) for line in file.readlines()]
        for im_path in im_list:
            name = os.path.basename(im_path)
            print(name)
            im, points, density = generate_data(im_path)
            if phase == 'train':
                dis = find_dis(points)
                if len(points)>0:
                    points = np.concatenate((points, dis), axis=1)
            im_save_path = os.path.join(sub_save_dir, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace('jpg', 'npy')
            np.save(gd_save_path, points)
            den_save_path = im_save_path.replace('.jpg', '_density.npy')
            np.save(den_save_path, density)
