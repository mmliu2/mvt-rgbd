import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import cv2
import subprocess
from tqdm import tqdm
import math
from glob import glob
import numpy as np

def box_to_mask(box, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    x, y, w, h = box
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), W)
    y2 = min(int(y + h), H)
    mask[y1:y2, x1:x2] = 1

    return mask

def precision_recall_f1(pred_mask, gt_mask):
    """Compute pixel-wise precision, recall, and F1."""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()

    if tp + fp == 0:
        precision = 1.0  # No predicted pixels
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 1.0  # No ground truth pixels
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def compute_avg_metrics(pr_list, re_list, f1_list):
    pr_array = [np.array(seq) for seq in pr_list]
    re_array = [np.array(seq) for seq in re_list]
    f1_array = [np.array(seq) for seq in f1_list]

    seq_avg = {
        'precision': np.mean([seq.mean() for seq in pr_array]),
        'recall':    np.mean([seq.mean() for seq in re_array]),
        'f1':        np.mean([seq.mean() for seq in f1_array])
    }

    all_pr = np.concatenate(pr_array)
    all_re = np.concatenate(re_array)
    all_f1 = np.concatenate(f1_array)

    frame_avg = {
        'precision': all_pr.mean(),
        'recall':    all_re.mean(),
        'f1':        all_f1.mean()
    }

    return {'seq_avg': seq_avg, 'frame_avg': frame_avg}

def sequence_results_to_metrics(ffmpeg_path, sequences_dir, pred_dir):

    pr_list = []
    re_list = []
    f1_list = []

    for sequence_name in sorted(os.listdir(sequences_dir)):
        gt_dir = os.path.join(sequences_dir, sequence_name)
        im_dir = os.path.join(gt_dir, 'color')
        save_dir = os.path.join(pred_dir, 'videos')
        os.makedirs(save_dir, exist_ok=True)

        gt_file = os.path.join(gt_dir, 'groundtruth.txt')
        pred_file = os.path.join(pred_dir, f'{sequence_name}.txt')

        if not os.path.isfile(pred_file): 
            print('skipping', sequence_name)
            continue

        # remove old frame files
        frame_files = glob(os.path.join(save_dir, "frame*.png"))
        for f in frame_files:
            os.remove(f)
        
        # get image files
        im_files = sorted([os.path.join(im_dir, f) for f in os.listdir(im_dir)
                        if f.lower().endswith(('.jpg', '.png'))])
        
        num_frames = len(im_files)

        # load boxes
        pred_boxes = []
        with open(pred_file, 'r') as f:
            for line in f:
                x, y, w, h = map(float, line.strip().split())
                pred_boxes.append([x, y, w, h])

        gt_boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                x, y, w, h = map(float, line.strip().split(','))
                gt_boxes.append([x, y, w, h])

        seq_pr_list = []
        seq_re_list = []
        seq_f1_list = []

        for i in tqdm(range(0, num_frames), total=num_frames): 
            im = cv2.imread(im_files[i])
            H, W, _ = im.shape

            if math.isnan(pred_boxes[i][0]): 
                seq_pr_list.append(0.0)
                seq_re_list.append(0.0)
                seq_f1_list.append(0.0)
                continue

            if math.isnan(gt_boxes[i][0]): 
                if math.isnan(pred_boxes[i][0]): 
                    seq_pr_list.append(1.0)
                    seq_re_list.append(1.0)
                    seq_f1_list.append(1.0)
                else:
                    seq_pr_list.append(0.0)
                    seq_re_list.append(0.0)
                    seq_f1_list.append(0.0)
                continue

            pred_mask = box_to_mask(pred_boxes[i], H, W)
            gt_mask = box_to_mask(gt_boxes[i], H, W)

            precision, recall, f1 = precision_recall_f1(pred_mask, gt_mask)

            seq_pr_list.append(precision)
            seq_re_list.append(recall)
            seq_f1_list.append(f1)

        pr_list.append(seq_pr_list)
        re_list.append(seq_re_list)
        f1_list.append(seq_f1_list)

    metrics = compute_avg_metrics(pr_list, re_list, f1_list)
    print(metrics)
    # print('Average precision across sequences:', avg_seq_pr)
    # print('Average precision across frames:', avg_frame_pr)


    
def main():
    # render video with bounding boxes on depthtrack color images
    parser = argparse.ArgumentParser(description='Convert single example sequence to video.')
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/local/bin/ffmpeg/ffmpeg-git-20240629-amd64-static/ffmpeg', help='Path to FFmpeg.')
    parser.add_argument('--sequences_dir', type=str, default='../data/depthtrack/test/', help='Directory containing sequences.')
    parser.add_argument('--pred_dir', type=str, default='../output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing', help='Directory containing predictions text file.')

    args = parser.parse_args()

    sequence_results_to_metrics(args.ffmpeg_path, args.sequences_dir, args.pred_dir)


if __name__ == '__main__':
    main()

