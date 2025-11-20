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

def sequence_results_to_video(ffmpeg_path, sequences_dir, pred_dir, sequence_name, start_frame, num_frames):
    gt_dir = os.path.join(sequences_dir, sequence_name)
    im_dir = os.path.join(gt_dir, 'color')
    save_dir = os.path.join(pred_dir, 'videos')
    os.makedirs(save_dir, exist_ok=True)

    gt_file = os.path.join(gt_dir, 'groundtruth.txt')
    pred_file = os.path.join(pred_dir, f'{sequence_name}.txt')

    # remove old frame files
    frame_files = glob(os.path.join(save_dir, "frame*.png"))
    for f in frame_files:
        os.remove(f)
    
    # get image files
    im_files = sorted([os.path.join(im_dir, f) for f in os.listdir(im_dir)
                    if f.lower().endswith(('.jpg', '.png'))])
    
    if num_frames == -1:
        num_frames = len(im_files)-1

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

    assert start_frame >= 0 and start_frame+num_frames < len(im_files)

    for i in tqdm(range(start_frame, start_frame+num_frames), total=num_frames): 
        im = cv2.imread(im_files[i])

        if not math.isnan(pred_boxes[i][0]): 
            px, py, pw, ph = map(int, pred_boxes[i])
            cv2.rectangle(im, (px, py), (px+pw, py+ph), (0, 0, 255), 2)   # red (BGR)

        if not math.isnan(gt_boxes[i][0]): 
            gx, gy, gw, gh = map(int, gt_boxes[i])
            cv2.rectangle(im, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 2)   # green (BGR)

        h, w, _ = im.shape
        im_small = cv2.resize(im, (640, 640*h//w)) # (w, h)

        save_path = os.path.join(save_dir, f"frame_{i:04d}.png")
        cv2.imwrite(save_path, im_small)

    output_video = os.path.join(save_dir, f"{sequence_name}.mp4")

    # FFmpeg command
    cmd = [
        ffmpeg_path,
        "-y",                    # overwrite output file
        "-framerate", "30",      # FPS
        "-i", os.path.join(save_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    # Run it
    subprocess.run(cmd, check=True)

    print("Video saved to:", output_video)

    # remove frame files
    frame_files = glob(os.path.join(save_dir, "frame*.png"))
    for f in frame_files:
        os.remove(f)

    
def main():
    # render video with bounding boxes on depthtrack color images
    parser = argparse.ArgumentParser(description='Convert single example sequence to video.')
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/local/bin/ffmpeg/ffmpeg-git-20240629-amd64-static/ffmpeg', help='Path to FFmpeg.')
    parser.add_argument('--sequences_dir', type=str, default='../data/depthtrack/test/', help='Directory containing sequences.')
    parser.add_argument('--pred_dir', type=str, default='../output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing', help='Directory containing predictions text file.')
    parser.add_argument('--sequence_name', type=str, default='adapter01_indoor', help='Sequence name.')

    parser.add_argument('--start_frame', type=int, default=0, help='Starting index.')
    parser.add_argument('--num_frames', type=int, default=-1, help='Number of frames in video. -1 for all frames')
    
    args = parser.parse_args()

    sequence_results_to_video(args.ffmpeg_path, args.sequences_dir, args.pred_dir, args.sequence_name, args.start_frame, args.num_frames)


if __name__ == '__main__':
    main()

