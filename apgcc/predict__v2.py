# ========================
# APGCC Inference - Stabilized Ghosting
# ========================
import cv2
import numpy as np
import scipy.ndimage
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from glob import glob
from typing import List
import sys

# --- Kalman Import ---
from kalman_tracker import SortPointTracker

# --- Project-specific imports ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cfg, merge_from_file
from models import build_model

def max_by_axis_pad(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    block = 128
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = max_by_axis_pad([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else: raise ValueError('not supported')
    return tensor

def get_args():
    parser = argparse.ArgumentParser('APGCC Inference Script')
    parser.add_argument('-c', '--config_file', required=True, type=str)
    parser.add_argument('-w', '--weights', required=True, type=str)
    parser.add_argument('-i', '--input', required=True, type=str)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()

def preprocess_image(image_input, cfg):
    if isinstance(image_input, str): img = Image.open(image_input).convert('RGB')
    else: img = image_input.convert('RGB')
    temp_tensor = standard_transforms.ToTensor()(img)
    max_size = max(temp_tensor.shape[1:])
    scale = 1.0
    upper_bound = cfg.DATALOADER.UPPER_BOUNDER
    if upper_bound != -1 and max_size > upper_bound: scale = upper_bound / max_size
    elif max_size > 2560: scale = 2560 / max_size
    transform_list = []
    new_w, new_h = img.width, img.height
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        transform_list.append(standard_transforms.Resize((new_h, new_w)))
    transform_list.extend([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = standard_transforms.Compose(transform_list)
    display_img = img
    if scale != 1.0: display_img = display_img.resize((new_w, new_h))
    img_tensor = transform(img)
    return display_img, img_tensor

@torch.no_grad()
def run_inference(model, image_tensor, threshold):
    model.eval()
    device = next(model.parameters()).device
    samples = nested_tensor_from_tensor_list([image_tensor.to(device)])
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
    return points, len(points)

def main():
    args = get_args()
    if args.config_file != "": cfg_ = merge_from_file(cfg, args.config_file)
    else: cfg_ = cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_model(cfg=cfg_, training=False)
    model.to(device)
    if not os.path.exists(args.weights): raise FileNotFoundError(f"Weight file not found: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    print("Model weights loaded.")

    if os.path.isdir(args.input): image_files = sorted(glob(os.path.join(args.input, '*.jpg')))
    elif os.path.isfile(args.input): image_files = [args.input]
    else: raise FileNotFoundError(f"Input not found: {args.input}")

    if len(image_files) == 1 and any(ext in image_files[0] for ext in ['.mp4', '.avi', '.mov']):
        video_path = image_files[0]
        print(f"Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if args.output_dir:
            if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
            output_video_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else: out = None

        frame_count = 0
        
        # --- TRACKER CONFIGURATION ---
        # max_age=40: Predict for up to 40 frames
        # min_hits=1: Show immediately
        tracker = SortPointTracker(max_age=40, min_hits=3, distance_threshold=80) 
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count +=1
            print(f"Processing frame {frame_count}", end='\r')
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display_img, img_tensor = preprocess_image(pil_img, cfg_)
            
            raw_points, count = run_inference(model, img_tensor, args.threshold)
            tracked_points = tracker.update(raw_points)
            
            if out:
                for point in tracked_points:
                    x, y, tid, is_pred = int(point[0]), int(point[1]), int(point[2]), int(point[3])
                    
                    if not (0 <= x < frame_width and 0 <= y < frame_height): continue

                    if is_pred == 0: 
                        # Real: Red Circle, Green Text
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) 
                        cv2.putText(frame, f"ID:{tid}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else: 
                        # Ghost: Blue Circle, Yellow Text
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1) 
                        cv2.putText(frame, f"ID:{tid}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.putText(frame, f"Count: {len(tracked_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)

        cap.release()
        if out: out.release()
        print(f"\nVideo saved to {output_video_path}")

if __name__ == '__main__':
    main()