##########################################################################
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw
##
## This script is for running inference on single or multiple images
## and visualizing the results.
##########################################################################
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as standard_transforms
from glob import glob
from typing import List
from kalman_tracker import Track, PointTracker

# --- Project-specific imports ---
# Add the project root to the Python path to allow for absolute imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import cfg, merge_from_file
from models import build_model

# This utility function is from the training pipeline (datasets/build.py)
# It's included here to make the prediction script self-contained.
def max_by_axis_pad(the_list: List[List[int]]) -> List[int]:
    """Helper function to find max dimensions for padding."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    # Pad to a multiple of 128, mimicking training behavior
    block = 128
    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Pads a list of tensors to the same size to create a single batch tensor.
    """
    if tensor_list[0].ndim == 3:
        max_size = max_by_axis_pad([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser('APGCC Inference Script')
    parser.add_argument('-c', '--config_file', required=True, type=str,
                        help='Path to the model config file (e.g., ./configs/SHHA_test.yml)')
    parser.add_argument('-w', '--weights', required=True, type=str,
                        help='Path to the trained model weights (e.g., ./output/SHHA_best.pth)')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to a single image or a directory of images')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Directory to save visualized results. If not set, results are shown on screen.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for point detection')
    return parser.parse_args()

def preprocess_image(image_input, cfg):
    """
    Loads and preprocesses a single image for model inference.
    - image_input: can be a file path (str) or a PIL Image object.
    - Converts to RGB
    - Scales if the image is too large (mimicking validation logic)
    - Converts to a Tensor and normalizes
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')
    
    # --- Mimic validation scaling from dataset.py ---
    # Use a temporary tensor to get dimensions for scaling
    temp_tensor = standard_transforms.ToTensor()(img)
    max_size = max(temp_tensor.shape[1:])
    scale = 1.0
    upper_bound = cfg.DATALOADER.UPPER_BOUNDER

    if upper_bound != -1 and max_size > upper_bound:
        scale = upper_bound / max_size
    elif max_size > 2560:  # A reasonable default from the original codebase
        scale = 2560 / max_size

    # --- Define transforms ---
    transform_list = []
    if scale != 1.0:
        # Use PIL resize for simplicity on a single image
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        transform_list.append(standard_transforms.Resize((new_h, new_w)))

    transform_list.extend([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = standard_transforms.Compose(transform_list)
    
    # The image for display should be the one that's resized
    display_img = img
    if scale != 1.0:
        display_img = display_img.resize((new_w, new_h))

    img_tensor = transform(img)
    return display_img, img_tensor

@torch.no_grad()
def run_inference(model, image_tensor, threshold):
    """
    Runs the model on a preprocessed image tensor and returns the predicted points.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # The model expects a batch, so we wrap the tensor in a list and use the collate function
    samples = nested_tensor_from_tensor_list([image_tensor.to(device)])
    
    outputs = model(samples)
    
    # Extract scores and points
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    
    # Filter points based on the confidence threshold
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
    predict_cnt = len(points)
    
    return points, predict_cnt

def visualize_results(image_to_display, points, count, image_path, output_dir=None, frame=None):
    """
    Visualizes the results using matplotlib.
    - Displays the image with predicted points overlaid.
    - Shows the predicted count in the title.
    - Saves the figure if an output directory is provided.
    """
    plt.figure(figsize=(12, 8))
    if frame is not None:
        # Convert frame to RGB for display with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title(f'Predicted Count: {count}')

        if len(points) > 0:
            # Plot points as red dots
            plt.scatter(points[:, 0], points[:, 1], c='red', s=15, marker='o', alpha=0.8, edgecolors='none')

        plt.axis('off')

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"pred_{filename}.jpg")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Result frame saved to {output_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()
    else: # Existing image handling
        plt.imshow(image_to_display)
        plt.title(f'Predicted Count: {count}')
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1], c='red', s=15, marker='o', alpha=0.8, edgecolors='none')
        plt.axis('off')
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"pred_{filename}")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Result saved to {output_path}")
            plt.close()

def merge_close_points(points, threshold):
    points = np.array(points)
    if len(points) == 0:
        return []

    merged = []
    visited = set()

    for i in range(len(points)):
        if i in visited:
            continue
        cluster = [points[i]]
        visited.add(i)

        for j in range(i + 1, len(points)):
            if j in visited:
                continue
            dist = np.linalg.norm(points[i] - points[j])
            if dist < threshold:
                cluster.append(points[j])
                visited.add(j)

        # cluster_avg = np.mean(cluster, axis=0)
        # merged.append(cluster_avg)
        # print(cluster)
        merged.append(cluster[0])  # Use the latest point in the cluster as the representative

    return merged

def main():
    args = get_args()
    
    # --- Load Configuration ---
    if args.config_file != "":
        cfg_ = merge_from_file(cfg, args.config_file)
    else:
        cfg_ = cfg

    # --- Setup Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_model(cfg=cfg_, training=False)
    model.to(device)
    
    # --- Load Weights ---
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight file not found: {args.weights}")
    
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")

    # --- Find Image Files ---
    if os.path.isdir(args.input):
        image_files = []
        supported_extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in supported_extensions:
            image_files.extend(glob(os.path.join(args.input, ext)))
        image_files = sorted(image_files)
    elif os.path.isfile(args.input):
        image_files = [args.input]
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Found {len(image_files)} image(s) to process.")

    # --- Process Each Image ---
    if len(image_files) == 1 and os.path.isfile(image_files[0]) and any(ext in image_files[0] for ext in ['.mp4', '.avi', '.mov']):  # Basic check for video file
        video_path = image_files[0]
        print(f"Processing video: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        if args.output_dir:
            output_video_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_output.mp4")
            # Revert to a widely supported codec like 'mp4v' (MPEG-4).
            # The error occurs because OpenCV's VideoWriter may not support encoding with the same
            # codec as the input video (e.g., HEVC/H.265), even if it can read it.
            # 'mp4v' is a safe and common choice for .mp4 files.
            fourcc = cv2.VideoWriter_fourcc(*'X265')
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        else:
            out = None

        frame_count = 0
        # --- Kalman Filter Tracker Setup ---
        tracks = []
        max_missed_frames = 5
        matching_threshold = 100  # pixels
        track_id_counter = 0
        tracker = PointTracker(distance_threshold=10000)
        collected_points = []
        displayed_points = np.empty((0, 2))  # For displaying points in the video
        window = 2  # Number of frames to collect before displaying points
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count +=1
            print(f"Processing frame {frame_count}")
            # Convert frame to PIL Image for preprocessing
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            display_img, img_tensor = preprocess_image(pil_img, cfg_)
            points, count = run_inference(model, img_tensor, args.threshold)
            if len(displayed_points) == 0:
                displayed_points = np.array(points)
            if frame_count % window:
                collected_points.extend(points.tolist())
            else:
                displayed_points = merge_close_points(collected_points, threshold=5)
                collected_points = []
            #Kalman filter integration would go here to update 'points'
            # updated_points = []
            # used_tracks = set()
            # assigned_tracks = {}

            # for pt in points:
            #     best_track = None
            #     best_dist = float('inf')
            #     for track in tracks:
            #         if track.track_id in used_tracks:
            #             continue
            #         pred = track.predict()
            #         dist = np.linalg.norm(pred - pt)
            #         if dist < best_dist and dist < matching_threshold:
            #             best_dist = dist
            #             best_track = track

            #     if best_track:
            #         best_track.update(pt)
            #         used_tracks.add(best_track.track_id)
            #         updated_pt = best_track.kf.x[:2].flatten()
            #         updated_points.append((updated_pt[0], updated_pt[1], best_track.track_id))
            #     else:
            #         new_track = Track(pt, track_id_counter)
            #         tracks.append(new_track)
            #         updated_points.append((pt[0], pt[1], track_id_counter))
            #         track_id_counter += 1

            # # Remove old tracks
            # tracks = [t for t in tracks if t.missed < max_missed_frames]

            # Result: updated_points = list of (x, y, id)
            points = tracker.update(points)
            

            if out:
                #Visualization for video output (using OpenCV directly on the frame)
                for point in displayed_points:
                    # print(point)
                    x, y, tid = int(point[0]), int(point[1]), 1
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red circles
                    # cv2.putText(frame, f"ID:{tid}", (x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                                        
                cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green count text
                out.write(frame)
            else:
                visualize_results(display_img, points, count, "frame", args.output_dir, frame) # Pass the frame for display

        cap.release()
        if out:
            out.release()
            print(f"Video with predictions saved to {output_video_path}")
    else:
        for image_path in image_files: # Existing image handling
            print(f"\nProcessing: {os.path.basename(image_path)}")
            display_img, img_tensor = preprocess_image(image_path, cfg_)
            points, count = run_inference(model, img_tensor, args.threshold)
            visualize_results(display_img, points, count, image_path, args.output_dir)
if __name__ == '__main__':
    main()