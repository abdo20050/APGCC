##########################################################################
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw
##
## This script is for running inference on single or multiple images
## and visualizing the results.
##########################################################################
import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from glob import glob
from typing import List

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

def preprocess_image(image_path, cfg):
    """
    Loads and preprocesses a single image for model inference.
    - Converts to RGB
    - Scales if the image is too large (mimicking validation logic)
    - Converts to a Tensor and normalizes
    """
    img = Image.open(image_path).convert('RGB')
    
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

def visualize_results(image_to_display, points, count, image_path, output_dir=None):
    """
    Visualizes the results using matplotlib.
    - Displays the image with predicted points overlaid.
    - Shows the predicted count in the title.
    - Saves the figure if an output directory is provided.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image_to_display)
    plt.title(f'Predicted Count: {count}')
    
    if len(points) > 0:
        # Plot points as red dots
        plt.scatter(points[:, 0], points[:, 1], c='red', s=15, marker='o', alpha=0.8, edgecolors='none')
        
    plt.axis('off')
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{filename}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Result saved to {output_path}")
        plt.close() # Close the figure to free memory
    else:
        plt.show()

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
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        display_img, img_tensor = preprocess_image(image_path, cfg_)
        points, count = run_inference(model, img_tensor, args.threshold)
        visualize_results(display_img, points, count, image_path, args.output_dir)

if __name__ == '__main__':
    main()