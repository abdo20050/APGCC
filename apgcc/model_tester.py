import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import argparse
import torch
import cv2
import numpy as np
import psutil
from PIL import Image
from datetime import datetime
from glob import glob
from predict__v2 import preprocess_image, run_inference  # Adjust import path
from config import cfg, merge_from_file
from models import build_model

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_bytes(bytes_val):
    return bytes_val / (1024 * 1024)

def log(msg, log_file):
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

def process_video(video_path, model, cfg, threshold, device, sample_duration, fps):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = min(total_frames, sample_duration * fps)

    frame_times, mem_usages, counts = [], [], []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _, img_tensor = preprocess_image(pil_img, cfg)

        start_time = time.time()
        mem_before = psutil.Process(os.getpid()).memory_info().rss

        points, count = run_inference(model, img_tensor, threshold)

        mem_after = psutil.Process(os.getpid()).memory_info().rss
        elapsed_time = time.time() - start_time

        frame_times.append(elapsed_time)
        mem_usages.append(mem_after - mem_before)
        counts.append(count)

    cap.release()

    return {
        "video": os.path.basename(video_path),
        "period_sec": frame_count / fps,
        "frames": frame_count,
        "resolution": f"{frame_w}x{frame_h}",
        "avg_time": np.mean(frame_times),
        "min_time": np.min(frame_times),
        "max_time": np.max(frame_times),
        "avg_mem": np.mean(mem_usages),
        "min_mem": np.min(mem_usages),
        "max_mem": np.max(mem_usages),
        "avg_count": np.mean(counts),
        "min_count": np.min(counts),
        "max_count": np.max(counts)
    }

def print_report(results, device, log_file):
    log("\n========= DETAILED PERFORMANCE REPORT =========\n", log_file)
    for res in results:
        log(f"Video: {res['video']}", log_file)
        log(f"  Duration: {res['period_sec']:.2f} seconds ({res['frames']} frames)", log_file)
        log(f"  Resolution: {res['resolution']}", log_file)
        log(f"  Inference Time [s]: Avg={res['avg_time']:.4f}, Min={res['min_time']:.4f}, Max={res['max_time']:.4f}", log_file)
        log(f"  Memory Usage [MB]: Avg={format_bytes(res['avg_mem']):.2f}, Min={format_bytes(res['min_mem']):.2f}, Max={format_bytes(res['max_mem']):.2f}", log_file)
        log(f"  Count Predictions: Avg={res['avg_count']:.2f}, Min={res['min_count']}, Max={res['max_count']}", log_file)
        log("-" * 60, log_file)

    all_times = [r['avg_time'] for r in results]
    all_mems = [r['avg_mem'] for r in results]
    all_counts = [r['avg_count'] for r in results]

    log("\n========= SUMMARY =========", log_file)
    log(f"Device used: {device}", log_file)
    log(f"Average Inference Time: {np.mean(all_times):.4f}s", log_file)
    log(f"Min/Max Inference Time: {np.min(all_times):.4f}s / {np.max(all_times):.4f}s", log_file)
    log(f"Average Memory Usage: {format_bytes(np.mean(all_mems)):.2f} MB", log_file)
    log(f"Min/Max Memory Usage: {format_bytes(np.min(all_mems)):.2f} MB / {format_bytes(np.max(all_mems)):.2f} MB", log_file)
    log(f"Average Count (all videos): {np.mean(all_counts):.2f}", log_file)

def get_args():
    parser = argparse.ArgumentParser("Model Tester Script")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--video_dir', required=True, help='Directory containing video files')
    parser.add_argument('--threshold', type=float, default=0.5, help='Model threshold')
    parser.add_argument('--duration', type=int, default=5, help='Seconds per video to evaluate')
    parser.add_argument('--fps', type=int, default=24, help='Video frame rate')
    return parser.parse_args()

def main():
    args = get_args()
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"tester_report_{timestamp}.log"
    log(f"Using device: {device}", log_file)

    config = merge_from_file(cfg, args.config)
    model = build_model(cfg=config, training=False)
    model.to(device)
    model.eval()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight file not found: {args.weights}")
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)
    log("Model weights loaded successfully.", log_file)

    supported_exts = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in supported_exts:
        video_files.extend(glob(os.path.join(args.video_dir, ext)))
    video_files = sorted(video_files)[:5]

    if len(video_files) < 5:
        raise ValueError(f"Expected at least 5 videos, found {len(video_files)}")

    log(f"Processing {len(video_files)} videos from: {args.video_dir}", log_file)

    results = []
    for video_path in video_files:
        log(f"\nProcessing video: {video_path}", log_file)
        metrics = process_video(
            video_path, model, config,
            args.threshold, device,
            args.duration, args.fps
        )
        results.append(metrics)

    print_report(results, device, log_file)
    log(f"\nReport saved to {log_file}", log_file)

if __name__ == "__main__":
    main()
