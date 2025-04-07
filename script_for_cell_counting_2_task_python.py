import subprocess
import sys
import importlib
import argparse
import cv2
import pandas as pd
from skimage import measure


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_and_install_package(package):
    try:
        importlib.import_module(package)
    except ImportError:
        install_package(package)


required_packages = ["opencv-python", "pandas", "scikit-image", "pyimagej", "scyjava", "numpy", "matplotlib",
                     "requests", "tifffile"]
for package in required_packages: check_and_install_package(package)


def process_image(image_path, output_csv, use_bg_subtraction=True, manual_thresh=None, brdu_thresh=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return print(f"Cannot open: {image_path}")

    blurred = cv2.GaussianBlur(cv2.subtract(img, cv2.medianBlur(img, 25)), (5, 5),
                               0) if use_bg_subtraction else cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blurred, manual_thresh if manual_thresh else 0, 255,
                            cv2.THRESH_BINARY + (cv2.THRESH_OTSU if manual_thresh is None else 0))
    labels = measure.label(mask)
    results = [{"Label": p.label, "Area": p.area, "Centroid X": p.centroid[1], "Centroid Y": p.centroid[0]} for p in
               measure.regionprops(labels)]
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}\nTotal cells counted: {len(results)}")

    brdu_mask = cv2.threshold(blurred, brdu_thresh, 255, cv2.THRESH_BINARY)[1]
    print(f"BrdU-positive cells counted: {len(measure.regionprops(measure.label(brdu_mask)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell counting script (manual/auto threshold, BrdU detection)")
    parser.add_argument('--input', '-i', required=True, help="Path to input .tif image")
    parser.add_argument('--output', '-o', required=True, help="Path to save CSV output")
    parser.add_argument('--manual-threshold', type=int, help="Manual threshold")
    parser.add_argument('--brdu-threshold', type=int, default=100, help="BrdU-positive detection threshold")
    parser.add_argument('--no-bg-sub', action='store_true', help="Skip background subtraction")
    parser.add_argument('--manual', action='store_true', help="Show manual counting instructions")
    args = parser.parse_args()
    process_image(args.input, args.output, not args.no_bg_sub, args.manual_threshold, args.brdu_threshold)
