import subprocess
import sys
import argparse
import os
from matplotlib.widgets import RectangleSelector
import matplotlib
matplotlib.use('TkAgg')

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        sys.exit(1)

required_packages = ['opencv-python', 'numpy', 'pandas', 'scikit-image', 'matplotlib', 'nd2']

try:
    import nd2
    print('nd2 package is installed and ready to use.')
except ImportError:
    print('nd2 package is not installed.')

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found, installing...")
        install_package(package)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nd2 import ND2File

def read_nd2_image(image_path):
    with ND2File(image_path) as ndf:
        data = ndf.asarray()
        print("ND2 data shape:", data.shape)
        images = data[0]
        return images


def select_rois(image_stack):
    image = image_stack[0]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.title("Draw ROI(s), close window when done")
    rois = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        rois.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

    toggle_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        minspanx=5,
        minspany=5,
        spancoords='pixels',
        interactive=True
    )
    plt.show()
    return rois

# Image processing functions
def split_channels(image): return cv2.split(image)
def convert_to_8bit(image): return cv2.convertScaleAbs(image)
def compute_histogram(image): return np.unique(image, return_counts=True)
def background_subtraction(image, radius): return cv2.subtract(image, cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                                                                                       cv2.getStructuringElement(
                                                                                           cv2.MORPH_ELLIPSE,
                                                                                           (radius, radius))))

def despeckle(image, radius):
    if radius % 2 == 0:
        radius += 1  # make it odd
    radius = max(3, radius)  # at least 3
    return cv2.medianBlur(image, radius)

def otsu_thresholding(image): _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); return thresholded
def measure_roi(image, rois): return [np.sum(image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] > 0) for roi in rois]

def main(image_path, output_path, apply_otsu=False, apply_bg_subtraction=True, despeckle_radius=20, rolling_radius=20):

    image = read_nd2_image(image_path)

    rois = select_rois(image)
    channels = split_channels(image)

    oversaturated_pixels_before = oversaturated_pixels_after_bg_subtraction = oversaturated_pixels_after_despeckle = oversaturated_pixels_after_otsu = 0
    roi_measurements = []

    for i, channel in enumerate(channels):
        channel_8bit = convert_to_8bit(channel)
        values, counts = compute_histogram(channel_8bit)
        oversaturated_pixels_before += counts[-1]

        if apply_otsu:
            otsu_image = otsu_thresholding(channel_8bit)
            values_after_otsu, counts_after_otsu = compute_histogram(otsu_image)
            oversaturated_pixels_after_otsu += counts_after_otsu[-1]
            channel_8bit = otsu_image

        if apply_bg_subtraction:
            bg_subtracted = background_subtraction(channel_8bit, rolling_radius)
            values_after_bg, counts_after_bg = compute_histogram(bg_subtracted)
            oversaturated_pixels_after_bg_subtraction += counts_after_bg[-1]
            channel_8bit = bg_subtracted

        despeckled = despeckle(channel_8bit, despeckle_radius)
        values_after_despeckle, counts_after_despeckle = compute_histogram(despeckled)
        oversaturated_pixels_after_despeckle += counts_after_despeckle[-1]

        roi_area = measure_roi(despeckled, rois)
        roi_measurements.extend(roi_area)

    print(
        f"Total oversaturated pixels: Before BG: {oversaturated_pixels_before}, After BG: {oversaturated_pixels_after_bg_subtraction}, After Despeckle: {oversaturated_pixels_after_despeckle}, After Otsu: {oversaturated_pixels_after_otsu}")
    print("ROI Measurements:", roi_measurements)

    pd.DataFrame({"ROI Index": range(1, len(roi_measurements) + 1), "Area": roi_measurements}).to_csv(
        output_path, index=False)
    print(f"ROI measurements saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image and select regions of interest (ROIs).")
    parser.add_argument('--input', required=True, help="Path to the input image file")
    parser.add_argument('--output', required=True, help="Path to the output CSV file")
    parser.add_argument('--apply_otsu', action='store_true', help="Apply Otsu thresholding")
    parser.add_argument('--apply_bg_subtraction', action='store_true', help="Apply background subtraction")
    parser.add_argument('--despeckle_radius', type=int, default=20, help="Radius for despeckling")
    parser.add_argument('--rolling_radius', type=int, default=20, help="Radius for rolling background subtraction")

    args = parser.parse_args()

    image_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)

    main(image_path, output_path, apply_otsu=args.apply_otsu, apply_bg_subtraction=args.apply_bg_subtraction,
         despeckle_radius=args.despeckle_radius, rolling_radius=args.rolling_radius)