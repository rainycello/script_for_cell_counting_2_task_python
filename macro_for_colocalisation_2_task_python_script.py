import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        sys.exit(1)

required_packages = ['opencv-python', 'numpy', 'pandas', 'scikit-image', 'matplotlib']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found, installing...")
        install_package(package)

import cv2, numpy as np, pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib import patches

def select_rois(image):
    rois = []
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        rois.append((x1, y1, x2, y2))
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none'))
        plt.draw()
    plt.gcf().canvas.mpl_connect('button_release_event', onselect)
    plt.show()
    return rois

def split_channels(image): return cv2.split(image)
def convert_to_8bit(image): return cv2.convertScaleAbs(image)
def compute_histogram(image): return np.unique(image, return_counts=True)
def background_subtraction(image, radius): return cv2.subtract(image, cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))))
def despeckle(image, radius): return cv2.medianBlur(image, radius)
def otsu_thresholding(image): _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); return thresholded
def measure_roi(image, rois): return [np.sum(image[roi[1]:roi[3], roi[0]:roi[2]] > 0) for roi in rois]

def main(image_path, apply_otsu=False, apply_bg_subtraction=True, despeckle_radius=20, rolling_radius=20):
    image = imread(image_path)
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
    print(f"Total oversaturated pixels: Before BG: {oversaturated_pixels_before}, After BG: {oversaturated_pixels_after_bg_subtraction}, After Despeckle: {oversaturated_pixels_after_despeckle}, After Otsu: {oversaturated_pixels_after_otsu}")
    print("ROI Measurements:", roi_measurements)
    pd.DataFrame({"ROI Index": range(1, len(roi_measurements) + 1), "Area": roi_measurements}).to_csv('roi_measurements.csv', index=False)
    print("ROI measurements saved to 'roi_measurements.csv'.")

if __name__ == "__main__":
    image_path = input("Enter the path to your image file: ")
    main(image_path)
