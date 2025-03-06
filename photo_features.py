import pandas as pd
from PIL import Image, ExifTags
from PIL.TiffImagePlugin import IFDRational
import cv2
import numpy as np
from skimage import exposure
import mediapipe as mp

# from skimage.feature import greycoprops
import pytesseract
import ssim
import logging
from functools import wraps
import time
from copy import deepcopy

logger = logging.getLogger(__name__)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result

    return wrapper


@time_it
def extract_exif(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        # Extract EXIF data
        exif_data = image._getexif()

        if not exif_data:
            logger.info("No EXIF data found.")
            return

        # Convert EXIF data into a readable format
        exif_dict = {}
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            exif_dict[tag_name] = value

        return exif_dict
    except Exception as e:
        logger.warning(f"Error extracting EXIF: {e}")
        return None


@time_it
def is_black_and_white(image_path):
    """Checks if the image is black and white."""
    image = cv2.imread(image_path)
    if len(image.shape) < 3 or image.shape[2] == 1:
        return True  # Single-channel image (grayscale)
    b, g, r = cv2.split(image)
    return np.array_equal(b, g) and np.array_equal(g, r)


@time_it
def calculate_sharpness(image_path):
    """Calculates image sharpness using the variance of the Laplacian."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()


@time_it
def calculate_brightness(image_path):
    """Estimates the brightness of the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return np.mean(image)


@time_it
def check_overexposure(image_path, threshold=245):
    """Checks if the image has overexposed areas."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    overexposed_pixels = np.sum(image > threshold)
    total_pixels = image.size
    return overexposed_pixels / total_pixels * 100  # % of overexposed pixels


@time_it
def check_darkness(image_path, threshold=10):
    """Checks if the image has underexposed areas."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dark_pixels = np.sum(image < threshold)
    total_pixels = image.size
    return dark_pixels / total_pixels * 100  # % of underexposed pixels


@time_it
def calculate_contrast(image_path):
    """Calculates the contrast of the image (difference between min and max brightness)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.max() - image.min()


@time_it
def calculate_colorfulness(image_path):
    """Estimates the colorfulness of the image."""
    image = cv2.imread(image_path)
    rg = np.absolute(image[:, :, 0] - image[:, :, 1])  # Difference between R and G
    yb = np.absolute(
        0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2]
    )  # Average difference between R, G, and B
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    return np.sqrt(rg_mean ** 2 + yb_mean ** 2)


# 1. Color Histogram
@time_it
def color_histogram(image_path):
    """Creates histograms for BGR channels."""
    image = cv2.imread(image_path)
    histogram = {}
    for i, color in enumerate(("Blue", "Green", "Red")):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histogram[color] = hist
    return histogram


# 2. Texture Features (GLCM)
# def texture_features(image_path):
#     """Extracts texture features using GLCM."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     glcm = greycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
#     features = {
#         "contrast": greycoprops(glcm, 'contrast')[0, 0],
#         "dissimilarity": greycoprops(glcm, 'dissimilarity')[0, 0],
#         "homogeneity": greycoprops(glcm, 'homogeneity')[0, 0],
#         "energy": greycoprops(glcm, 'energy')[0, 0],
#         "correlation": greycoprops(glcm, 'correlation')[0, 0],
#     }
#     return features


# 3. Object Count
@time_it
def count_objects(image_path):
    """Counts the number of objects in the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


# 4. Aspect Ratio
@time_it
def aspect_ratio(image_path):
    """Calculates the aspect ratio of the image."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    return w / h


# 5. Object Density
@time_it
def object_density(image_path):
    """Estimates the density of objects (ratio of their area to the image area)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    image_area = image.shape[0] * image.shape[1]
    return total_area / image_area * 100  # Density in percentage


# 6. Dynamic Range
@time_it
def dynamic_range(image_path):
    """Determines the dynamic range of the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.max() - image.min()


# 7. Frequency Domain Analysis (FFT)
@time_it
def fft_analysis(image_path):
    """Computes the amplitude spectrum of the image using FFT."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


# Error Level Analysis (ELA)
@time_it
def error_level_analysis(image_path, quality=90):
    """Checks error levels in image compression."""
    image = cv2.imread(image_path)
    temp_path = "temp_ela.jpg"
    cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread(temp_path)
    ela_image = cv2.absdiff(image, compressed)
    return ela_image


# Face Detection
@time_it
def detect_face(image_path):
    """Detects faces in the image using Haar Cascades."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces  # Coordinates of face rectangles


@time_it
def face_density(image_path, min_detection_confidence=0.5):
    """Returns the number of faces and their locations in the image."""
    mp_face_detection = mp.solutions.face_detection
    image = cv2.imread(image_path)
    with mp_face_detection.FaceDetection(min_detection_confidence) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            return len(
                results.detections
            )  # , [d.location_data for d in results.detections]
        return 0


@time_it
def face_symmetry(image_path):
    """Calculates face symmetry along the Y-axis."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    left = image[:, : w // 2]
    right = cv2.flip(image[:, w // 2:], 1)  # Flip the right side
    score = ssim(left, right)
    return score


@time_it
def skin_tone_distribution(image_path):
    """Returns the average LAB color values of the skin."""
    image = cv2.imread(image_path)
    image_lab = rgb2lab(image)
    mask = cv2.inRange(
        image, (45, 34, 30), (255, 204, 170)
    )  # Example mask for skin tones
    skin_pixels = image_lab[mask > 0]
    return np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else None


# Shadow and Highlight Analysis
@time_it
def analyze_shadows_and_highlights(image_path):
    """Evaluates shadows and highlights in the image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    high_brightness = np.sum(image > 200)
    low_brightness = np.sum(image < 50)
    total_pixels = image.size
    return {
        "high_brightness_ratio": high_brightness / total_pixels * 100,
        "low_brightness_ratio": low_brightness / total_pixels * 100,
    }


# # Skin Texture Analysis
# def analyze_skin_texture(image_path):
#     """Analyzes skin texture on the face using GLCM."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#     texture_features = {
#         "contrast": greycoprops(glcm, 'contrast')[0, 0],
#         "homogeneity": greycoprops(glcm, 'homogeneity')[0, 0],
#     }
#     return texture_features

@time_it
def detect_rectangular_stamp(image_path):
    """
    Detects a rectangular stamp on the image.
    Returns the coordinates of the rectangle (x, y, w, h) if a stamp is found.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to detect edges
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []

    # Filter rectangular contours
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filtering conditions: minimum and maximum rectangle size
        aspect_ratio = w / float(h)
        # if 1.5 < aspect_ratio < 3.0 and (0.1 * image_h * image_w) < w * h < 0.3 * image_h * image_w:  # Aspect ratio and area
        # return (x, y, w, h)
        if w * h > 0.05 * image_h * image_w:
            result += [[(x, y, w, h, w * h, w * h / float(image_h * image_w))]]

    return result  # None  # If no rectangle is found


@time_it
def visualize_rectangular_stamp(image_path, rect):
    """
    Visualizes the detected rectangular stamp on the image.
    """
    image = cv2.imread(image_path)
    if rect:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Rectangular Stamp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@time_it
def extract_text(image_path, lang="eng"):
    """Extracts text from the image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang=lang).replace("\n\n", "\n")
    text_number = "".join(filter(str.isdigit, text))  # Keep only digits
    return text, text_number


# @time_it
# def extract_card_number(image_path):
#     """Extracts a card number from the image."""
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     custom_config = r"--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
#     text = pytesseract.image_to_string(gray, config=custom_config)  # "--psm 6")
#     # card_number = ''.join(filter(str.isdigit, text))  # Keep only digits
#     return text


# @time_it
# def extract_card_number_q(image_path, pytesseract=None):
#     """
#     Extracts a card number from the image, optimizing the process for speed.
#     """
#     # Load the image
#     image = cv2.imread(image_path)
#
#     # Resize the image to speed up processing
#     scale_percent = 50  # Scale to 50% of the original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#     # # Binarization to improve OCR quality
#     # _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
#
#     # Configure Tesseract to extract digits
#     custom_config = r"--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
#
#     # Extract text
#     # text = pytesseract.image_to_string(binary, config=custom_config)
#     text = pytesseract.image_to_string(gray, config=custom_config)
#
#     # Extract only digits
#     card_number = "".join(filter(str.isdigit, text))
#     return card_number


def calc_timediff(date_1: pd.Timestamp, date_2: str):
    """
    Calculates the time difference between two dates (creation/editing of the photo and upload) without considering time zones.
    """
    try:
        date_2 = pd.to_datetime(date_2, format="%Y:%m:%d %H:%M:%S")
        date_1 = pd.to_datetime(date_1, format="%Y:%m:%d %H:%M:%S")
        time_diff = pd.to_datetime(date_1) - pd.to_datetime(date_2)

    except TypeError as e:
        logger.info("No creation date found in metadata")
        return

    return time_diff.total_seconds() / 60


@time_it
def calc_unique_doctypes(photo_dict: dict) -> dict:
    """
    For each photo, counts the number of unique document types that were photographed
    using the same mobile device model and manufacturer.
    """
    data = deepcopy(photo_dict)
    model_documents, make_documents = {}, {}

    for photo_data in data.values():
        exif = photo_data.get("exif", {})
        model = exif.get("Model")
        make = exif.get("Make")

        doc_type = photo_data["file_name"]
        if doc_type.startswith("card_"):
            doc_type = "card"

        if model:
            model_documents.setdefault(model, set()).add(doc_type)
        if make:
            make_documents.setdefault(make, set()).add(doc_type)

    for photo_data in data.values():
        exif = photo_data.get("exif", {})
        model = exif.get("Model")
        make = exif.get("Make")

        if model:
            photo_data["unique_doctype_count_model"] = len(model_documents[model])
        if make:
            photo_data["unique_doctype_count_make"] = len(make_documents[make])

    return data
