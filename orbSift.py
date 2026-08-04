from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import cv2 as cv
import numpy as np

# Read bytes and convert it into CV image
def decodeImage(file_bytes: bytes) -> np.ndarray:

    
    nparr = np.frombuffer(file_bytes, np.uint8)

    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image format")

    return img

# Use SIFT or ORB to find keypoints and descriptors
def extractFeatures(img1: np.ndarray, img2: np.ndarray, method: str = "SIFT"):
    
    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    
    if method.upper() == "SIFT":
        detector = cv.SIFT_create()
    elif method.upper() == "ORB":
        detector = cv.ORB_create(nfeatures=2000)
    else:
        raise ValueError("Method must be 'SIFT' or 'ORB'")

    kp1, des1 = detector.detectAndCompute(gray1,None)
    kp2, des2 = detector.detectAndCompute(gray2,None)


    if des1 is None or des2 is None:
        raise ValueError("No features found")

    return kp1, des1, kp2, des2


def matchFeatures(des1: np.ndarray, des2: np.ndarray, method: str = "SIFT"):
    
    # If SIFT -> cv2.NORM_L2 -> Simple euclidean
    # If ORB  -> cv2.NORM_HAMMING -> XOR and then counting number of 1's

    if method.upper() == "SIFT":
        normType = cv.NORM_L2
    elif method.upper() == "ORB":
        normType = cv.NORM_HAMMING
    else:
        raise ValueError("Method must be 'SIFT' or 'ORB'")


    # BFMATCHER

    # We match descriptors of both the images in this stage
    # For each descriptor we find Top 2 closest matches

    matcher = cv.BFMatcher(normType=normType)

    matches = matcher.knnMatch(des1,des2,k=2)


    # Lowe's Ratio Test

    # if a keypoint is unique then it's distance from No 1 match is tiny while the distance from No 2 match is huge

    goodMatches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                goodMatches.append(m)

    if len(goodMatches) < 4:
        raise ValueError(f"Not enough good matches ({len(goodMatches)}/4 required)")

    # minimum 4 matches are required bcz RANSAC needs minimum 4

    return goodMatches


def computeHomography(kp1: list, kp2: list, good_matches: list):

    # Changing the format of keypoints in the way Homography needs

    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Ransac uses 4 points, makes a trial homography matrix
    # Uses that trial matrix to check map points from source to destination
    # We check wether the distance is less than 5 btw ORB's point and our H trial's predicted one
    
    H, mask = cv.findHomography(src,dst,cv.RANSAC,5.0)

    if H is None:
        raise ValueError("Rasnsac failed to compute")

    return H, mask

def warpAndStitch(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:

    canvasWidth = img1.shape[1] + img2.shape[1]
    canvasHeight = max(img1.shape[0], img2.shape[0])

    # Warp the image using the computed homography matrix on the canvas
    panorama = cv.warpPerspective(img1, H, (canvasWidth, canvasHeight))

    panorama[0:img2.shape[0], 0:img2.shape[1]] = img2

    return panorama