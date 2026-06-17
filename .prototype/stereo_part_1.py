import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread("data/scene_0.png", cv.IMREAD_GRAYSCALE)  # queryimage # left image
img2 = cv.imread("data/scene_1.png", cv.IMREAD_GRAYSCALE)  # trainimage # right image
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
# fundation matrix F
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)

# --- Depth Estimation ---
h, w = img1.shape

# Rectify without known camera intrinsics using the fundamental matrix.
# H1, H2 are the homographies that make the epipolar lines horizontal.
_, H1, H2 = cv.stereoRectifyUncalibrated(pts1.astype(np.float32), pts2.astype(np.float32), F, imgSize=(w, h))

print(f"## H1: \n{H1}\n\n## H2: \n{H2}\n")

img1_rect = cv.warpPerspective(img1, H1, (w, h))
img2_rect = cv.warpPerspective(img2, H2, (w, h))

# Compute disparity with Semi-Global Block Matching.
# numDisparities must be divisible by 16; blockSize must be odd.
num_disparities = 16 * 10  # search range in pixels
block_size = 11

stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,
    P2=32 * 3 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

disparity = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0

# Without calibration we can't recover metric depth, but relative depth is
# proportional to 1/disparity (closer objects → larger disparity → smaller value).
# Mask out invalid (zero/negative) disparities before inverting.
valid = disparity > 0
depth_relative = np.zeros_like(disparity)
depth_relative[valid] = 1.0 / disparity[valid]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img1_rect, cmap="gray")
axes[0].set_title("Rectified left")
axes[1].imshow(disparity, cmap="plasma")
axes[1].set_title("Disparity")
axes[2].imshow(depth_relative, cmap="inferno")
axes[2].set_title("Relative depth (1/disparity)")
plt.tight_layout()
plt.show()
