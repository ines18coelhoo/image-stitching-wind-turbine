from unittest import result
import cv2 
import numpy as np
import math
import random
import os
import time

#Colours RGB code 
PINK = (147, 20, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
featureRadius = 10
featureThickness = 6
line_thickness = 6

# Number of SIFT features per image
no_sift_features = 10000 # 7000

# Threshold for ratio between 2 best correspondences
sift_threshold = 0.8 # 0.6

# Variables for RANSAC
n_ransac = 4 # number of correspondences for random sample [4<n<10] 6
e_ransac = 0.75 # fraction of false correspondences 0.2
p_ransac = 0.99 # prob. of atleast one trial being free of outliers 0.99
del_ransac = 3.0 # decision threshold to construct the inlier set 3.0

# Variables for Levenberg-Marquardt Optimization
LM_threshold = 0.00000000000000000000001

input_folder = 'data/no_background/'
output_folder = 'data/results/'

def detectAndDescribe(image, method):
    """
        Compute key points and feature descriptors using an specific method
            :param image: image from where the kps are extracted
            :param method: feature extraction method
            :return: kps and feature descriptors
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'beblid':
        descriptor = cv2.xfeatures2d.BEBLID_create(1000)
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create(no_sift_features)
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'brief':
        descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create(10000)
    
    # get keypoints and descriptors
    #kpts = detector.detect(image, None)
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)


def find_sift_correspondences(kp1, des1, kp2, des2):
    """
        Function to find the correspondences between feature points in image1
        to those in image2
            :param kp1: keypoints in image 1
            :param des1: feature descriptors for keypoints in kp1
            :param kp2: keypoints in image 2
            :param des2: feature descriptors for keypoints in kp2
    """
    print("Determining correspondences")
    class data: # class structure used for sorting point pairs
        def __init__(self, point_pair, value):
            self.point_pair = point_pair
            self.value = value
    
    des1_order = des1.shape
    des2_order = des2.shape
    correspondences = []
    corr_new = []
    # Find best 2 matches for each feature descriptor of image 1 with
    # those of image 2
    for i in range(des1_order[0]):
        pairs = [data([], np.Inf), data([], np.Inf)]
        point1 = kp1[i].pt

        for j in range(des2_order[0]):
            point2 = kp2[j].pt
            dist = np.linalg.norm(des1[i, :] - des2[j, :])
            new_pair = [point1, point2]
            pairs.append(data(new_pair, dist))
            pairs = sorted(pairs, key=lambda x: x.value)
            pairs.pop() # maintain the number of elements as 2
        
        # Find ratio of distances to best and second best matches
        thisvalue = pairs[0].value / pairs[1].value
        # Sort the correspondences based on this ratio
        correspondences.append(data(pairs[0].point_pair, thisvalue))
    
    # Sort point pairs based on ratio of euclidean
    # distance between best and second best matches
    correspondences = sorted(correspondences, key=lambda x: x.value)

    # Choose only the best point pairs as output, i.e. small value of ratio
    for i in range(len(correspondences)):
        if correspondences[i].value > sift_threshold:
            break # exit loop when the correspondence strength is low
        corr_new.append(correspondences[i].point_pair)
    
    return corr_new

def ransac(correspondences):
    """
        Function that carries out RANSAC algorithm for the input correspondences
            :param correspondences: list of correspondences between images.
            :return Homography from first set to second set of points.
            Also return the inliers.
    """
    print("RANSAC Algorithm Started.")
    inliers = []
    # Number of trials
    N_ransac = math.log(1 - p_ransac) / math.log(1 - (1 - e_ransac) ** n_ransac)
    N_ransac = int(math.ceil(N_ransac))
    print("N_ransac = {}".format(N_ransac))
    # Total number of correspondences
    n_total = len(correspondences)
    if n_total < n_ransac:
        return inliers
    print("No. of correspondences = {}".format(n_total))
    # Minimum size of inlier set
    M_ransac = (1 - e_ransac) * n_total
    M_ransac = int(math.ceil(M_ransac))
    print("Minimum required size of inlier set = {}".format(M_ransac))
    num_inliers = -1 # number of inliers
    
    # Strip the correspondences into X and Xdash, so that these can be used
    # in other functions readily.
    X = []
    Xdash = []
    for point_pair in correspondences:
        X.append(point_pair[0][0])
        X.append(point_pair[0][1])
        X.append(1)
        Xdash.append(point_pair[1][0])
        Xdash.append(point_pair[1][1])
        Xdash.append(1)
    X = np.array(X, dtype='float64')
    Xdash = np.array(Xdash, dtype='float64')
    X = X.reshape(-1, 3)
    X = X.T
    Xdash = Xdash.reshape(-1, 3)
    Xdash = Xdash.T

    # Run RANSAC loop for N_ransac iterations
    for trial in range(N_ransac):
        # choose n random samples
        this_sample = random.sample(correspondences, n_ransac)
        H = compute_homography(this_sample) # compute homography.
        if H is None:
            continue
        # Find out the inliers in correspondences with del_ransac as
        # decision threshold
        inliers_temp = find_inliers(H, X, Xdash, del_ransac, correspondences)
        if len(inliers_temp) > num_inliers:
            num_inliers = len(inliers_temp)
            inliers = inliers_temp

    print("Maximum inlier size obtained = {}".format(num_inliers))
    if num_inliers < M_ransac or num_inliers < n_ransac:
        print("Warning: Minimum size of inlier set not obtained!")
    
    return inliers

def compute_homography(sample):
    """
        Function that calls gethomography_multi after
        arranging data as X and Xdash from sample.
        Sample contains correspondences as pairs of points.
        It also returns the X and Xdash values used.
    """
    points1 = []
    points2 = []
    for point_pair in sample:
        points1.append(point_pair[0][0])
        points1.append(point_pair[0][1])
        points2.append(point_pair[1][0])
        points2.append(point_pair[1][1])
    points1 = np.array(points1).reshape(-1, 1, 2)
    points2 = np.array(points2).reshape(-1, 1, 2)

    """
    Methods:
        - 0 : no outliers
        - RANSAC : handle practically any number of outliers 
        - LMEDS : more than 50% of inliers, does not need any threshold 
        - PROSAC [RHO] : , RHO, 3.0
    """
    (opencv_H, status) = cv2.findHomography(points1, points2, 0)
    return opencv_H
    #return gethomography_multi(points1, points2)

def compute_affine_transform(sample):
    points1 = []
    points2 = []
    for point_pair in sample:
        points1.append(point_pair[0][0])
        points1.append(point_pair[0][1])
        points2.append(point_pair[1][0])
        points2.append(point_pair[1][1])
    points1 = np.array(points1).reshape(-1, 1, 2)
    points2 = np.array(points2).reshape(-1, 1, 2)
    opencv_H, st = cv2.estimateAffinePartial2D(points1, points2)
    print(f'AFFINE TRANSFORM: {opencv_H}')
    row = np.array([0,0,1])
    return np.vstack([opencv_H, row])

def gethomography_multi(X, Xdash):
    """
        Function that returns the homography from X to Xdash (given Xdash = HX)
        for more than 4 points of correspondence
        X is the coordinates of the points in the domain of the mapping in the form
        [x1,y1,x2,y2,x3,y3,x4,y4,...].T
        Xdash is the coordinates of the points in the range of the mapping in the
        form [x1',y1',x2',y2',x3',y3',x4',y4',...].T
    """
    # defining A
    try:
        assert (X.shape == Xdash.shape)
    except:
        print('Error in dimensions of inputs to the function >>GETHOMOGRAPHY')
    A = np.array([]) # define A as a vector and reshape later
    for i in range(X.shape[0] // 2):
        A = np.append(A, [X[2 * i][0], X[2 * i + 1][0], 1, 0, 0, 0,
                     -X[2 * i][0] * Xdash[2 * i][0],
                     -X[2 * i + 1][0] * Xdash[2 * i][0], 0, 0, 0,
                      X[2 * i][0], X[2 * i + 1][0],
                      1, -X[2 * i][0] * Xdash[2 * i + 1][0],
                     -X[2 * i + 1][0] * Xdash[2 * i + 1][0]])
    
    A = A.reshape(X.shape[0], 8)
    try:
        ATAinv = np.linalg.inv(np.dot(A.T, A))
    except:
        print('A is not invertible!!! >>GETHOMOGRAPHY')

    h = np.dot(ATAinv, np.dot(A.T, Xdash)) # h still doesn't have (3,3)th element.
    assert (h.shape == (8, 1))
    h = np.append(h, [[1]], axis=0)
    return h.reshape(3, 3) # return the 3x3 H matrix

def find_inliers(H, X, Xdash, delta, correspondences):
    """
        The function to find the inliers for a given H, X, Xdash set, and delta.
        The list of correspondences is also input so as to reduce regeneration
        of point pairs list.
        Returns the list of inliers.
    """
    Y = np.matmul(H, X) # Actual mapping with H
    Y = Y / Y[2, :] # Normalising wrt x3 in HC repr.
    # But Xdash is the observed mapping
    error = Y - Xdash # 3rd element in each point's HC becomes 0.
    # So, #peace during sum.
    squared_error = error ** 2
    d_squared = np.sum(squared_error, axis=0)
    ind = np.where(d_squared <= delta ** 2)
    
    return [correspondences[i] for i in ind[0]]

def find_H_between_a_pair(img1, img2, i, LM_on=True):
    """
        Function to find the homography between a pair of images based on
        the feature descriptors in the pair of images
        outfile is the file to save the image showing correpondence match
        between the pair of input images
            :param LM_on: Nonlinear Optim with LM alg on/off.
            :return Homography from img1 to img2
    """
    # Detect and describe kps based on SIFT descriptors
    (kps1, features1) = detectAndDescribe(img1, 'sift')
    (kps2, features2) = detectAndDescribe(img2, 'sift')
    
    # Determine the correspondences between the pair of images
    corr = find_sift_correspondences(kps1, features1, kps2, features2)

    # Find the homography based on RANSAC algorithm
    inliers = ransac(corr)
    # Check if inliers are in the edge of the blade.
    # If so remove them.
    
    new_inliers = []
    for point in inliers:
        point1 = point[0]
        point2 = point[1]
        point1 = (round(point1[0]), round(point1[1]))
        point2 = (round(point2[0]), round(point2[1]))
        if img1[point1[1]][point1[0]+50] < 20 or img1[point1[1]][point1[0]-50] < 20 or img2[point2[1]][point2[0]+50] < 20 or img2[point2[1]][point2[0]-50] < 20 or point1[1] > point2[1]:
            print("pop")
        else:
            new_inliers.append(point)

    print(f'\nnew inliers: {new_inliers}\n')
    if len(new_inliers) < 3:
        status = -1
        H = None
        # TODO
    else: 
        status = 1
        H = compute_affine_transform(new_inliers)
    print(H)    
    # Generate image showing inliers and outliers
    image_inlier_outlier(img1, img2, new_inliers, corr, [GREEN, RED], i)
    
    return H, status

def image_inlier_outlier(img1, img2, inliers, corr, colors, i):
    """
        Function to save and return images showing inliers and outliers in
        correspondences between 2 images
            :param img1file: file name 1
            :param img2file: file name 2
            :param infolder: input files' folder
            :param outfile: output filename's first part
            :param outfolder: output folder
            :param inliers: set of inliers
            :param corr: set of all correspondences, of which inliers is a part
            :param colors: list of 2 colors, in the order [inlier-color, outlier-color]
            :return [inlier image, outlier image]
    """
    # Generate outlier set
    outliers = []
    for point_pair in corr:
        if point_pair not in inliers:
            outliers.append(point_pair)
    
    image_inliers = get_matching_image(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), inliers, colors[0])
    cv2.imwrite(f'{output_folder}inliers_{i}.jpg', image_inliers)
    
    image_outliers = get_matching_image(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), outliers, colors[1])
    cv2.imwrite(f'{output_folder}outliers_{i}.jpg', image_outliers)
    
    print("Inliers and outliers saved to images")
    return [image_inliers, image_outliers]

def get_matching_image(image1, image2, correspondences, color):
    """
        Function that returns the image matching the correspondences
            :param image1: 1st image
            :param image2: 2nd image
            :param correspondences: pairs of pixels with correspondences
            between image1 and image2
            :param color: Color for matching line
            :return: image with correspondences matched
    """
    shape1 = image1.shape
    shape2 = image2.shape
    height = max(shape1[0], shape2[0])
    img1 = np.zeros((height, shape1[1], 3), dtype='uint8')
    img1[0:height, 0:shape1[1]] = image1
    img2 = np.zeros((height, shape2[1], 3), dtype='uint8')
    img2[0:height, 0:shape2[1]] = image2
    # generate an image with image 1 and 2 on left and right
    image = np.concatenate((img1, img2), axis=1)
    for point_pair in correspondences:
        point1 = point_pair[0]
        point2 = point_pair[1]
        point1 = (int(round(point1[0])), int(round(point1[1])))
        point2 = (int(round(point2[0])), int(round(point2[1])))
        cv2.circle(image, tuple(point1), featureRadius,
        color, featureThickness, cv2.LINE_AA)
        cv2.circle(image, tuple([point2[0] + shape1[1], point2[1]]), featureRadius,
        color, featureThickness, cv2.LINE_AA)
        cv2.line(image, tuple(point1), tuple([point2[0] + shape1[1], point2[1]]),
        color, line_thickness)

    return image

def generate_panorama(imagefiles, infolder, panorama_out, outfolder, LM_on):
    """
        Function to generate the panorama
            :parameter imagefiles: list of filenames of images
            :parameter panorama_out: output file name for panorama
            :parameter LM_on: Non linear optimn with LM on/off
    """
    
    gray_img = []
    for image in imagefiles:
        img = cv2.imread(f'{input_folder}/{image}', 0)
        print(f'Reading img: {image}')
        gray_img.append(img)

    image_count = len(imagefiles)
    # Determine the homographies between each of the adjacent pairs
    # of images
    H_all = []
    #Downscale images 90%
    H_all.append(np.float32([[0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 1]]).reshape(3, 3))
    
    
    for i in range(image_count-1):
        print("\nProcessing Image Pair {}\n----------------------------".format(i + 1))
        # Store images in different variables for future use
        aux_img1 = gray_img[i]
        aux_img2 = gray_img[i+1]        
        status = 0
        cl = 2
        tgs = 8
        while True:
            H, status = find_H_between_a_pair(aux_img1, aux_img2, i, LM_on)
            if status == 1:
                print("HOMOGRAPHY FOUND")
                break
            else:
                print(f'HOMOGRAPHY NOT FOUND!!!!!!\n cl = {cl} \t tgs = {(tgs,tgs)} ')
                clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(tgs,tgs))
                aux_img1 = clahe.apply(aux_img1)
                aux_img2 = clahe.apply(aux_img2)
                cl += 1
                tgs += 2
        
        H_all.append(H)
    
    """
    # [TIP OF THE BLADE]
    # 1. Find A & B
    p_A = find_edge_pixel(gray_img[1].shape[0]-1, 0, gray_img[1])
    p_B = find_edge_pixel(gray_img[1].shape[0]-1, -1, gray_img[1])
    # 2. Compute A' & B'
    p_A_l = warp_edge_point(np.linalg.inv(H_all[1]), p_A)
    p_B_l = warp_edge_point(np.linalg.inv(H_all[1]), p_B)
    # 3. Find C' & D'
    x_edge_pts = []
    y_edge_pts = [] 
    for i in range(0, 2500, 600):
        pt = find_edge_pixel(i, -1, gray_img[0])
        x_edge_pts.append(pt[0])
        y_edge_pts.append(pt[1])
    a, b, c = np.polyfit(x_edge_pts, y_edge_pts, 2)
    m, d = get_line_parameters(p_A_l[0], p_A_l[1], p_B_l[0], p_B_l[1])

    # 4. Compute intersection A'B' & C'D'
    p_E_l = get_intersection_point(a, b, c, m, d)

    # 5. Compute E'
    p_E = warp_edge_point(H_all[1], p_E_l)
    # 6. Rescale Image
    H_rescale = rescale_homography(p_A, p_B, p_E)
    H_all[1] = np.matmul(np.linalg.inv(H_rescale), H_all[1])
    
    """
    print('\nRescaling Homographies\n----------------------')
    H_rescale = np.eye(3, dtype=np.float32)
    # Adjust the Homographies to align the right edge of the blade
    for i in range(2,image_count):
        # 1. Find A & B
        p_A = find_edge_pixel(gray_img[i].shape[0]-1, 0, gray_img[i])
        p_B = find_edge_pixel(gray_img[i].shape[0]-1, -1, gray_img[i])

        # 2. Compute A' & B'
        p_A_l = warp_edge_point(np.linalg.inv(H_all[i]), p_A)
        p_B_l = warp_edge_point(np.linalg.inv(H_all[i]), p_B)
        p_A_l = warp_edge_point(H_rescale, p_A_l)
        p_B_l = warp_edge_point(H_rescale, p_B_l)

        # 3. Find C' & D'
        p_C_l = find_edge_pixel(gray_img[i-1].shape[0]//2, -1, gray_img[i-1])
        p_D_l = find_edge_pixel(0, -1, gray_img[i-1])
        p_C_l = warp_edge_point(H_rescale, p_C_l)
        p_D_l = warp_edge_point(H_rescale, p_D_l)

        # 4. Compute intersection A'B' & C'D'
        p_E_l = line_intersection([(p_A_l[0],p_A_l[1]),(p_B_l[0],p_B_l[1])],[(p_C_l[0],p_C_l[1]),(p_D_l[0],p_D_l[1])])
        p_E_l = warp_edge_point(np.linalg.inv(H_rescale), p_E_l)

        # 5. Compute E'
        p_E = warp_edge_point(H_all[i], p_E_l)

        # If the point is already where it is supposed to be 
        if abs(p_E[0] - p_B[0]) < 5:
            print('\nskip...\n')
            continue

        # 6. Rescale Homography
        H_rescale = rescale_homography(p_A, p_B, p_E)
        H_all[i] = np.matmul(np.linalg.inv(H_rescale), H_all[i])

    
    # Compute all homographies with respect to the first image.
    H_wrt_ref = H_all[0] 

    for i in range(1, len(H_all)):
        H_wrt_ref = np.matmul(H_wrt_ref, np.linalg.inv(H_all[i]))
        H_all[i] = H_wrt_ref
        print(f'Homography {i}: {H_all[i]}')

    # Now the first image needs to be placed on the bottom of the panorama
    # Hence, we need to multiply each of the homographies with the translation matrix too

    # To find out the translation matrix one has to find out the coordinates of the upper framer
    rows, cols = gray_img[0].shape[:2]
    rows*=0.1
    cols*=0.1
    temp_points = np.float32([[0, 0], 
                              [0, math.ceil(rows)], 
                              [math.ceil(cols), math.ceil(rows)], 
                              [math.ceil(cols), 0]]).reshape(-1,1,2)

    # The last homography has info about the largest translation 
    list_of_points = cv2.perspectiveTransform(temp_points, H_all[-1])
    list_of_points = np.concatenate((temp_points, list_of_points), axis=0) 

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)

    H_trans = np.array([1, 0, -x_min, 0, 1, -y_min, 0, 0, 1], dtype=float).reshape(3,3)
    
    # Translating all the homographies
    for i in range(len(H_all)):
        H_all[i] = np.matmul(H_trans, H_all[i])
        print(f'H translated {H_all[i].shape}: {H_all[i]}')

    # Stitch together all the images
    print(f'xmin = {x_min}')
    image = stitch_images(imagefiles, infolder, H_all, math.ceil(-x_min+cols+1800), math.ceil(-y_min+rows))
    print("Panorama generated")
    cv2.imwrite(outfolder + panorama_out, image)
    print("Panorama saved to file")
    return image

def stitch_images(imagefiles, infolder, H_all, width, height):
    """
        Function to stitch together all the images based on H_all
    """
    print("\nStitching images\n----------------------")

    image_all = []
    for image_name in imagefiles:
        image = cv2.imread(infolder + image_name)
        image_all.append(image)

    print(f'Canva width = {width} \t canva height = {height}')
    # Generate canvas
    canvas = np.zeros((height, width, 3), np.uint8)
    print("Length of image_all: {}".format(len(image_all)))
    print("Length of H_all: {}".format(len(H_all)))
    for count, image in enumerate(image_all):
        canvas = applyhomography(image, canvas, H_all[count])
    
    return canvas

def find_edge_pixel(y_coord, x_l_r, image):
    try:
        foreground_pixels = np.column_stack(np.where(image[y_coord,:] > 0))
        return np.array([foreground_pixels[x_l_r][0], y_coord, 1]).reshape(3,1)
    except:
        return None

def warp_edge_point(H, p):
    # p' = H * p
    p_linha = np.matmul(H, p)
    p_linha /= p_linha[2,:] # Normalize
    return p_linha

def get_line_parameters(x1, y1, x2, y2):
    m = (y2-y1) / (x2-x1)
    d = y1 - m * x1
    return m, d

def get_intersection_point(a, b, c, m, d):
    x = ((m-b) + np.sqrt((b-m)**2 - 4 * a * (c-d))) / (2 * a)
    y = a*x**2 + b*x + c
    return np.array([x, y, 1]).reshape(3,1)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.float32([x[0], y[0], 1]).reshape(3,1)

def rescale_homography(p_A, p_B, p_E):
    
    # Pseudo-inverse
    M = np.float32([[1, 0, p_A[0][0]], 
                    [0, 1, p_A[1][0]],
                    [1, 0, p_B[0][0]],
                    [0, 1, p_B[1][0]]]).reshape(4,3)
    
    y = np.float32([p_A[0][0], p_A[1][0], p_E[0][0], p_E[1][0]]).reshape(4,1)
    
    # theta = [tx, ty, s]
    theta = np.linalg.inv(np.matmul(M.T, M))
    theta = np.matmul(theta, M.T)
    theta = np.matmul(theta, y)
    # rescaling mat = [s, 0, tx]
    #                 [0, s, ty]
    #                 [0, 0,  1]
    rescaling_mat = np.float32([[theta[2][0], 0, theta[0][0]],
                                [0, theta[2][0], theta[1][0]],
                                [0, 0, 1]]).reshape(3,3)

    return rescaling_mat

def applyhomography(image, canvas, H):
    """ 
    Function to project image onto canvas
        :param image: Input image to transform
        :param canvas: Canvas
        :param H: Homography from image to canvas
        :return: canvas with image projected
    """
    tic = time.time()
    print('Projecting to canvas \nTimer started')
    temp = image.shape # Get dimensions of image
    ipimageH = temp[0]
    ipimageL = temp[1]
    H = np.linalg.inv(H) # Invert H so that H is the homography
    # from outputimage to image
    worldH = canvas.shape[0]
    worldL = canvas.shape[1]

    # Define a matrix for the canvas pixel coordinates
    worldCoordinates = np.array([[i, j, 1] for i in range(worldL) for j in range(worldH)])
    
    worldCoordinates = worldCoordinates.T # create a matrix with all the pixels in the
    # canvas. These are surely integers.

    transformedCoord = np.dot(H, worldCoordinates) # apply homography
    transformedCoord = transformedCoord / transformedCoord[2, :] # normalize the points
    # with x3 so as to get the points in the form [x,y,1].T. Note that the coordinates
    # will be float values.
    toc = time.time()
    print('Transformed coordinates computed at: ' + str(toc - tic) + ' seconds')
    
    [validWorld, validTransWorld] = getValidPixels(worldCoordinates, transformedCoord,
                                                   0, ipimageL - 2, 0, ipimageH - 2)
    # Get the pixel coordinates of the canvas that need to be changed, and their
    # corresponding transformations.
    toc = time.time()
    print('Pixels that need to be updated determined at: ' + str(toc - tic) + ' seconds')

    numbPts = validWorld.shape[1]
    for i in range(numbPts): # Loop through each of the valid canvas pixels and replace
        # with ipimage pixels
        Pointnew = validTransWorld[:, i] # Pointnew is the transformed point in the ipimage.
        # It is of type float. It is of shape (3,)
        canvas[validWorld[1][i]][validWorld[0][i]][:] = getpixel(image, Pointnew)
        # Get the ipimage-pixel from ipimage
    
    toc = time.time()
    print('Image Projection finished at: ' + str(toc - tic) + ' seconds')
    return canvas

def getpixel(image, floatPoint):
    """
        Function to find the pixel value at a point on the image. As the point
        is not an integer, weighted average based on L2norm is used
    """

    x = int(np.floor(floatPoint[0]))
    y = int(np.floor(floatPoint[1]))
    Point = np.array([floatPoint[0], floatPoint[1]])
    d00 = np.linalg.norm(Point - np.array([x, y]))
    d01 = np.linalg.norm(Point - np.array([x, y + 1]))
    d10 = np.linalg.norm(Point - np.array([x + 1, y]))
    d11 = np.linalg.norm(Point - np.array([x + 1, y + 1]))
    
    return (image[y][x][:] * d00 + image[y + 1][x][:] * d01 + image[y][x + 1][:] * d10
            + image[y + 1][x + 1][:] * d11) / (d00 + d01 + d10 + d11)

def getValidPixels(worldCoordinates, transformedCoord, xLow, xUp, yLow, yUp):
    """
        :param worldCoordinates: All the coordinates in the world for which
        corresponding transformation is trans..Coord
        :param transformedCoord: Transformed values for world coordinates
        :xLow, xUp, yLow, yUp: allowed range of x and y in transformed output
        :return: [validWorld, validTransWorld] Valid pixels in the world and their
        coreesponding pixels in the transformed world
    """
    temp = transformedCoord[0, :] >= xLow
    worldCoordinates = worldCoordinates[:, temp]
    transformedCoord = transformedCoord[:, temp]

    temp = transformedCoord[0, :] <= xUp
    worldCoordinates = worldCoordinates[:, temp]
    transformedCoord = transformedCoord[:, temp]

    temp = transformedCoord[1, :] >= yLow
    worldCoordinates = worldCoordinates[:, temp]
    transformedCoord = transformedCoord[:, temp]

    temp = transformedCoord[1, :] <= yUp
    worldCoordinates = worldCoordinates[:, temp]
    transformedCoord = transformedCoord[:, temp]
    
    return [worldCoordinates, transformedCoord]



img_files = sorted(os.listdir(input_folder))
image = generate_panorama(img_files, input_folder, 'panorama_with_my_LM.jpg', output_folder, True)
