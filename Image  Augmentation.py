import cv2
import numpy as np


def feature_detection(img, algorithm='ORB', process='Detect', features=10000):
    global det_alg
    # Initiate ORB object
    if algorithm == 'HCD':
        keypoints = cv2.cornerHarris(img, 2, 3, 0.04)
        keypoints = cv2.dilate(keypoints, None, iterations=2)
    elif algorithm == 'SIFT':
        det_alg = cv2.cv2.xfeatures2d.SIFT_create(features)
    elif algorithm == 'SURF':
       det_alg = cv2.xfeatures2d.SURF_create(features)
    elif algorithm == 'ORB':
        det_alg = cv2.ORB_create(nfeatures=features)
        keypoints, descriptors = det_alg.detectAndCompute(img, None)
    elif algorithm == 'FAST':
        det_alg = cv2.FastFeatureDetector_create(features)
        keypoints = det_alg.detect(img, None)
    elif algorithm == 'BRIEF':
        det_alg = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        keypoints = det_alg.detect(img, None)

    return keypoints, descriptors



def compute_matches(descriptors_input, descriptors_output, algorithm='Flann'):
    global mat_alg


    if algorithm == 'Flann':
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        mat_alg = cv2.FlannBasedMatcher(index_params, search_params)
    elif algorithm == 'Brute-Force Matcher':
        mat_alg = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    if (len(descriptors_output) != 0 and len(descriptors_input) != 0):
        matches = mat_alg.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(descriptors_output, np.float32),
                                 k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.69 * n.distance:
                good.append(m)
        return good
    else:
        return None




def main(input_image, aug_image, vid, process='Augment'):

    global alg, det_alg, mat_alg
    alg = 'ORB'
    MIN_MATCHES = 100
    det_alg = None
    mat_alg = None

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    input_keypoints, input_descriptors = feature_detection(gray_image, algorithm=alg)

    if process == 'Detect':
        if alg == 'HCD':
            final = input_image
            final[input_keypoints > 0.01 * input_keypoints.max()] = [0, 0, 255]
        else:
            final = cv2.drawKeypoints(gray_image, input_keypoints, gray_image, (0, 255, 0))
        return final





    vid = cv2.resize(vid, (500, 500))
    vid_gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    if(len(input_keypoints) > MIN_MATCHES):
        output_keypoints, output_descriptors = det_alg.detectAndCompute(vid_gray, None)
        matches = compute_matches(input_descriptors, output_descriptors)
    else:
        return None

    if process == 'Match':
        if (matches!=None):
            output_final = cv2.drawMatchesKnn(input_image, input_keypoints, vid_gray, output_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return output_final
        else:
            return gray_image



    if (matches != None):
        if (len(matches) > 10):
            src_pts = np.float32([input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()
            pts = np.float32([[0, 0], [0, 399], [299, 399], [299, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            M_aug = cv2.warpPerspective(aug_image, M, (500, 500))

            # getting the frame ready for addition operation with Mask Image
            frameb = cv2.fillConvexPoly(frame, dst.astype(int), 0)
            Final = frameb + M_aug

            return Final
        else:
            return vid





input_image = cv2.imread('download.png')
input_image = cv2.resize(input_image, (500, 500))
aug_image = cv2.imread('26776-cool-orange-square-background-pattern.jpg')
aug_image = cv2.resize(aug_image, (500, 500))

cap = cv2.VideoCapture('5.mp4')
ret, frame = cap.read()

while(ret):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))


    out_frame = main(input_image, aug_image, frame, process='Detaect')

    if out_frame.any() == None:
        continue
    cv2.imshow('ORB keypoints', out_frame)

    key = cv2.waitKey(5)
    if (key == 27):
        break