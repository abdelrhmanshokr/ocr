import cv2, re, json, pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt 

# first crop the card from the given image 
MIN_MATCH_COUNT = 4
# (1) prepare data
template = cv2.imread('test.jpeg')
image_to_match = cv2.imread('test.jpeg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image_to_match, cv2.COLOR_BGR2GRAY)

# (2) Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# (3) Create flann matcher
matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

# (4) Detect keypoints and compute keypointer descriptors
kpts1, descs1 = sift.detectAndCompute(template_gray,None)
kpts2, descs2 = sift.detectAndCompute(gray2,None)

# (5) knnMatch to get Top2
matches = matcher.knnMatch(descs1, descs2, 2)

# Sort by their distance.
matches = sorted(matches, key = lambda x:x[0].distance)

# (6) Ratio test, to get good matches.
good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

canvas = image_to_match.copy()

## (7) find homography matrix
if len(good)>MIN_MATCH_COUNT:
    # (queryIndex for the small object, trainIndex for the scene )
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # find homography matrix in cv2.RANSAC using good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # matchesMask2 = mask.ravel().tolist()
    h,w = template.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))


# (8) drawMatches
matched = cv2.drawMatches(template,kpts1,canvas,kpts2,good,None)#,**draw_params)

# (9) Crop the matched region from scene
h,w = template.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(image_to_match,perspectiveM,(w,h))

# (10) save and display
cv2.imwrite("matched.jpeg", matched)
cv2.imwrite("found.jpeg", found)
cv2.imshow("found", found)
cv2.waitKey(0)

# then applying the same algorithm on the found image
date_regs = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
three_digit_scratch_code_regex = '^\d{3}$'
four_digit_scratch_code_regex = '^\d{4}$'

img = cv2.imread('found.jpeg')
custom_config = r'--oem 3 --psm 11'

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, threshimg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
dilation = cv2.dilate(threshimg, rect_kernel, iterations = 1) 
img_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)

for cnt in img_contours:
    x, y, w, h = cv2.boundingRect(cnt)  
    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cropped_img = img[y:y + h, x:x + w] 
    text = pytesseract.image_to_data(cropped_img, output_type=Output.DICT, config=custom_config)

parse_text = []
scratch_code = ''

for word in text['text']:
    if re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word):
        print('from the append block scrath code', word)
        scratch_code += word
    elif re.match(barcode_regex, word):
        print('from the append block barcode', word)
        parse_text.append({ 'barcode': word })
    elif re.match(date_regs, word):
        print('from the append block date', word)
        parse_text.append({ 'expiry date': word })
parse_text.append({ 'scratch code': scratch_code })

with open('output.txt', 'w') as outfile:
    json.dump(parse_text, outfile)