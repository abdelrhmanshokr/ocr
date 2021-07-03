# first patch of images 
import cv2, pytesseract, json, re
from pytesseract import Output
from imutils import contours 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import imutils

# 1) crop image so it shows only the card
read_image = cv2.imread('test.jpeg')
# resize since image is huge
resized_image = cv2.resize(read_image, None, fx=0.25, fy=0.25)
#cropping the strip dimensions
blurred = cv2.blur(read_image, (3,3))
canny = cv2.Canny(blurred, 50, 200)
pts = np.argwhere(canny>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)
# crop the region
cropped = read_image[y1:y2, x1:x2]
cv2.imwrite("cropped.jpeg", cropped)


# ROI selector to test a certain area
# just for testing on the spot
img_raw = cv2.imread('cropped.jpeg')
roi = cv2.selectROI(img_raw)
print(roi)
roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite("ROI_crop.jpeg",roi_cropped)
cv2.imshow("ROI", roi_cropped)
cv2.waitKey(0)


# removing noise by using grayscale filter 
# working on the cropped image from the first step
# then get contours to detect text/data from within
# Step 1: Read in the image as grayscale
im = cv2.imread('cropped.jpeg')
cv2.imshow('Original', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Inverse the image to get black background
im2 = im.copy()
im2 = 255 - im2
img_grey = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
cv2.imshow('Inverse', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Get an adaptive binary image
im3 = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('Inverse_binary', im3)
cv2.waitKey(0)
cv2.destroyAllWindows()

rect_cnts = []
hexagon_cnts = []
# Step 4: find contours
contours, hierachy = cv2.findContours(im3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx) == 4:
        rect_cnts.append(cnt)
    elif len(approx) == 6:
        hexagon_cnts.append(cnt)

# sort the hexagon contours to get the bigeest area wise
# then appending the biggest 3 area wise (because it recognises the scratch code area as a hexagon not as a rectangle for some reason)
# if we worked with both rectangles and hexas it will take too much time to get it done
# that's why it's better to append only biggest three/four hexas to the rectangle array
sorted_hexagon_cnts = sorted(hexagon_cnts, key=cv2.contourArea)
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 1])
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 2])
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 3])


print('leeeeen', len(hexagon_cnts))
print(len(rect_cnts))

# Step 5: This creates a white image instead of a black one to plot contours being black
out = 255*np.ones_like(im)
cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
cv2.drawContours(im, contours, -1, (0, 255, 0))
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

date_regex = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
three_digit_scratch_code_regex = '^\d{3}$'
four_digit_scratch_code_regex = '^\d{4}$'

parse_text = []
date, barcode, scratch_code = '', '', ''
highest_barcode_conf, highest_date_conf, highest_scratch_code_conf = 0, 0, 0

# custom_config = r'--oem 3 --psm 8' 
custom_config = r'--oem 3 --psm 6'    
# custom_config = r'--oem 3 --psm 12' 

for rects in rect_cnts:
    box = cv2.boundingRect(rects)
    x,y,w,h = box[:]
    # extract rectangle from the original image
    newimg = im[y:y+h,x:x+w]

    # use 'pytesseract' to get the text in the new image
    # text = pytesseract.image_to_string(Image.fromarray(newimg))
    text = pytesseract.image_to_data(Image.fromarray(newimg), output_type=Output.DICT, config=custom_config)
    print(text['text'])
    for word in text['text']:
        if int(text['conf'][text['text'].index(word)]) >= highest_scratch_code_conf and (re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word)):
            print('from the append block scratch code', word)
            highest_scratch_code_conf = int(text['conf'][text['text'].index(word)])
            scratch_code += word
        elif int(text['conf'][text['text'].index(word)]) >= highest_barcode_conf and re.match(barcode_regex, word):
            print('from the append block barcode', word)
            highest_barcode_conf = int(text['conf'][text['text'].index(word)])
            barcode = word 
        elif int(text['conf'][text['text'].index(word)]) >= highest_date_conf and re.match(date_regex, word):
            print('from the append block date', word)
            highest_date_conf = int(text['conf'][text['text'].index(word)])
            date = word

parse_text.append({ 'barcode': barcode })
parse_text.append({ 'expiry date': date })
parse_text.append({ 'scratch code': scratch_code })
with open('output.txt', 'w') as outfile:
    json.dump(parse_text, outfile)