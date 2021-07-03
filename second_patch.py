# second patch of images it's slightly different so it needs to be handled it some other way 
import cv2, pytesseract, json, re
from pytesseract import Output
from imutils import contours 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import imutils

# 1) crop image so it shows only the card
# read image
imgc = cv2.imread('test.jpeg')
# resize since image is huge
img = cv2.resize(imgc, None, fx=0.25, fy=0.25)
#cropping the strip dimensions
blurred = cv2.blur(imgc, (3,3))
canny = cv2.Canny(blurred, 50, 200)
pts = np.argwhere(canny>0)
y1,x1 = pts.min(axis=0)
y2,x2 = pts.max(axis=0)
# crop the region
cropped = imgc[y1:y2, x1:x2]
#Select the bounded area around white boundary
tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
cv2.imwrite("cropped.jpeg", tagged)
r = cv2.selectROI(tagged)
imcrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#Bounded Area
cv2.imshow("taggd2.jpeg", imcrop)
cv2.imwrite('roi_crop.jpeg', imcrop)
cv2.waitKey(0)


# 2) rotate the cropped image so it's horizontal levelled 
# 3) remove noise 
# 4) extract the three wanted regions 
img_raw = cv2.imread('cropped.jpeg')

#select ROI function
roi = cv2.selectROI(img_raw)

#print rectangle points of selected roi
print(roi)

#Crop selected roi from raw image
roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

#show cropped image
cv2.imwrite("ROI_crop.jpeg",roi_cropped)
cv2.imshow("ROI", roi_cropped)
cv2.waitKey(0)

# image = cv2.imread("cropped.jpeg", -1)

# resize image to speed up computation
# rows,cols,_ = image.shape
# image = cv2.resize(image, (np.int32(cols/2),np.int32(rows/2)))

# # convert to gray and binarize
# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

# # note: erosion and dilation works on white forground
# binary_img = cv2.bitwise_not(binary_img)

# # dilate the image to fill the gaps
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# dilated_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel,iterations=4)

# # find contours, discard contours which do not belong to a rectangle
# contours,hierachy= cv2.findContours(dilated_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# # Rectangular contours 
# rect_cnts = [] 
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
#     if len(approx) == 4:
#         rect_cnts.append(cnt)

# # sort contours based on area
# rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)[:1]

# # find bounding rectangle of biggest contour
# for rects in rect_cnts:

#     box = cv2.boundingRect(rects)
#     x,y,w,h = box[:]

#     # extract rectangle from the original image
#     newimg = image[y:y+h,x:x+w]

#     # use 'pytesseract' to get the text in the new image
#     text = pytesseract.image_to_string(Image.fromarray(newimg))
#     print(text)

#     # cv2.imwrite('cropped_parcode.jpeg', newimg)
#     cv2.imshow('Image', newimg)
#     cv2.waitKey(0)
# print(len(rect_cnts))


# Step 1: Read in the image as grayscale - Note the 0 flag
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
sorted_hexagon_cnts = sorted(hexagon_cnts, key=cv2.contourArea)
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 1])
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 2])
rect_cnts.append(sorted_hexagon_cnts[len(sorted_hexagon_cnts) - 3])

# Step 5: This creates a white image instead of a black one to plot contours being black
out = 255*np.ones_like(im)
cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
cv2.drawContours(im, contours, -1, (0, 255, 0))
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('leeeeen', len(hexagon_cnts))
print(len(rect_cnts))

date_regex = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
# TODO replace the following regexs with one regex representing the whole 17 digits of the scratch code
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


# large = cv2.imread('ROI_crop.jpeg')
# rgb = cv2.pyrDown(large)
# small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

# _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
# connected = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

# contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# count = 0
# mask = np.zeros(bw.shape, dtype=np.uint8)

# date_regex = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
# barcode_regex = '^\d{12}$'
# three_digit_scratch_code_regex = '^\d{3}$'
# four_digit_scratch_code_regex = '^\d{4}$'

# parse_text = []
# scratch_code = ''
# date, barcode, scratch_code = '', '', ''
# highest_barcode_conf, highest_date_conf, highest_scratch_code_conf = 0, 0, 0

# print(len(contours))
# for idx in range(len(contours)):
#     count += 1 
#     x, y, w, h = cv2.boundingRect(contours[idx])
#     # mask[y:y+h, x:x+w] = 0
#     # newimg = large[y:y+h,x:x+w]
#     cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
#     r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
#     # text = pytesseract.image_tso_string(Image.fromarray(rgb))
#     # print(text)
#     text = pytesseract.image_to_data(Image.fromarray(rgb), output_type=Output.DICT, config=custom_config)
#     for word in text['text']:
#         if int(text['conf'][text['text'].index(word)]) >= highest_scratch_code_conf and (re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word)):
#             print('from the append block scratch code', word)
#             # update highest conf 
#             highest_scratch_code_conf = int(text['conf'][text['text'].index(word)])
#             scratch_code += word
#         elif int(text['conf'][text['text'].index(word)]) >= highest_barcode_conf and re.match(barcode_regex, word):
#             print('from the append block barcode', word)
#             # update highest conf
#             highest_barcode_conf = int(text['conf'][text['text'].index(word)])
#             barcode = word 
#         elif int(text['conf'][text['text'].index(word)]) >= highest_date_conf and re.match(date_regex, word):
#             print('from the append block date', word)
#             # update highest conf
#             highest_date_conf = int(text['conf'][text['text'].index(word)])
#             date = word
#     print(text['text'])
#     print('config', text['conf'])
#     print('new iteration', idx)
#     # if r > 0.45 and w > 8 and h > 8:
#     #     # print('in if', text['text'])
#     #     cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

# parse_text.append({ 'barcode': barcode })
# parse_text.append({ 'expiry date': word })
# parse_text.append({ 'scratch code': scratch_code })
# with open('output.txt', 'w') as outfile:
#     json.dump(parse_text, outfile)
# # print(count)

# # cv2.imshow('rects', rgb)
# # cv2.waitKey(0)