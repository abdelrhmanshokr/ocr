import cv2, re, json, pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt

# generatets sharpened image from which text can be extracted if needed
# image = cv2.imread('all_images.jpeg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
# thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# result = 255 - close

# cv2.imwrite('sharpen.jpeg', sharpen)
# cv2.imwrite('thresh.jpeg', thresh)
# cv2.imwrite('close.jpeg', close)
# cv2.imwrite('result.jpeg', result)
# cv2.imshow('sharpen', sharpen)
# cv2.imshow('thresh', thresh)
# cv2.imshow('close', close)
# cv2.imshow('sharpen.jpeg', sharpen)
# cv2.waitKey(0)



# first crop the card out of the given image 
imageSrc = cv2.imread('sharpen.jpeg')

# First cut the source down slightly
h = imageSrc.shape[0]
w = imageSrc.shape[1]
cropInitial = 10
imageSrc = imageSrc[100:50+(h-cropInitial*2), 50:50+(w-cropInitial*2)]

# Threshold the image and find edges (to reduce the amount of pixels to count)
ret, imageDest = cv2.threshold(imageSrc, 180, 255, cv2.THRESH_BINARY_INV)
imageDest = cv2.Canny(imageDest, 100, 100, 3)

# Create a list of remaining pixels
points = cv2.findNonZero(imageDest)

# Calculate a bounding rectangle for these points
hull = cv2.convexHull(points)
x,y,w,h = cv2.boundingRect(hull)

# Crop the original image to the bounding rectangle
imageResult = imageSrc[y:y+h,x:x+w]
cv2.imwrite('cropped_image.jpeg', imageResult)
cv2.imshow('cropped_image.jpeg', imageResult)
cv2.waitKey(0)





# then applying the same algorithm on the found image to extract scratch code, barcode and date
date_regs = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
three_digit_scratch_code_regex = '^\d{3}$'
four_digit_scratch_code_regex = '^\d{4}$'

img = cv2.imread('test.jpeg')
# custom_config = r'--oem 3 --psm 8' # works with the cropped a sharpened image where each part is cropped on its own
# custom_config = r'--oem 3 --psm 6' # works with the whole image after cropping it using the upove code   
custom_config = r'--oem 3 --psm 12' # works with the whole image without cropping it if it's in the proper size

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, threshimg = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
dilation = cv2.dilate(threshimg, rect_kernel, iterations = 10)
img_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 5000:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

for cnt in img_contours:
    x, y, w, h = cv2.boundingRect(cnt)  
    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cropped_img = img[y:y + h, x:x + w] 
    text = pytesseract.image_to_data(cropped_img, output_type=Output.DICT, config=custom_config)

parse_text = []
scratch_code = ''
print(text['text'])
for word in text['text']:
    if re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word):
        print('from the append block scratch code', word)
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
