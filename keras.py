import cv2, re, json, pytesseract
from pytesseract import Output

date_regs = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
three_digit_scratch_code_regex = '^\d{3}$'
four_digit_scratch_code_regex = '^\d{4}$'

img = cv2.imread('test.jpeg')
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