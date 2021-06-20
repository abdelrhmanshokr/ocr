from matplotlib import pyplot as plt
from pytesseract import Output
import cv2, pytesseract, re, json, imutils
import numpy as np

# reading an image
image = cv2.imread('test.jpeg')

# image preprocessing 
# image resizing 
# gray scale
gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# apply adaptive threshold
preprocessed_image = cv2.adaptiveThreshold(gray_scale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 3)

# configuring parameters for the tesseract with the image
custom_config = r'--oem 3 --psm 6'

# feeding the image to the tesseract 
details = pytesseract.image_to_data(image, output_type=Output.DICT, \
                        config=custom_config)

preprocessed_details = pytesseract.image_to_data(preprocessed_image, output_type=Output.DICT, \
                        config=custom_config)

date_regs = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
barcode_regex = '^\d{12}$'
three_digit_scratch_code_regex = '^\d{3}$'
four_digit_scratch_code_regex = '^\d{4}$'

print('details', details['text'])
print('pre processed details', preprocessed_details['text'])
total_boxes = len(details['text'])
total_adaptive_boxes = len(details['text'])
for sequence_number in range(total_boxes):
    if re.match(date_regs, details['text'][sequence_number]) or re.match(barcode_regex, details['text'][sequence_number]) or re.match(three_digit_scratch_code_regex, details['text'][sequence_number]) or re.match(four_digit_scratch_code_regex, details['text'][sequence_number]):
        print('from the boxes block image', details['text'][sequence_number], details['conf'][sequence_number])
        (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], \
                        details['width'][sequence_number],  details['height'][sequence_number])
        threshold_img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # if int(details['conf'][sequence_number]) >= int(preprocessed_details['conf'][sequence_number]):
    # else: 
    #     if re.match(date_regs, preprocessed_details['text'][sequence_number]) or re.match(barcode_regex, preprocessed_details['text'][sequence_number]) or re.match(three_digit_scratch_code_regex, preprocessed_details['text'][sequence_number]) or re.match(four_digit_scratch_code_regex, preprocessed_details['text'][sequence_number]):
    #         print('from the boxes block image', preprocessed_details['text'][sequence_number], preprocessed_details['conf'][sequence_number])
    #         (x, y, w, h) = (preprocessed_details['left'][sequence_number], preprocessed_details['top'][sequence_number], \
    #                         preprocessed_details['width'][sequence_number],  preprocessed_details['height'][sequence_number])
    #         threshold_img = cv2.rectangle(preprocessed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

parse_text = []
scratch_code = ''
for word in details['text']:
    if re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word):
        print('from the append block scrath code', word)
        scratch_code += word
    elif re.match(barcode_regex, word):
        print('from the append block barcode', word)
        parse_text.append({ 'barcode': word })
    elif re.match(date_regs, word):
        print('from the append block date', word)
        parse_text.append({ 'expiry date': word })
    # if int(details['conf'][details['text'].index(word)]) >= int(preprocessed_details['conf'][details['text'].index(word)]): 
    # else:
    #     print('preprocessed word is higher in conf', word)
    #     word = preprocessed_details['text'][details['text'].index(word)]
    #     if re.match(three_digit_scratch_code_regex, word) or re.match(four_digit_scratch_code_regex, word):
    #         print('from the append block scrath code', word)
    #         scratch_code += word
    #     elif re.match(barcode_regex, word):
    #         print('from the append block barcode', word)
    #         parse_text.append({ 'barcode': word })
    #     elif re.match(date_regs, word):
    #         print('from the append block date', word)
    #         parse_text.append({ 'expiry date': word })
parse_text.append({ 'scratch code': scratch_code })

with open('output.txt', 'w') as outfile:
    json.dump(parse_text, outfile)
        
print('parse text', parse_text)
plt.imshow(image)
plt.show()