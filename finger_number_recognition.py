import cv2
import numpy as np
from matplotlib import pyplot as plt

video_capture = cv2.VideoCapture(0)

# template1 = cv2.imread('1.jpg', 0)
# template1 = cv2.resize(template1, (320, 240))

template1 = cv2.resize(cv2.imread('1.jpg', 0), (200, 200))
template2 = cv2.resize(cv2.imread('2.jpg', 0), (200, 200))
template3 = cv2.resize(cv2.imread('3.jpg', 0), (200, 200))
template4 = cv2.resize(cv2.imread('4.jpg', 0), (200, 200))
template5 = cv2.resize(cv2.imread('5.jpg', 0), (200, 200))

w1, h1 = template1.shape[::-1]


# cv2.imshow('1', template1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# template2 = cv2.imread('2.jpg', 0)
# template2 = cv2.resize(template2, (320, 240))
w2, h2 = template2.shape[::-1]

# template3 = cv2.imread('3.jpg', 0)
# template3 = cv2.resize(template3, (320, 240))
w3, h3 = template3.shape[::-1]
w4, h4 = template4.shape[::-1]
w5, h5 = template5.shape[::-1]

while True:
    ret, image = video_capture.read()
    image = cv2.resize(image, (640, 480))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res1 = cv2.matchTemplate(grayscale_image, template1, cv2.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    res2 = cv2.matchTemplate(grayscale_image, template2, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    res3 = cv2.matchTemplate(grayscale_image, template3, cv2.TM_CCOEFF_NORMED)
    min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
    res4 = cv2.matchTemplate(grayscale_image, template4, cv2.TM_CCOEFF_NORMED)
    min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)
    res5 = cv2.matchTemplate(grayscale_image, template5, cv2.TM_CCOEFF_NORMED)
    min_val5, max_val5, min_loc5, max_loc5 = cv2.minMaxLoc(res5)

    template_list = {
        '1' : max_val1,
        '2' : max_val2,
        '3' : max_val3,
        '4' : max_val4,
        '5' : max_val5
    }

    correct_num = max(template_list.items(), key = lambda i : i[1])
    correct_num_finger = correct_num[0]
    correct_num_val = correct_num[1]

    print(correct_num_finger)
    print(correct_num_val)
    print()

    if correct_num_finger == '1' and correct_num_val > 0.5:
        print('masuk ke 1')
        grayscale_image = cv2.rectangle(grayscale_image, max_loc1, (max_loc1[0] + w1, max_loc1[1] + h1), 255, 2)
        cv2.putText(grayscale_image, 'One', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
    elif correct_num_finger == '2' and correct_num_val > 0.5:
        print('masuk ke 2')
        grayscale_image = cv2.rectangle(grayscale_image, max_loc2, (max_loc2[0] + w2, max_loc2[1] + h2), 255, 2)
        cv2.putText(grayscale_image, 'Two', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
    elif correct_num_finger == '3' and correct_num_val > 0.5:
        print('masuk ke 3')
        grayscale_image = cv2.rectangle(grayscale_image, max_loc3, (max_loc3[0] + w3, max_loc3[1] + h3), 255, 2)
        cv2.putText(grayscale_image, 'Three', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

    elif correct_num_finger == '4' and correct_num_val > 0.5:
        print('masuk ke 4')
        grayscale_image = cv2.rectangle(grayscale_image, max_loc4, (max_loc4[0] + w4, max_loc4[1] + h4), 255, 2)
        cv2.putText(grayscale_image, 'Four', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

    elif correct_num_finger == '5' and correct_num_val > 0.5:
        print('masuk ke 5')
        grayscale_image = cv2.rectangle(grayscale_image, max_loc5, (max_loc5[0] + w5, max_loc5[1] + h5), 255, 2)
        cv2.putText(grayscale_image, 'Five', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Processed Video', grayscale_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()