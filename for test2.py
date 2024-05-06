import cv2 as cv
import numpy as np
import time 

## 定义摄像头开启函数
def opencamera(): 
    cap = cv.VideoCapture(0) 
    if not cap.isOpened():
        print(f'No camera detected. Please check if your camera is being used by other applications.')
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        return

    x, y, w, h = 100, 100, 450, 200 # 裁剪检测位置
    cropped_frame = frame[y:y+h, x:x+w]
    #cv.imshow("frame", frame)
    #cv.imshow("cap", cropped_frame)    
    time.sleep(0.5)
    print("Delaying for 0.5s ...")
    cv.imwrite("captured_image.png", cropped_frame)

    cap.release()
    print("Image captured successfully.")
    time.sleep(0.5)
    print("Waiting for resizing image...")

## 定义图像裁剪函数
def find_banknote_contour(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_gluss = cv.GaussianBlur(gray, (5,5), 0)
    time.sleep(0.1)
    #cv.imshow("check", gray_gluss)
    _, edges = cv.threshold(gray_gluss, 100, 255, cv.THRESH_BINARY)
    #cv.imshow("edges", edges)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x,y,w,h) = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        if area > 12000 and area > 7000 :
            roi = image[y+10:y+h-10, x+10:x+w-10]
            cv.imwrite("ROI_image.png", roi)
            return roi
    # 如果未找到有效区域，返回None
    return None

## 定义模板匹配函数
def match_template(input_image, templates):
    orb = cv.ORB_create()
    kp_input, desc_input = orb.detectAndCompute(input_image, None)

    best_match_template = None
    best_match_score = -1
    best_template_index = None

    for i, template in enumerate(templates):
        kp_template, desc_template = orb.detectAndCompute(template, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_template, desc_input)
        score = sum([match.distance for match in matches]) / len(matches)

        if best_match_score == -1 or score < best_match_score:
            best_match_template = template
            best_match_score = score
            best_template_index = i

    return best_template_index

start_time = time.time()
## 打开摄像头并拍照
opencamera()
image = cv.imread("captured_image.png")
roi = find_banknote_contour(image)

if roi is not None:
    ## 调整图像大小
    resize_w = 500 
    resize_h = 500
    input_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    input_image_resized = cv.resize(input_image, (resize_h, resize_w))
    cv.imwrite("ROI_image_resize.png", input_image_resized)

    ## 调整模板图像大小
    template100top = cv.imread("100top.jpg", cv.IMREAD_GRAYSCALE)
    template100top_resized = cv.resize(template100top, (resize_h, resize_w))

    template100bot = cv.imread("100bot.jpg", cv.IMREAD_GRAYSCALE)
    template100bot_resized = cv.resize(template100bot, (resize_h, resize_w))

    template500top = cv.imread("500top.jpg", cv.IMREAD_GRAYSCALE)
    template500top_resized = cv.resize(template500top, (resize_h, resize_w))

    template500bot = cv.imread("500bot.jpg", cv.IMREAD_GRAYSCALE)
    template500bot_resized = cv.resize(template500bot, (resize_h, resize_w))

    template1000top = cv.imread("1000top.png", cv.IMREAD_GRAYSCALE)
    template1000top_resized = cv.resize(template1000top, (resize_h, resize_w))

    template1000bot = cv.imread("1000bot.png", cv.IMREAD_GRAYSCALE)
    template1000bot_resized = cv.resize(template1000bot, (resize_h, resize_w))

    ## 进行模板匹配
    templates = [template100top_resized, template100bot_resized, template500top_resized, template500bot_resized, template1000top_resized, template1000bot_resized]
    best_template_index = match_template(input_image_resized, templates)

    ## 判斷最佳模板並輸出結果
    if best_template_index is not None:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total time: {execution_time} s")
        denominations = [100, 100, 500, 500, 1000, 1000]
        recognized_denomination = denominations[best_template_index]
        print(f"面額為: {recognized_denomination}元")
        cv.imshow("Best Template", templates[best_template_index])
        #cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Failed to find best template.")
else:
    print("Failed to find banknote contour.")
