import cv2 as cv
import numpy as np
import time 


denominations = [100, 100, 500, 500, 1000, 1000]

##camera on then save image 
def opencamera(): #write name "captured_image.png"
    cap =cv.VideoCapture(0) # if you only have 2 camera  set (0) to (1)
    if not cap.isOpened():
        print(f'None camera detect plz cheak your camera not been using on other side')
        return

    ret, frame = cap.read()
    # 检查图像读取是否成功
    if not ret:
        print("Failed to capture image....")
        return
    x, y, w, h = 150, 150, 400, 200         # 调整检测位置
    cropped_frame = frame[y:y+h, x:x+w]
    cv.imshow("frame",frame)
    cv.imshow("cap",cropped_frame) 
    cv.waitKey(200)
    cv.destroyAllWindows()   
    time.sleep(0.5)
    print("Delay 0.5s ...")
    cv.imwrite("captured_image.png", cropped_frame)

    
    cap.release()
    print("Image captured successfully....")
    time.sleep(0.5)
    print("Waiting for resize image......")


resize_w = 500 
resize_h = 500

def find_banknote_contour(image):
    for _ in range(10):  # 尝试10次
        print(f'Retry {_}times wait for 1s....')
        time.sleep(1)
        try:
            opencamera()  # find_banknote_contour() 找不到範圍會retry開啟camera再讀一次
            
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gray_gluss = cv.GaussianBlur(gray,(5,5),0)
            time.sleep(0.1)
            cv.imshow("cheak",gray_gluss)
            _, edges = cv.threshold(gray_gluss,100,255,cv.THRESH_BINARY)
            
            # 找轮廓
            contours,_= cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                (x,y,w,h) =cv.boundingRect(cnt)
                area = cv.contourArea(cnt)
                if area > 12000 and area > 7000 :
                    roi = image[y+10:y+h-10, x+10:x+w-10]
                    cv.imwrite("ROI_image.png", roi)
                    print(f'ROI suscess')
                return roi
        except Exception as e:
            print(f"Error: {e}")

    print("Failed to find banknote contour after 10 attempts.")
    return None
        

def match_template(input_image, templates):
    # 初始化ORB特征检测器
    orb = cv.ORB_create()

    # 检测输入图像的特征和描述符
    kp_input, desc_input = orb.detectAndCompute(input_image, None)

    best_match_template_index = None
    best_match_score = -1

    # 对每个模板进行匹配
    for i, template in enumerate(templates):
        # 检测模板的特征和描述符
        kp_template, desc_template = orb.detectAndCompute(template, None)

        # 使用暴力匹配器计算特征向量之间的相似度
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_template, desc_input)

        # 计算匹配得分
        score = sum([match.distance for match in matches]) / len(matches)

        # 更新最佳匹配模板和匹配得分
        if best_match_score == -1 or score < best_match_score:
            best_match_template_index = i
            best_match_score = score
        denomination = denominations[i] if i < len(denominations) else "Unknown"
        print(f'Denomination: {denomination}, Score: {score}')

    return best_match_template_index

#第一次啟動
opencamera()
image = cv.imread("captured_image.png")
roi = find_banknote_contour(image) #roi是抓取範圍後將範圍內的圖片

if roi is not None:
    ## 调整图像大小
    input_image = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    input_image_resized = cv.resize(input_image, (resize_h, resize_w))
    cv.imwrite("ROI_image_resize.png",input_image_resized)
    # 调整模板图像的大小
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

    templates = [template100top_resized, template100bot_resized, template500top_resized, template500bot_resized, template1000top_resized, template1000bot_resized]

    best_template_index = match_template(input_image_resized, templates)

    # 判断最佳值
    if best_template_index is not None:
        recognized_denomination = denominations[best_template_index]
        print(f"Recognized denomination: {recognized_denomination}")
    else:
        print("Failed to find best template.")
else:
    print("Failed to find banknote contour.")
