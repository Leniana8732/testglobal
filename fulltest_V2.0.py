import cv2 as cv
import numpy as np

def preprocess_image(image):
    # 將圖像轉換為灰度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 進行高斯模糊以降噪
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny邊緣檢測算法找到圖像中的邊緣
    edges = cv.Canny(blurred, 50, 150)

    return edges

def find_banknote_contour(image):
    # 找到輪廓
    contours, _ = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 對輪廓進行排序，取面積最大的輪廓（假設是鈔票）
    contour = max(contours, key=cv.contourArea)

    return contour

def four_point_transform(image, contour):
    # 獲取輪廓的近似多邊形
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)

    # 如果近似多邊形有四個頂點，進行透視變換
    if len(approx) == 4:
        # 對輪廓進行透視變換
        warped = cv.warpPerspective(image, cv.getPerspectiveTransform(approx.astype(np.float32), np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)), (500, 500))
        return warped
    else:
        print("Unable to perform perspective transform. Contour does not have 4 vertices.")
        return None

def detect_and_save_banknote():
    # 啟動攝像頭
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to capture frame.")
            break

        # 預處理圖像
        edges = preprocess_image(frame)

        # 找到鈔票的輪廓
        banknote_contour = find_banknote_contour(edges)

        if banknote_contour is not None:
            # 在原始圖像上繪製鈔票的輪廓
            cv.drawContours(frame, [banknote_contour], -1, (0, 255, 0), 2)

            # 對鈔票圖像進行透視變換
            warped_banknote = four_point_transform(frame, banknote_contour)

            if warped_banknote is not None:
                # 將處理後的紙鈔圖像保存為 PNG 文件
                cv.imwrite("detected_banknote.png", warped_banknote)

                # 顯示原始圖像和透視變換後的圖像
                cv.imshow("Original Image", frame)
                cv.imshow("Warped Banknote", warped_banknote)
                cv.waitKey(0)

                # 退出迴圈
                break

    # 釋放攝像頭
    cap.release()
    cv.destroyAllWindows()

    print("Banknote detected and saved as detected_banknote.png.")

# 執行紙鈔檢測和保存功能
detect_and_save_banknote()
