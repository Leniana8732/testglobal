import cv2 as cv
import numpy as np

# 定义找到并处理图像的函数
def findcor(img):
    # 3. 尋找鈔票的四個角落
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(gray_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 尋找近似的輪廓
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # 如果近似輪廓有四個角點，進行透視變換
        if len(approx) == 4:
            # 計算透視變換的目標座標
            target_corners = np.array([[0, 0], [0, 500], [500, 500], [500, 0]], dtype=np.float32)

            # 進行透視變換
            transform_matrix = cv.getPerspectiveTransform(approx.astype(np.float32), target_corners)
            warped_image = cv.warpPerspective(img, transform_matrix, (500, 500))
            print("return sus")
            return warped_image

# 1. 讀取圖片
image = cv.imread("test image//real2.jpg")

# 2. 调用处理图像的函数
warped_image = findcor(image)

# 4. 讀取1000top.png圖片
template = cv.imread("test image//1000tom.png")
# 5. 调用处理图像的函数
gray_template = findcor(template)
cv.imwrite("master.png",warped_image)
cv.imwrite("slave.png",gray_template)
# 如果 warped_image 和 gray_template 未找到，不执行后续操作
if warped_image is None or gray_template is None:
    exit("Failed to find corners and warp the images.")

# 5. 將first deal的圖片與參考圖片進行比對相似度
# 3. 使用ORB算法檢測特徵
orb = cv.ORB_create()
keypoints_image, descriptors_image = orb.detectAndCompute(warped_image, None)
keypoints_template, descriptors_template = orb.detectAndCompute(gray_template, None)

# 4. 使用FLANN算法進行特徵匹配
flann = cv.FlannBasedMatcher()
matches = flann.knnMatch(descriptors_image, descriptors_template, k=2)

# 5. 提取良好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 6. 顯示匹配結果
result_image = cv.drawMatches(image, keypoints_image, gray_template, keypoints_template, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("Match Result", result_image)

# 7. 顯示相似度
similarity = len(good_matches) / len(keypoints_template) * 100
print(f"Similarity: {similarity}%")
# 使用相似性比對算法，例如結構相似性(SSIM)或均方誤差(MSE)來計算相似度

# 6. 顯示處理階段的圖片
cv.imshow("Original Image", image)
cv.imshow("Warped Image", warped_image)
cv.imshow("Gray Template", gray_template)

cv.waitKey(0)
cv.destroyAllWindows()
