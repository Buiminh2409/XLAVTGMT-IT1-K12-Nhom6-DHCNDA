import cv2
import pandas as pd
import numpy as np

# Gán các giá trị cho webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Chiều rộng khung hình
cap.set(4, 1920)  # Chiều cao khung hình

# Đọc tệp CSV chứa màu
index = ["color", "color_name", "hex", "R", "G", "B"]
csv_path = "C:/Users/Minh/Downloads/ColorDetector-main/ColorDetector-main/colors.csv"
csv = pd.read_csv(csv_path, names=index, header=None)

colors = csv[["R", "G", "B"]].to_numpy()

# Tìm màu gần nhất từ tệp CSV sử dụng NumPy
# Tìm màu gần nhất từ tệp CSV sử dụng NumPy (hoán đổi BGR -> RGB)
def getColorName(b, g, r):
    if b > 200 and g > 200 and r > 200:
        return "WHITE"
    distances = np.abs(colors - np.array([r, g, b])).sum(axis=1)
    min_index = np.argmin(distances)
    return csv.loc[min_index, "color_name"]



# Làm sạch mask bằng các phép toán hình thái học
def applyMorphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Lọc các vùng nhỏ bằng Connected Components Analysis (CCA)
def filterSmallRegions(mask, min_area=500):
    # Thực hiện Connected Components Analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Tạo một mask rỗng
    filtered_mask = np.zeros_like(mask)

    for label in range(1, num_labels):  # Bỏ qua label 0 (background)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:  # Giữ lại vùng có diện tích lớn hơn ngưỡng
            filtered_mask[labels == label] = 255

    return filtered_mask

# Tạo mask dựa trên HSV và LAB để nhận diện chính xác hơn
def createMasks(hsv_img, lab_img):
    masks = {
        "RED": cv2.inRange(hsv_img, np.array([0, 120, 120]), np.array([10, 255, 255])),
        "GREEN": cv2.inRange(hsv_img, np.array([40, 100, 100]), np.array([80, 255, 255])),
        "BLUE": cv2.inRange(hsv_img, np.array([100, 120, 120]), np.array([140, 255, 255])),
        "ORANGE": cv2.inRange(hsv_img, np.array([10, 120, 120]), np.array([25, 255, 255])),
        "PINK": cv2.inRange(hsv_img, np.array([160, 100, 100]), np.array([170, 255, 255])),
        "BLACK": cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 50, 70])),
        "WHITE": cv2.inRange(hsv_img, np.array([0, 0, 180]), np.array([180, 50, 255])),
        "GRAY": cv2.inRange(lab_img, np.array([50, 120, 120]), np.array([200, 135, 135]))
    }
    return masks

# Phát hiện và vẽ viền, tâm các vùng màu
def detect_and_draw_contours(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            # Vẽ viền mỏng
            cv2.polylines(img, [contour], True, (0, 255, 255), 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                size = 5  # Kích thước vùng trung tâm
                h, w, _ = img.shape
                x1, y1 = max(cX - size, 0), max(cY - size, 0)
                x2, y2 = min(cX + size, w - 1), min(cY + size, h - 1)

                region = img[y1:y2, x1:x2]
                b, g, r = np.mean(region, axis=(0, 1))

                cname = getColorName(b, g, r)

                # Màu chữ theo danh sách đã định nghĩa
                text_colors = {
                    "red": (0, 0, 255),
                    "green": (0, 255, 0),
                    "blue": (255, 0, 0),
                    "yellow": (0, 255, 255),
                    "orange": (0, 165, 255),
                    "pink": (255, 20, 147),
                    "purple": (128, 0, 128),
                    "brown": (42, 42, 165),
                    "white": (255, 255, 255),
                    "gray": (128, 128, 128),
                    "black": (0, 0, 0)
                }

                # Lấy màu phù hợp cho văn bản
                text_color = text_colors.get(cname.lower(), (255, 255, 255))

                # Vẽ tâm màu trung bình lên hình ảnh
                cv2.circle(img, (cX, cY), 5, text_color, -1)

                # Hiển thị văn bản với viền đậm
                font_scale = 0.7
                thickness = 2

                # Vẽ viền chữ trước tiên để nổi bật
                cv2.putText(img, cname, (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(img, cname, (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

# Sử dụng thêm làm mịn mask để cải thiện nhận diện
def applyMorphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)  # Làm mịn mask
    return mask


# Lặp đọc webcam
while True:
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.flip(img, 1)  # Lật ảnh theo chiều ngang
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Xác định vùng trung tâm (ROI)
    height, width, _ = blurred_img.shape
    roi_x1, roi_y1 = width // 4, height // 4
    roi_x2, roi_y2 = 3 * width // 4, 3 * height // 4
    roi = blurred_img[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)

    masks = createMasks(hsv_img, lab_img)

    for mask_name, mask in masks.items():
        mask = applyMorphology(mask)
        mask = filterSmallRegions(mask, min_area=500)  # Lọc vùng nhỏ
        detect_and_draw_contours(roi, mask)

    # Vẽ khung chỉ với 4 góc
    corner_length = 80  # Độ dài của các đoạn thẳng ở góc

    # Góc trên bên trái
    cv2.line(blurred_img, (roi_x1, roi_y1), (roi_x1 + corner_length, roi_y1), (0, 255, 0), 2)
    cv2.line(blurred_img, (roi_x1, roi_y1), (roi_x1, roi_y1 + corner_length), (0, 255, 0), 2)

    # Góc trên bên phải
    cv2.line(blurred_img, (roi_x2, roi_y1), (roi_x2 - corner_length, roi_y1), (0, 255, 0), 2)
    cv2.line(blurred_img, (roi_x2, roi_y1), (roi_x2, roi_y1 + corner_length), (0, 255, 0), 2)

    # Góc dưới bên trái
    cv2.line(blurred_img, (roi_x1, roi_y2), (roi_x1 + corner_length, roi_y2), (0, 255, 0), 2)
    cv2.line(blurred_img, (roi_x1, roi_y2), (roi_x1, roi_y2 - corner_length), (0, 255, 0), 2)

    # Góc dưới bên phải
    cv2.line(blurred_img, (roi_x2, roi_y2), (roi_x2 - corner_length, roi_y2), (0, 255, 0), 2)
    cv2.line(blurred_img, (roi_x2, roi_y2), (roi_x2, roi_y2 - corner_length), (0, 255, 0), 2)

    cv2.imshow('Live Color Detection', blurred_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ OpenCV
cap.release()
cv2.destroyAllWindows()