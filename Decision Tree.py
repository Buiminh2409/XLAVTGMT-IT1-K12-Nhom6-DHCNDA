import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import cv2
from tkinter import Tk, filedialog, Button, Label, Toplevel, Frame, LEFT, RIGHT
from PIL import Image, ImageTk

# Load dữ liệu từ CSV
csv_path = "C:/Users/Minh/Downloads/sample_colors.csv"
csv = pd.read_csv(csv_path)

# Sử dụng các cột RGB làm input và cột color làm nhãn
colors = csv[['R', 'G', 'B']].to_numpy()  
labels = csv['color'].to_numpy() 

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(colors, labels, test_size=0.3, random_state=42)

# Huấn luyện mô hình K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Tính các chỉ số hiệu suất trên tập kiểm thử trước
y_pred_test = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')
report = classification_report(y_test, y_pred_test)

# Giao diện Tkinter chính
root = Tk()
root.title("Color Detection with KNN")

def select_and_process_image():
    file_path = filedialog.askopenfilename(title="Select an Image")
    if not file_path:
        return

    # Load ảnh từ máy tính
    image = cv2.imread(file_path)
    if image is None:
        return

    # Chuyển đổi ảnh sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    # Dự đoán màu trên từng pixel trong ảnh
    predicted_labels = knn.predict(pixels)

    # Dictionary ánh xạ màu với giá trị RGB
    label_to_rgb = {
        "Red": [255, 0, 0],
        "Green": [0, 255, 0],
        "Blue": [0, 0, 255],
        "Yellow": [255, 255, 0],
        "Purple": [128, 0, 128],
        "Orange": [255, 165, 0],
        "Pink": [255, 192, 203],
        "Brown": [165, 42, 42],
        "Gray": [128, 128, 128],
        "Cyan": [0, 255, 255],
        "Magenta": [255, 0, 255],
        "Olive": [128, 128, 0],
        "Maroon": [128, 0, 0],
        "Teal": [0, 128, 128],
        "Navy": [0, 0, 128],
        "Lime": [0, 255, 0],
        "Indigo": [75, 0, 130],
        "Gold": [255, 215, 0],
        "Silver": [192, 192, 192],
        "Beige": [245, 245, 220],
        "Peach": [255, 218, 185]
    }

    new_image_rgb = np.array([label_to_rgb[label] for label in predicted_labels], dtype=np.uint8).reshape(image_rgb.shape)

    # Tạo cửa sổ mới để hiển thị ảnh đã xử lý và kết quả
    top = Toplevel(root)
    top.title("Processed Image Analysis")

    left_frame = Frame(top)
    left_frame.pack(side=LEFT, padx=10, pady=10)

    right_frame = Frame(top)
    right_frame.pack(side=RIGHT, padx=10, pady=10)

    img = Image.fromarray(new_image_rgb)
    img.thumbnail((400, 400))

    img = ImageTk.PhotoImage(img)
    img_label = Label(left_frame, image=img)
    img_label.image = img
    img_label.pack()

    # Hiển thị các chỉ số hiệu suất trên tập kiểm thử
    info_text = (
        f"Precomputed Evaluation Metrics:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision (weighted): {precision:.2f}\n"
        f"Recall (weighted): {recall:.2f}\n"
        f"F1-Score (weighted): {f1:.2f}\n"
        f"\nClassification Report:\n{report}"
    )
    result_label = Label(right_frame, text=info_text, justify=LEFT, anchor="nw")
    result_label.pack()

# Nút để chọn ảnh và phân tích
Button(root, text="Analyze Image with KNN", command=select_and_process_image).pack()

# Chạy vòng lặp chính Tkinter
root.mainloop()
