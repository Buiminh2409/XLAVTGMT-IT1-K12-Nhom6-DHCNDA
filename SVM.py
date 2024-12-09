import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import cv2
from tkinter import Tk, filedialog, Button, Label, Toplevel, Frame, LEFT, RIGHT, Scale, HORIZONTAL
from PIL import Image, ImageTk

# Load dữ liệu từ CSV
csv_path = "C:/Users/Minh/Downloads/sample_colors.csv"
csv = pd.read_csv(csv_path)

# Sử dụng các cột RGB làm input và cột color làm nhãn
colors = csv[['R', 'G', 'B']].to_numpy()  # RGB data
labels = csv['color'].to_numpy()         # Tên màu tương ứng

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(colors, labels, test_size=0.3, random_state=42)

# Huấn luyện mô hình SVM
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

root = Tk()
root.title("Color Detection with SVM")

def select_and_process_image():
    file_path = filedialog.askopenfilename(title="Select an Image")
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    predicted_labels = svm.predict(pixels)

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

    y_pred_test = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    report = classification_report(y_test, y_pred_test)

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

    info_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}\n\n{report}"
    result_label = Label(right_frame, text=info_text, justify=LEFT, anchor="nw")
    result_label.pack()

Button(root, text="Analyze Image with SVM", command=select_and_process_image).pack()

root.mainloop()
