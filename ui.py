import os
import sys
import time

import keras.models
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QFileDialog, QWidget, \
    QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
import PyQt5.QtCore
saved_model = keras.models.load_model('model_casia_best.h5')
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# %%

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 98).resize([128, 128])).flatten() / 255.0

class ImageSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.imagePath = None
        self.initUI()

    def initUI(self):
        # Tạo nút chọn ảnh
        self.title = QLabel(self)
        self.title.setText("Chương trình dự đoán ảnh")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        new_font = QFont("Times New Roman", 20, QFont.Bold)
        self.title.setFont(new_font)
        self.selectImageButton = QPushButton('Chọn ảnh', self)
        self.selectImageButton.clicked.connect(self.showDialog)
        self.selectImageButton.setFixedSize(100,50)
        self.predictButton = QPushButton('Dự đoán ảnh', self)
        self.predictButton.clicked.connect(self.predictImage)
        self.predictButton.setFixedSize(100, 50)
        # Hiển thị ảnh được chọn
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setFixedSize(300, 300)
        self.textLabel = QLabel(self)
        self.textLabel.setAlignment(QtCore.Qt.AlignCenter)

        # Tạo layout
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        image_layout.addStretch(1)
        image_layout.addWidget(self.imageLabel)
        image_layout.addStretch(1)
        image_layout.addWidget(self.textLabel)
        image_layout.addStretch(1)
        # Tạo layout ngang để đặt nút về bên phải
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.selectImageButton)
        button_layout.addWidget(self.predictButton)
        button_layout.addStretch(1)

        # Thêm các widget và layout con vào layout chính
        main_layout.addStretch(1)
        main_layout.addWidget(self.title)
        main_layout.addStretch(1)
        main_layout.addLayout(image_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(button_layout)
        # Tạo widget chính và thiết lập layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Chọn ảnh từ máy tính')

    def showDialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh', '', 'Ảnh (*.png *.jpg *.bmp *.jpeg *.tif)')

        if fname:
            self.imagePath = fname
            pixmap = QPixmap(fname)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)
    def predictImage(self):
        start_time = time.time()
        X = (prepare_image(self.imagePath))
        X = np.array(X)
        X = X.reshape(-1, 128, 128, 3)
        A = saved_model(X)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if A[0][1] < A[0][0]:
            self.textLabel.setText(f"Ảnh thật, thời gian dự đoán {elapsed_time:.4f}")
        else:
            self.textLabel.setText(f"Ảnh giả mạo, thời gian dự đoán {elapsed_time:.4f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageSelectorApp()
    ex.show()
    sys.exit(app.exec_())
