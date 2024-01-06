import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2) #Giá trị seed giúp cho mỗi lần chạy đều cho ra cùng giá trị random

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from PIL import Image, ImageChops, ImageEnhance #Thư viện PIL xử lý ảnh
import os
import itertools

def convert_to_ela_image(path, quality):
  temp_filename = 'temp_file_name.jpg'
  ela_filename = 'temp_ela.png'

  image = Image.open(path).convert('RGB') #Chuyển đổi không gian màu thành RGB
  image.save(temp_filename, 'JPEG', quality= quality) #Lưu hình ảnh dưới dạng tệp JPEG với đường dẫn tệp là temp_filename
  temp_image = Image.open(temp_filename) #Mở hình ảnh từ tệp temp_filename

  ela_image = ImageChops.difference(image, temp_image) #Tính toán sự khác nhau giữa image và temp_image

  extrema = ela_image.getextrema() #Lấy ra giá trị cực đại và cực tiểu của pixel trong hình ảnh ela_image
  max_diff = max([ex[1] for ex in extrema])
  if max_diff == 0:
    max_diff = 1
  scale = 255.0/ max_diff

  ela_image = ImageEnhance.Brightness(ela_image).enhance(scale) #Điều chỉnh độ sáng cho hình ảnh

  return ela_image

def prepare_image(image_path):
  return np.array(convert_to_ela_image(image_path,9).resize([128,128])).flatten() / 255.0
  #Trả về vector mảng 1 chiều có các pixel được chuẩn hóa về khoảng từ 0 -> 1
  #Hàm flatten() sử dụng đề biến mảng nhiều chiều thành mảng 1 chiều

X_train = []
X_test = []
Y_train = [] #0 là ảnh giả, 1 là ảnh thật
Y_test = [] #0 là ảnh giả, 1 là ảnh thật

import random
path = './Casia_database/Au_Train/Au_Train/'
j = 0
for dirname, _, filenames in os.walk(path): #walk(path) trả về các thư mục và tệp trong đường dẫn path
  for filename in filenames:

    if filename.endswith('jpg') or filename.endswith('bmp'):
      j = j + 1
      full_path = os.path.join(dirname, filename) #Tạo ra đường dẫn đầy đủ của tệp filename trong thư mục dirname
      X_train = np.append(X_train, prepare_image(full_path))
      Y_train = np.append(Y_train, 0)
      if len(Y_train) % 500 == 0:
        print(f'Processing {len(Y_train)} images')
        print(j)

path = './Casia_database/Tp_Train/Tp_Train/'
for dirname, _, filenames in os.walk(path):
  for filename in filenames:
    j = j+1
    if filename.endswith('jpg') or filename.endswith('tif'):
      full_path = os.path.join(dirname, filename)
      X_train = np.append(X_train, prepare_image(full_path))
      Y_train = np.append(Y_train, 1)
      if len(Y_train) % 500 == 0:
        print(f'Processing {len(Y_train)} images')
        print(j)

#random.shuffle(X)
path = './Casia_database/Au_Test/Au_Test/'
for dirname, _, filenames in os.walk(path):
  for filename in filenames:
    j=j+1;
    if filename.endswith('jpg') or filename.endswith('bmp'):
      full_path = os.path.join(dirname, filename)
      X_test = np.append(X_test, prepare_image(full_path))
      Y_test = np.append(Y_test, 0)
      if len(Y_test) % 500 == 0:
        print(f'Processing {len(Y_test)} images')
        print(j)

path = './Casia_database/Tp_Test/Tp_Test/'
for dirname, _, filenames in os.walk(path):
  for filename in filenames:
    j=j+1;
    if filename.endswith('jpg') or filename.endswith('tif'):
      full_path = os.path.join(dirname, filename)
      X_test = np.append(X_test, prepare_image(full_path))
      Y_test = np.append(Y_test, 1)
      if len(Y_test) % 500 == 0:
        print(f'Processing {len(Y_test)} images')
        print(j)


#%%

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = to_categorical(Y_train, 2) #Biến đổi Y_train về dạng one-hot Y_train chứa các nhãn, 2 là số lớp
#One-hot coding, giả sử có K lớp dữ liệu, giả sử dữ liệu thuộc lớp n (n < K) sẽ được biểu diễn bằng cách sử dụng
#một vector y có chiều 1 * K sao cho tất cả phần tử của vector = 0, ngoại trừ phần tử thứ n của vector y = 1

Y_test = to_categorical(Y_test, 2)
X_train = X_train.reshape(-1, 128, 128, 3)
X_test = X_test.reshape(-1, 128, 128, 3)
#%%

#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
#X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_test), len(Y_test))


def build_model():
  model = Sequential()  # Xây dựng mô hình theo kiểu tuyến tính trong Keras (các lớp chồng lên nhau)
  model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(128, 128,
                                                                                                    3)))  # Thêm lớp Convolutional vào mô hình, sử dụng 32 bộ lọc và mỗi bộ lọc có kích thước 3 x 3, đầu vào là hình ảnh có kích thước 128 x 128 pixel
  model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(128, 128, 3)))

  model.add(Conv2D(filters=32, kernel_size=(7, 7), padding='valid', activation='relu', input_shape=(128, 128, 3)))

  model.add(MaxPool2D(pool_size=(2, 2)))  # sử dụng kỹ thuật max-pooling với pooling size là 2 x 2

  model.add(Dropout(0.25))  # Kỹ thuật regularization để ngăn chặn overfitting
  model.add(Flatten())  # Biểu diễn vector đa chiều thành vector 1 chiều
  model.add(Dense(256,
                  activation='relu'))  # Mỗi đơn vị(unit) của lớp này sẽ kết nối tới tất cả đơn vị của lớp trước đó, 256 là số trọng số sẽ dc học
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))
  return model


from keras.optimizers import Adam
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping

# %%
model = build_model()
model.summary()
epochs =  5 #100  # Mỗi lần duyệt một lượt qua tất cả các điểm trên toàn bộ dữ liệu được gọi là một epoch
batch_size = 64
init_lr = 1e-5
optimizer = Adam(lr=init_lr)  # lr là tốc độ học trong GD, decay là cách giảm tốc độ học sau mỗi epoch
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# %%


checkpoint = ModelCheckpoint("model_best_98.h5", monitor='val_accuracy', verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto') #Dùng để lưu lại checkpoint nếu như mô hình ở sau tốt hơn mô hình ở trước (save_best_only = True)
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=300, verbose=0, mode='auto')
# %%
hist = model.fit(X_train,
                 Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_test, Y_test),
                 callbacks=[checkpoint, early_stopping])

# %%
model.save('model_casia_best.h5')
# %%
