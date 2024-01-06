import os
from tensorflow import _tf_uses_legacy_keras
#from main import prepare_image
import numpy as np
saved_model = _tf_uses_legacy_keras.models.load_model("model_best_98.h5")

path =  './Casia_database/Tp_Test/Tp_Test/'
B=[]
j=0
m=0
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        j=j+1
        if filename.endswith('jpg') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            X=(prepare_image(full_path))
            X = np.array(X)
            X= X.reshape(-1, 128, 128, 3)
            A=saved_model(X)
            if A[0][1] < A[0][0]:
                B=filename
                m=m+1
print(m)

path =  './Casia_database/Au_Test/Au_Test/'
B=[]
j=0
k=0
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        j=j+1
        if filename.endswith('jpg') or filename.endswith('bmp'):
            full_path = os.path.join(dirname, filename)
            X=(prepare_image(full_path))
            X = np.array(X)
            X= X.reshape(-1, 128, 128, 3)
            A=saved_model(X)
            if A[0][1] > A[0][0]:
                B=filename
                k=k+1
print(k)
