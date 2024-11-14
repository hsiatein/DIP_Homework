import face_alignment
from skimage import io
from PIL import Image
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,device='mps')

input = io.imread('1.jpg')
preds = fa.get_landmarks(input)
for p in preds[0]:
    print(p)
    for i in range(-3,4):
        for j in range(-3,4):
            input[int(p[1])+i,int(p[0])+j,:]=np.array([255,0,0])
Image.fromarray(input)
