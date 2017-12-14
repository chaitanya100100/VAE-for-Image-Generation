import os
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cPickle

inp_dir = '101_ObjectCategories'
target_size = (128, 128)

classes = os.listdir(inp_dir)
all_images = []
all_labels = []

i = 0
for idx, c in enumerate(classes):
    img_list = os.listdir(inp_dir + '/' + c)
    print idx
    j = 0
    for img in img_list:
        fname = inp_dir + '/' + c + '/' + img
        image = image_utils.load_img(fname).resize(target_size,Image.ANTIALIAS)
        image = np.array(image.getdata()).reshape(target_size[0], target_size[1], 3)
        image = image.astype('float32')/255
        all_images.append(image)
        all_labels.append(idx)
        #j += 1
        #if j >= 20:
        #    break
        #plt.imshow(image)
        #plt.show()
    #i += 1
    #if i >= 50:
    #    break


all_images = np.array(all_images)
all_labels = np.array(all_labels)

print all_images.shape
print all_labels.shape

np.save('full_x', all_images)
np.save('full_y', all_labels)
