import numpy as np
import os
from PIL import Image
import sys
import tensorflow as tf

my_path = "C:/Users/Teodor\Documents/NBU/NETB380 Programming Practice/trainind_data_creation/Generated/test2/"

def printBMP(image):
    length = 32
    index = 0
    for pixel in image:
        if pixel == 0:
            print(" ", end='')
        else:
            print("O", end='')
        index += 1
        if index >= length:
            index = 0
            print('')


def getTop5Pred(predictions):
    with tf.Session() as sess:
        predictions = sess.run(tf.nn.softmax(predictions))[0]
        newPred = []
        i = '0'
        for x in predictions:
            newPred.append([i, x])
            if i == '9':
                i = 'a'
            elif i == 'z':
                i = 'A'
            else:
                i = chr(ord(i) + 1)
        newPred.sort(key=lambda x: x[1], reverse=True)
        newPred = newPred[:5]
        return newPred


def load(file_path = "", number = -1):
    images = []
    labels = []
    if file_path == "":
        file_path = my_path
    files = list(os.listdir(file_path))
    at = 0
    print("Loading data from", file_path, "...")
    if number != -1:
        print("Where the index of the symbol is", number)
    print(len(files), "files")
    print("0%  10%  20%  30%  40%  50%  60%  70%  80%  90%  100%")
    print("|----"*10 + '|')
    print('#', end='')
    sys.stdout.flush()
    for f in files:
        img = Image.open(file_path + f)
        img = list(img.getdata())
        index = ord(f[0]) - ord('0')
        if f[1] != '_':
            index = (ord(f[0]) - ord('0')) * 10 + ord(f[1]) - ord('0')
        if number == -1 or index == number:
            img = [0.5 if x > 0 else -0.5 for x in img]
            label = []
            for i in range(26+26+10):
                if i == index:
                    label.append(1)
                else:
                    label.append(0)
            images.append(img)
            labels.append(label)
        at += 1
        if ((100 * at) / len(files)) % 2 < 99 / len(files):
            # print((100 * at) / len(files), "% loaded")
            print('#', end='')
            sys.stdout.flush()
    print("\nLoaded!")
    return [images, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
