# Copyright (c) 2016 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Definitions that don't fit elsewhere.

"""
import glob
import numpy

import cv2
import numpy as np

# Constants
import time

#SPACE_INDEX = 0
#FIRST_INDEX = ord('0') - 1  # 0 is reserved to space

SPACE_TOKEN = '<space>'

__all__ = (
    'DIGITS',
    'sigmoid',
    'softmax',
)

OUTPUT_SHAPE = (64, 256)

DIGITS = "0123456789"
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

f = open("chinese_3754.txt", "r")
c_list = None
for file_line in f:
    file_line = file_line.strip()
    c_list = unicode(file_line, 'utf8')

# Hyper-parameters
num_epochs = 1000000
num_hidden = 1024
num_layers = 2

DIGITS = c_list

CHARS = DIGITS
LENGTH = 16
LENGTHS = [16, 20] # the number of digits varies from LENGTHS[0] to LENGTHS[1] in a image
TEST_SIZE = 200
ADD_BLANK = True   # if add a blank between digits
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 4000 * num_epochs / 10


# parameters for bdlstm ctc
BATCH_SIZE = 256
BATCHES = 4000

#BATCH_SIZE = 64
#BATCHES = 10

TRAIN_SIZE = BATCH_SIZE * BATCHES

MOMENTUM = 0.9
REPORT_STEPS = BATCHES / 10


# Some configs
# Accounting the 0th indice +  space + blank label = 28 characters
# num_classes = ord('9') - ord('0') + 1 + 1 + 1
num_classes = len(DIGITS) + 1 + 1  # 10 digits + blank + ctc blank
print num_classes


def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]


def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))


"""
    {"dirname":{"fname":(im,code)}}
"""
data_set = {}


def load_data_set(dirname):
    fname_list = glob.glob(dirname + "/*.png")
    result = dict()
    for fname in sorted(fname_list):
        print "loading", fname
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = list(fname.split("/")[1].split("_")[1])
        index = fname.split("/")[1].split("_")[0]
        result[index] = (im, code)
    data_set[dirname] = result


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):
    start = time.time()
    fname_list = []
    if not data_set.has_key(dirname):
        load_data_set(dirname)

    if start_index is None:
        f_list = glob.glob(dirname + "/*.png")
        fname_list = [fname.split("/")[1].split("_")[0] for fname in f_list]

    else:
        for i in range(start_index, end_index):
            fname_index = "{:08d}".format(i)
            fname_list.append(fname_index)
    start = time.time()

    dir_data_set = data_set.get(dirname)
    for fname in sorted(fname_list):
        # im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        # code = list(fname.split("/")[1].split("_")[1])
        im, code = dir_data_set.get(fname)
        yield im, numpy.asarray([DIGITS.find(x)  for x in list(unicode("".join(code),  "utf8"))])
        # print("get time ", time.time() - start)


#def convert_original_code_train_code(code):
#    return numpy.asarray([SPACE_INDEX if x == SPACE_TOKEN else (ord(x) - FIRST_INDEX) for x in code])


def unzip(b):
    xs, ys = zip(*b)

    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


if __name__ == '__main__':
    train_inputs, train_codes = unzip(list(read_data_for_lstm_ctc("test"))[:2])
    print train_inputs.shape
    print train_codes
    print("train_codes", train_codes)
    targets = np.asarray(train_codes).flat[:]
    print targets
    print list(read_data_for_lstm_ctc("test", 0, 10))
