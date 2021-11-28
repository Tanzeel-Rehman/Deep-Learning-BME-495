import os
import shutil
import sys

filenames = os.listdir('./')
zipfile = None
for i in range(len(filenames)):
    filename, extension = os.path.splitext(filenames[i])
    if extension == '.zip' or extension == '.gz' or extension == '.rar':
        zipfilename = filenames[i]
        break

import zipfile
z = zipfile.ZipFile(zipfilename, 'r')
z.extractall('./HW2')

sys.path.append('./HW2')

import HW2
from HW2.neural_network import NeuralNetwork
import numpy as np
import HW2.logic_gates as logic_gates


def main():

    AND = logic_gates.AND()
    XOR = logic_gates.XOR()

    x1 = [False, False, True, True]
    x2 = [False, True, False, True]
    y = [False, False, False, True]
    and_c = 0
    for i in range(4):
        and_c = and_c+1 if AND(x1[i], x2[i]) == y[i] else 0

    if and_c == 4:
        print("Passed AND!")
    else:
        print("Failed AND!")

    y = [False, True, True, False]
    xor_c = 0
    for i in range(4):
        xor_c = xor_c+1 if XOR(x1[i], x2[i]) == y[i] else 0

    if xor_c == 4:
        print("Passed XOR!")
    else:
        print("Failed XOR!")


if __name__ == "__main__":
    main()
    shutil.rmtree('HW2')
    os.remove(zipfilename)
