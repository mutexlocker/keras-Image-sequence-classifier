import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
import os


def show_images(images, cols=1,rows = 5, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, rows, n + 1)
        #if image.ndim == 2:
            #plt.gray()
        plt.imshow(image)
        a.set_yticklabels([])
        a.set_xticklabels([])
        #a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def findcountor(img):
    contours = measure.find_contours(img, .8)
    return contours
def showSpecefictypes(type):
    thislist = []
    for filename in os.listdir(dir):
        if type in filename:
            img = Image.open(dir+filename)
            counters = findcountor(img)
            plt.plot(counters[::, 1], counters[::, 0], linewidth=0.5)
            thislist.append(img)

    show_images(thislist,1,5)

def normaldisGenerator(size):
    mid = size/2
    shalf = np.geomspace(mid, size -5, num=5, endpoint=False).astype(int)
    print(shalf)
    fhalf = np.geomspace(mid - 3, 5, num=5, endpoint=False).astype(int)
    fhalf = fhalf[::-1]
    full = np.concatenate((fhalf,shalf))
    print(full)

dir = "/Volumes/ex_drive/Nima/AM_Iresine_diffusa_cropped/21/"

#showSpecefictypes("wall")

list = np.arange(50)
normaldisGenerator(len(list))

