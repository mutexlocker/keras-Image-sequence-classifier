import cv2
import numpy as np
import os
from os.path import isfile, join


def convert_frames_to_video(fileno , pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    print (files)
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[:-4]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    print("save to", pathOut + fileno + ".avi")
    out = cv2.VideoWriter(pathOut + fileno + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    fps = 5.0
    path_to_folder = "/Volumes/ex_drive/Nima/AM_amaranthus_hybridus_cropped/"
    pathIn = path_to_folder
    pathOut = path_to_folder[:-1] + "_vid/"
    for subdirs,dirs,files in os.walk(pathIn):
        for dir in dirs:
            pathIn = subdirs + dir + "/"
            print (pathIn)
            convert_frames_to_video(dir,pathIn, pathOut, fps)


if __name__ == "__main__":
    main()