import cv2
import numpy as np
from matplotlib import pyplot as plt
import  os
from shutil import copyfile


def matchTemplate(template_dir, image_dir):
    img = cv2.imread(image_dir,0)
    img2 = img.copy()
    template = cv2.imread(template_dir,0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        #cv2.rectangle(img,top_left, bottom_right, 0, 3)
        #crop_img = img[top_left[1]:top_left[1] + w, top_left[0]:top_left[0]+ h]
        return top_left,w,h
        # plt.subplot(121),plt.imshow(img,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(crop_img,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)
        #
        # plt.show()
def autocrop(type, templatefile) :
    parent_dir = "/Volumes/ex_drive/Nima/" + type
    # matchTemplate("template.png" , "/Volumes/ex_drive/Nima/AM_amaranthus_hybridus/10/16.png")
    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            # directory name
            tmp_dir = subdir + "/" + dir
            # f varibles has the number of the files in the subdire(tmp_dir)
            p, d, f = next(os.walk(tmp_dir))
            midfile_index = len(f) / 2
            top_left, w, h = matchTemplate(templatefile, tmp_dir + "/" + str(midfile_index) + ".png")
            print (tmp_dir + " , " + str(len(f)))
            print ("midfile is = " + tmp_dir + "/" + str(midfile_index) + ".png")
            for filename in os.listdir(tmp_dir):
                imagedir = tmp_dir + "/" + filename
                print("load image from " + imagedir)
                img = cv2.imread(imagedir, 0)
                crop_img = img[top_left[1]:top_left[1] + w, top_left[0]:top_left[0] + h]
                cropdir = subdir + "_cropped/" + dir + "/"
                print("Check if directory exist : " + cropdir)
                if not os.path.exists(cropdir):
                    os.makedirs(cropdir)
                cropdirfile = subdir + "_cropped/" + dir + "/" + filename
                print("Saving cropped file to : " + cropdir)
                cv2.imwrite(cropdirfile, crop_img)

def sampleimages(type,samples):
    parent_dir = "/Volumes/ex_drive/Nima/" + type
    # matchTemplate("template.png" , "/Volumes/ex_drive/Nima/AM_amaranthus_hybridus/10/16.png")
    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            counter = 0
            # directory name
            tmp_dir = subdir + "_cropped" + "/" + dir
            # f varibles has the number of the files in the subdire(tmp_dir)
            p, d, f = next(os.walk(tmp_dir))
            filecount = len(f) - 1
            #files = np.linspace(0, filecount, samples)
            files = normaldisGenerator(filecount,samples)
            print ("sample these files" , files)
            #print ("sample these files: " , files)
            for filename in files:
                imagedir = tmp_dir + "/" + str(int(filename)) + ".png"
                savedir = subdir + "_" + str(samples) + "/" + dir + "/"
                print ("copy this file : " + imagedir )
                print ("check if path :" + savedir +" exists!")
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                savepath = savedir + str(int(counter)) + ".png"
                print ("to  this file : " + savepath)
                copyfile(imagedir, savepath)
                counter += 1
def normaldisGenerator(size , samplecount):
    trim = size / 4
    mid = size/2
    shalf = np.geomspace(mid, size - trim, num=samplecount/2, endpoint=False).astype(int)
    print(shalf)
    fhalf = np.geomspace(mid - 3, trim, num=samplecount/2, endpoint=False).astype(int)
    fhalf = fhalf[::-1]
    full = np.concatenate((fhalf,shalf))
    return full

def sampleImagesLinear(type,samples):
    parent_dir = "/Volumes/ex_drive/Nima/" + type
    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            counter = 0
            # directory name
            tmp_dir = subdir + "_cropped" + "/" + dir
            # f varibles has the number of the files in the subdire(tmp_dir)
            p, d, f = next(os.walk(tmp_dir))
            filecount = len(f) - 1
            files = np.linspace(0, filecount, samples)
            #files = normaldisGenerator(filecount,samples)
            print ("sample these files" , files)
            for filename in files:
                imagedir = tmp_dir + "/" + str(int(filename)) + ".png"
                savedir = subdir + "_" + str(samples) + "/" + dir + "/"
                print ("copy this file : " + imagedir )
                print ("check if path :" + savedir +" exists!")
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                savepath = savedir + str(int(counter)) + ".png"
                print ("to  this file : " + savepath)
                copyfile(imagedir, savepath)
                counter += 1



def main():
    type = ["AM_amaranthus_hybridus" , "AM_chenopodum_album" , "AM_Iresine_diffusa" , "AM_chenopodium_ambrosioides"]
    templatefile = "template_iresine.png"
    for types in type:
        #sampleimages(types, 50)
        sampleImagesLinear(types, 50)


if __name__ == '__main__':
    main()
