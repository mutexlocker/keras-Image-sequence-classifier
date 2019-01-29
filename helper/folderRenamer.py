import os
import shutil
parent_dir = '/Volumes/ex_drive/Nima/AM_Iresine_diffusa'
def renamefolders() :
    counter = 0
    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            counter_files = 0
            tmp_dir = subdir + "/" + dir
            os.rename(tmp_dir,subdir + "/" +str(counter))
            counter += 1
            print (tmp_dir)


def renamefiles() :
    for subdir, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            counter_files = 0
            tmp_dir = subdir + "/" + dir
            print (tmp_dir)
            for filename in os.listdir(tmp_dir):
                tmpfile = tmp_dir + "/" + filename
                print(tmpfile)
                os.rename(tmpfile, tmp_dir + "/" + str(counter_files)+".png")
                counter_files += 1



def main():
    renamefolders()
    renamefiles()


if __name__ == '__main__':
    main()
