from PIL import Image
import os
# img = Image.open('image.png').convert('LA')
# img.save('greyscale.png')
dir = "/Users/nimaaghli/PycharmProjects/Pollen/dataset_samples/color"
for filename in os.listdir(dir):
    print(filename)
    #img = Image.open(dir+filename).convert('LA').convert('RGB')
    #img.save(dir+"gray_"+filename)

