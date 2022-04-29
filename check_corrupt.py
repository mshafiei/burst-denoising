from os import listdir
from PIL import Image
import glob
import tqdm
dr = '/home/mohammad/Projects/optimizer/burst-denoising/data/challenge2018/train/*.jpg'
fns = glob.glob(dr)
for fn in tqdm.tqdm(fns):
    try:
        img = Image.open(fn) # open the image file
        img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
        print('Bad file:', fn) # print out the names of corrupt files