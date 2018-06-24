from PIL import Image
import os
import glob
src = "./POKEMON"
dst = "./Data"

for each in glob.glob(src+'/*.png'):
    png = Image.open(each)
    # print each
    png.load() # required for png.split()
    background = Image.new("RGB", png.size, (255,255,255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    print(each,os.path.join(dst,each.split('/')[2] + '.jpg'))
    background.save(os.path.join(dst,each.split('/')[2][:4] + '.jpg'), 'JPEG')
