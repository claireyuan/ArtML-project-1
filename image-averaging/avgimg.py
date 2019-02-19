"""
First run style_transfer_keras.ipynb
Put this code with your results in the same directory
Run the code
It will produce an image with name "avgoutput.png" in your current directory
"""

import os, numpy, PIL
from PIL import Image

allfiles=os.listdir(os.getcwd())
imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]

avg=Image.open(imlist[0]).convert("RGB")

for i in range(len(imlist)):
    img=Image.open(imlist[i]).convert("RGB")
    avg=Image.blend(avg,img,1.0/float(i+1))
avg.save("avgoutput.png")
avg.show()







