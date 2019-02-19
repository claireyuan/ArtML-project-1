How to run: 

open up the terminal and type in the following line:

<i>python3 multiple_style_transfer.py</i>

put style and semantic images in the images folder which is in the same directory as multiple_style_transfer.py
paths to the style/semantic images are global variables in the script:

<i>content_file_name</i>: semantic file name<br>
<i>style_file_name_1</i>: name for the first style file<br>
<i>style_file_name_2</i>: name for the second style file<br>

results for the first round of style transfer, which only applys the first style image to the semantic input will be stored in the images/results/1 folder, and results for the second round of style transfer, which applys the second style image on the result of the first round will be stored in the images/results/2 folder.
