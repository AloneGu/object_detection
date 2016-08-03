# Test Human Detection API

Author: Wei Gu

Please open this document with stackedit.

##Design:
Assume the python tool named human_detection_api.py
all the python files are under py_scripts.
###**1. Input**
**FLAG SETTING:**
####[1]. source file: you can only choose one mode to perform human detection
**-x,--xml:** the trained xml file.
**-feat,--feature:** hog/haar, what kind of feature used.
**--hog_config** : hog_config.json, the json file contains the setting for hog feature:
example ( also the default setting, if you do not set config file, it will use following setting for hog feature ):

    {
    "WINSIZE":"(96,96)",
    "BLOCKSIZE":"(16,16)",
    "BLOCKSTRIDE":"(8,8)",
    "CELLSIZE":"(8,8)",
    "NBINS":"9",
    "winStride" : "(8, 8)",
    "padding ": "(0, 0)",
    "scale" : "1.01",
    "finalThreshold" : "2"
    }

**--haar_config** : haar_config.json, the json file contains the setting for haar feature:
example ( also the default setting, if you do not set config file, it will use following setting for haar feature ):

    {
    "SCALEFACTOR":"1.1",
    "MINNEIGHBORS": "5",
    "MINSIZE":"(30, 30)",
    "MAXSIZE":"(96, 96)",
    "FLAGS":"CV_HAAR_SCALE_IMAGE"
    }


**-v, --video :** video file to perform human detection ( video mode, default = None )
**-img, --image:** image file to perform human detection ( image mode, default = None ), if the input file path is a folder, then perform human detection on all the image files under this folder. ( **NOTES: the image file name or image directory path should contains placement name like 8600154 !!!** )
**-b,--background:** background image, default = None, if you have set the background image, the api will use this image to do the background subtraction, otherwise it will generate a background first under video mode.

####**2. detection on frames **
--org_frame: True/False ：True means perform human detection on the original whole frame, default = True
--foreground: True/False: True means perform human detection on the foreground, default = False ( background image needed under image mode )
--fgmask: True/False: True means perform human detection on the fgmask frame, default = False


####**3. save result**
--output_dir: the directory to save the algo result

**video mode:**
suppose the video name is 8600141-2015-10-13-8600141_video_2015-10-13-14-15-38_1.3gp
currently just show the detection results on the video.

**image mode:**
suppose the image name is 8600141-2015-10-13-8600141_video_2015-10-13-14-15-38_1_2500.png

[1]. draw boxes of people and save the image as : output_dir/image_result/images/8600141-2015-10-13-8600141_video_2015-10-13-14-15-38_1_2500_(hog/haar)_(origin_frame/foreground/fgmask).png

suppose the image directory is 8600141_image

[2]. save the count result in a csv file ( if the input is a image directory ):
output_dir/image_result/csv_result/8600141_image_result.csv
format:
image name, human counts

####**4. Example usage:**

[1]. perform human detection on a image ( with/without background ):
haar feature without background subtraction:

    cd py_scripts
    python test_human_detection_api.py -x ../xmls/haarcascade_frontalface_default.xml -img ../test_images/face_test.png -f haar -o '/tmp/'

![enter image description here](https://lh3.googleusercontent.com/-cMbQJjwPDUg/VjiEmJ3zfjI/AAAAAAAAAMA/4kebhrBulW4/s0/123.JPG "123.JPG")

hog feature with background subtraction:

    cd py_scripts
     python test_human_detection_api.py -x ../xmls/stable_8600154.xml -img ../test_images/8600154_test/3050.png -f hog -b ../backgrounds/8600154_stable.png --org_frame False --foreground True -o '/tmp/'

![enter image description here](https://lh3.googleusercontent.com/-vMUKk-4r9GA/VjnTkKV3D_I/AAAAAAAAAN0/r8iEe0ZM8Ak/s0/newhog.JPG "newhog.JPG")
[2]. perform human detection on many images:

       cd py_scripts
       python test_human_detection_api.py -x ../xmls/stable_8600154.xml -img ../test_images/8600154_test/ --org_frame False --foreground True -b ../backgrounds/8600154_stable.png -o ~/Documents/tmp_test/


  you will see the result saved under ~/Documents/tmp_test/
  ![enter image description here](https://lh3.googleusercontent.com/-axBgVl2ihvg/VjnSBqEysVI/AAAAAAAAANA/1SWd1iAZgLU/s0/11112.JPG "11112.JPG")
  ![enter image description here](https://lh3.googleusercontent.com/-ydCKA2KP7Wo/VjnSGn-huSI/AAAAAAAAANQ/aoeT7839Uv0/s0/ddfs.JPG "ddfs.JPG")
  ![enter image description here](https://lh3.googleusercontent.com/-Crn2JdwuCHE/VjnSOzQVS-I/AAAAAAAAANc/QN3752hfzuU/s0/sdfsdf.JPG "sdfsdf.JPG")

[3]. perform human detection on video ( hog default setting, without input background, on foreground):

    cd py_scripts
    python test_human_detection_api.py -x ../xmls/stable_8600154.xml -v ../videos/8600154_es-2015-07-25-17-45-01_1_89919.3gp --org_frame False --foreground True
you will see a window shows the background generated, and a dynamic window shows the detection result.![enter image description here](https://lh3.googleusercontent.com/-y1w4RoUxaaw/VjnOuXurTwI/AAAAAAAAAMw/6Zr7qHAwOFQ/s0/dya.JPG "dya.JPG")



