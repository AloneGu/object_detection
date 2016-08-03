def get_place_id(video):
    """
        Extracts camera ID from video filenames.
    """
    import re

    m = re.search('\d\d\d\d\d\d\d', video)
    if m:
        return m.group()
    else:
        return None


def process_sizes(sizes_file):
    """
        Parses a sizes file and returns a dictionary of bin to size.
    """
    import sys

    try:
        guide_file = open(sizes_file, 'r')
    except IOError:
        print "No valid sizes file found"
        sys.exit()
    sizes = {}
    for line in guide_file:
        line = line.split()
        sizes[(int(line[0]), int(line[1]))] = int(line[2])
    return sizes


def generate_bg(video_file):
    import cv2
    import numpy

    cap = cv2.VideoCapture(video_file)
    # video_length is the total frames' number
    video_length = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    count, bg_cnt = 0, 0
    fgbg = cv2.BackgroundSubtractorMOG()
    bg = numpy.zeros((cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), dtype='int32')
    bg_pixel_map = numpy.zeros((cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), dtype='int32')
    print bg.shape
    while count < video_length - 1000:
        ret, frame = cap.read()
        count += 1
        fgmask = fgbg.apply(frame)
        bg_fgmask = ~fgmask  # background is 255, foreground is 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        curr_bg = frame & bg_fgmask
        # buffer one bg every 75 frames ( 5 seconds)
        if count % 75 == 0:
            print 'add one bg from frame', count
            bg += curr_bg
            bg_pixel_map += (bg_fgmask/255) # bg count for this pixel
            bg_cnt += 1
        if bg_cnt > 100:  # break if got 100 background images
            break
    cap.release()
    bg_new = numpy.uint8(bg / bg_pixel_map)
    bg_old = numpy.uint8(bg / bg_cnt)
    cv2.imshow('bg', bg_old)
    cv2.imshow('bg_2',bg_new)
    cv2.waitKey(0)
    # cv2.imwrite('~/Documents/bg.png',bg)
    return bg_new

# test function
if __name__ == '__main__':
    generate_bg('../videos/8600176_video-2015-09-14-19-00-02_1.3gp')




