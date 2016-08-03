# copy from backend/videomturk / videolabeler_client / lib / videolabeler_client / video_previewer.py

from numpy.linalg import inv

import copy
import cv
import cv2
import json
import logging
import math
import numpy

logger = logging.getLogger(__name__)


def on_mouse(event, x, y, flag, move_event):
    if event == cv.CV_EVENT_MOUSEMOVE:
        if move_event.is_move:
            move_event.point.x = x
            move_event.point.y = y
    elif event == cv.CV_EVENT_LBUTTONDOWN:
        if not move_event.is_move:
            move_event.edge = -1
            move_event.is_move = True
            move_event.point.x = x
            move_event.point.y = y
    elif event == cv.CV_EVENT_LBUTTONUP:
        move_event.is_move = False


class Size(object):
    width = None
    height = None

    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height


class Point(object):
    x = None
    y = None

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def to_tuple(self, method=int):
        return (method(self.x), method(self.y))


class Roi(Point, Size):
    pass


class VideopUnwarper(object):
    def __init__(self, config):
        self._config = config
        if self._config.has_key('unwarp'):
            self._config = self._config['unwarp']
        self._init()

    def _init(self):
        self._rect_f = self._config["rectF"]
        self._distort_f = self._config["distortF"]
        self._resolution_factor = self._config["resolutionFactor"]
        self._use_bilinear = self._config["useBilinear"]
        self._centeroffset = Point()
        self._centeroffset.x = self._config["xcenteroffset"]
        self._centeroffset.y = self._config["ycenteroffset"]
        self._roi = Roi()
        self._roi.x = self._config["roiX"]
        self._roi.y = self._config["roiY"]
        self._roi.width = self._config["roiW"]
        self._roi.height = self._config["roiH"]
        self._rotation = self._config["rotation"] * math.pi / 180.0
        self._xmap = None
        self._ymap = None

    def _distance(self, p1, p2):
        dist_in_square = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
        return math.sqrt(dist_in_square)

    def _get_rotation_matrix(self, rotation_radian):
        data = [
            math.cos(rotation_radian), -math.sin(rotation_radian),
            math.sin(rotation_radian), math.cos(rotation_radian)
        ]
        rotation_matrix = numpy.array(data, dtype=numpy.float32).reshape(2, 2)
        return rotation_matrix

    def _get_affine_transformation(self, image_size, rotation_radian, scaling_factor):
        w = image_size.width
        h = image_size.height
        r_theta = self._get_rotation_matrix(rotation_radian)
        scaling_mat = numpy.zeros((2, 2), dtype=numpy.float32)
        scaling_mat[0, 0] = scaling_factor
        scaling_mat[1, 1] = scaling_factor
        a = numpy.dot(r_theta, scaling_mat)

        image_center = numpy.zeros((2, 1), dtype=numpy.float32)
        image_center[0, 0] = w / 2.0
        image_center[1, 0] = h / 2.0
        rotated_center = numpy.dot(a, image_center)
        corners_coordindates = [0.0, w, w, 0.0,
                                0.0, 0.0, h, h]
        corners = numpy.array(corners_coordindates, dtype=numpy.float32).reshape(2, 4)

        # calculate position of four corners
        rotated_corners = numpy.dot(a, corners)
        # calculte how tall and wide the rotated image should be
        diag1 = Point(
            math.fabs(rotated_corners[0, 0] - rotated_corners[0, 2]),
            math.fabs(rotated_corners[1, 0] - rotated_corners[1, 2]))
        diag2 = Point(
            math.fabs(rotated_corners[0, 1] - rotated_corners[0, 3]),
            math.fabs(rotated_corners[1, 1] - rotated_corners[1, 3]))
        target_center = Point(
            max(diag1.x, diag2.x) / 2.0,
            max(diag1.y, diag2.y) / 2.0)
        center_offset = Point(
            target_center.x - rotated_center[0, 0],
            target_center.y - rotated_center[1, 0]);
        result = numpy.zeros((3, 3), dtype=numpy.float32)
        # first, assign rotation matrix
        result[0:2, 0:2] = a.copy()
        result[0, 2] = center_offset.x
        result[1, 2] = center_offset.y
        result[2, 2] = 1.0
        return result

    def _get_inverse_transformation(self, homo_mat):
        a = numpy.copy(homo_mat[0:2, 0:2])
        inv_a = inv(a)
        inv_t = numpy.dot(numpy.dot(-1.0, inv_a), homo_mat[0:2, 2:3])
        data = [
            inv_a[0, 0], inv_a[0, 1], inv_t[0, 0],
            inv_a[1, 0], inv_a[1, 1], inv_t[1, 0],
            0.0, 0.0, 1.0]
        return numpy.array(data).reshape(3, 3)

    def _get_affine_transformed_image_size(self, img_size, homo_mat):
        w = img_size.width;
        h = img_size.height;
        corners_coordindates = [0.0, w, w, 0.0,
                                0.0, 0.0, h, h,
                                1.0, 1.0, 1.0, 1.0]
        homogeneous_corners = numpy.array(corners_coordindates, dtype=numpy.float32).reshape(3, 4)
        transformed_corners = numpy.dot(homo_mat, homogeneous_corners)
        transformed_corners[0] = transformed_corners[0] / transformed_corners[2];
        transformed_corners[1] = transformed_corners[1] / transformed_corners[2];
        transformed_corners[2] = numpy.ones((1, len(transformed_corners[2])), dtype=numpy.float32)
        max_corner_coord = cv2.reduce(transformed_corners, 1, cv.CV_REDUCE_MAX)
        return Size(int(max_corner_coord[0, 0]), int(max_corner_coord[1, 0]))

    def _get_image_affine_map(self, transformed_image_size, inv_homo_mat):
        pixel_locations = numpy.ones(
            (3, transformed_image_size.width * transformed_image_size.height),
            dtype=numpy.float32)
        pixel_idx = 0
        for row in xrange(transformed_image_size.height):
            for col in xrange(transformed_image_size.width):
                pixel_locations[0, pixel_idx] = col * 1.0
                pixel_locations[1, pixel_idx] = row * 1.0
                pixel_idx += 1

        transformed_pixel_locations = numpy.dot(inv_homo_mat, pixel_locations)
        image_affine_map = numpy.zeros(
            (transformed_image_size.height, transformed_image_size.width, 2),
            dtype=numpy.float32)
        pixel_idx = 0
        for row in xrange(transformed_image_size.height):
            for col in xrange(transformed_image_size.width):
                image_affine_map[row, col, 0] = transformed_pixel_locations[0, pixel_idx] / transformed_pixel_locations[
                    2, pixel_idx]
                image_affine_map[row, col, 1] = transformed_pixel_locations[1, pixel_idx] / transformed_pixel_locations[
                    2, pixel_idx]
                pixel_idx += 1
        return image_affine_map

    def _rectification_map(self, img_size):
        w = img_size.width
        h = img_size.height
        r = self._distort_f * w
        f = self._rect_f * w

        rect_center = Point(w / 2.0, h / 2.0)
        distort_center = Point(w / 2.0 + self._centeroffset.x, h / 2.0 + self._centeroffset.y)
        # get homogeneous transformation matrix and its inverse
        homo_mat = self._get_affine_transformation(img_size, self._rotation, self._resolution_factor)
        inv_homo_mat = self._get_inverse_transformation(homo_mat)
        transformed_image_size = self._get_affine_transformed_image_size(img_size, homo_mat)
        inv_image_affine_map = self._get_image_affine_map(transformed_image_size, inv_homo_mat)

        # check boundary on x axis
        if self._roi.x + self._roi.width > 1.0:
            self._roi.width = 1.0 - self._roi.x
        # check boundary on y axis
        if self._roi.y + self._roi.height > 1.0:
            self._roi.height = 1.0 - self._roi.y

        absolute_roi = Roi()
        absolute_roi.x = int(transformed_image_size.width * self._roi.x)
        absolute_roi.y = int(transformed_image_size.height * self._roi.y)
        absolute_roi.width = int(transformed_image_size.width * self._roi.width)
        absolute_roi.height = int(transformed_image_size.height * self._roi.height)

        self._xmap = numpy.zeros((absolute_roi.height, absolute_roi.width), dtype=numpy.float32)
        self._ymap = numpy.zeros((absolute_roi.height, absolute_roi.width), dtype=numpy.float32)

        # for every pixel on rectified img
        for i in xrange(absolute_roi.height):
            y = absolute_roi.y + i;
            for j in xrange(absolute_roi.width):
                x = absolute_roi.x + j
                untransformed_rect_point = Point(
                    inv_image_affine_map[y, x, 0],
                    inv_image_affine_map[y, x, 1]
                )
                xout = 0.0
                yout = 0.0
                if r > 0 and f > 0:
                    dist = self._distance(untransformed_rect_point, rect_center)
                    rout = r * math.atan(dist / f)
                    dist = 1 if dist == 0 else dist
                    xout = rout * (untransformed_rect_point.x - rect_center.x) / dist + distort_center.x
                    yout = rout * (untransformed_rect_point.y - rect_center.y) / dist + distort_center.y
                else:
                    xout = untransformed_rect_point.x
                    yout = untransformed_rect_point.y
                if xout < 0 or xout >= w or yout < 0 or yout >= h:
                    self._xmap[i, j] = -1
                    self._ymap[i, j] = -1
                else:
                    self._xmap[i, j] = xout
                    self._ymap[i, j] = yout

    def rectify_mat(self, distorted_mat):
        if self._xmap is None or self._ymap is None:
            self._rectification_map(Size(len(distorted_mat[1][0]), len(distorted_mat[1])))
        interpolation = cv2.INTER_LINEAR if self._use_bilinear else cv2.INTER_NEAREST
        return cv2.remap(distorted_mat[1], self._xmap, self._ymap, interpolation)  # , cv2.BORDER_CONSTANT , (0, 0, 0))

    def rectified_size(self, img_size):
        homo_mat = self._get_affine_transformation(img_size, self._rotation, self._resolution_factor)
        transformed_image_size = self._get_affine_transformed_image_size(img_size, homo_mat)

        roiw = self._roi.width
        roih = self._roi.height
        # check boundary on x axis
        if self._roi.x + self._roi.width > 1.0:
            roiw = 1.0 - self._roi.x
        # check boundary on y axis
        if self._roi.y + self._roi.height > 1.0:
            roih = 1.0 - self._roi.y

        return Size(int(transformed_image_size.width * roiw), int(transformed_image_size.height * roih))

    def config(self):
        return self._config


class MoveEvent(object):
    is_move = False
    is_exit = False
    edge = 0
    point = Point()


class VideoPreviewer(object):
    title = 'optimize_unwarp_config'

    def __init__(self, video_file, config_file):
        """
        """
        self._video_capture = cv2.VideoCapture(video_file)
        with open(config_file, 'r') as fin:
            self._config = json.loads(fin.read())

    def _instruct(self):
        logger.info(
            "Instructions: "
            "\nIt will show you which region of the video will be cropped for doing video mturk."
            "\nThe red line is the red line for doing video mturk."
            "\nYou can enlarge the region by dragging the *bottom* or *top* line. The aspect ratio of "
            "\nthe region is always 3:4."
            "\nAlso, you can move the red line by pressing *\"LEFT\"* or *\"RIGHT\"* Key."
            "\n"
            "\nPress *\"ENTER\"* to confirm the config."
            "\nPress *\"ESC\"* to quit without doing any thing."
        )

    def preview(self):
        uncropped_unwarp_config = copy.deepcopy(self._config)
        if uncropped_unwarp_config.has_key('unwarp'):
            uncropped_unwarp_config = uncropped_unwarp_config['unwarp']
        uncropped_unwarp_config['roiX'] = 0.0
        uncropped_unwarp_config['roiY'] = 0.0
        uncropped_unwarp_config['roiW'] = 1.0
        uncropped_unwarp_config['roiH'] = 1.0
        unwarper = VideopUnwarper(uncropped_unwarp_config)

        original_frame_size = Size(self._video_capture.get(cv.CV_CAP_PROP_FRAME_WIDTH),
                                   self._video_capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_size = unwarper.rectified_size(original_frame_size);

        new_unwarp_params = self._config;
        if self._config.has_key('unwarp'):
            new_unwarp_params = self._config['unwarp']
        right_x = (new_unwarp_params["roiX"] + new_unwarp_params["roiW"]) * frame_size.width
        bottom_y = new_unwarp_params["roiY"] * frame_size.height
        top_y = (new_unwarp_params["roiY"] + new_unwarp_params["roiH"]) * frame_size.height

        frame_left_bottom = Point(0, bottom_y)
        frame_right_bottom = Point(0, bottom_y)
        frame_left_top = Point(0, top_y)
        frame_right_top = Point(0, top_y)
        red_line_top = Point(right_x, top_y)
        red_line_bottom = Point(right_x, bottom_y)

        aspect_ratio = 0.75;  # w/h
        logger.info("unwarped size: %d x %d\nunwarp config:%s\n" % (
        frame_size.width, frame_size.height, str(new_unwarp_params)))
        self._instruct()

        cv2.namedWindow(self.title, cv.CV_WINDOW_AUTOSIZE)
        move_event = MoveEvent
        cv.SetMouseCallback(self.title, on_mouse, move_event)

        original = None
        rectified_frame = None
        prev_frame = None
        gray_frame = None
        diff_frame = None
        corner_weights = None
        min_ped_size = None
        freeze = False
        percent_change = 1.0;
        c = -1;
        # loop the video
        while True:
            # fast playback if no motion
            if percent_change < 0.005:
                c = cv2.waitKey(1)
            # slow playback if motion
            else:
                c = cv2.waitKey(40)
            c = c & 0xff
            if c == 27 or move_event.is_exit:  # esc exit
                move_event.is_exit = True;
                break
            elif c == 0x20:  # space, toggle freeze
                freeze = False if freeze else True
            elif c == 13 or c == 10:  # enter, confirm reference
                new_unwarp_params["roiY"] = red_line_bottom.y / float(frame_size.height)
                new_unwarp_params["roiH"] = (red_line_top.y - red_line_bottom.y) / float(frame_size.height)
                tmp_width = (red_line_top.y - red_line_bottom.y) * aspect_ratio
                if tmp_width - red_line_bottom.x > 0:
                    new_unwarp_params["roiX"] = (red_line_bottom.x - tmp_width) / float(frame_size.width)
                else:
                    new_unwarp_params["roiX"] = 0
                new_unwarp_params["roiW"] = red_line_bottom.x / float(frame_size.width)
                break
            elif c == 81 or c == 83:
                offset = 1 if c == 83 else -1
                right_x += offset
                if right_x > frame_size.width or right_x < 0:
                    right_x -= offset
                else:
                    red_line_top.x += offset
                    red_line_bottom.x += offset
            if not freeze:
                original = self._video_capture.read()
                if not original:
                    # end of video, return to the beginning and keep going
                    self._video_capture.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
                    continue  # break loop if no more frame

                rectified_frame = unwarper.rectify_mat(original)
                # calculate frame difference
                gray_frame = cv2.cvtColor(rectified_frame, cv.CV_RGB2GRAY).astype(numpy.float32)
                if prev_frame is None:
                    prev_frame = numpy.zeros((frame_size.height, frame_size.width), dtype=numpy.float32)
                    numpy.copyto(prev_frame, gray_frame)

                diff_frame = cv2.absdiff(gray_frame, prev_frame)
                diff_frame = cv2.threshold(diff_frame, 16, 1, cv.CV_THRESH_BINARY)
                diff_sum = cv2.countNonZero(diff_frame[1])
                percent_change = float(diff_sum) / float(gray_frame.size)
            else:
                if not original:
                    self._video_capture.set(cv.CV_CAP_PROP_POS_FRAMES, 0);
                    break;  # break loop if no more frame
                else:
                    rectified_frame = unwarper.rectify_mat(original)

            if move_event.is_move and move_event.point.y > 0 and move_event.point.y < frame_size.height:
                if move_event.edge == -1:
                    # decide moving which edge
                    frame_top_y = red_line_top.y
                    frame_bottom_y = red_line_bottom.y
                    threshold = (frame_top_y - frame_bottom_y) / 2 + frame_bottom_y
                    if threshold > move_event.point.y:
                        move_event.edge = 1
                    else:
                        move_event.edge = 0
                if move_event.edge == 0 and move_event.point.y > top_y:
                    frame_right_top.y = move_event.point.y
                    frame_left_top.y = move_event.point.y
                    red_line_top.y = move_event.point.y
                elif move_event.edge == 1 and move_event.point.y < bottom_y:
                    frame_right_bottom.y = move_event.point.y
                    frame_left_bottom.y = move_event.point.y
                    red_line_bottom.y = move_event.point.y

            expected_width = (red_line_top.y - red_line_bottom.y) * aspect_ratio
            if right_x < expected_width:
                frame_left_top.x = 0
                frame_left_bottom.x = 0
                if expected_width > frame_size.width:
                    frame_right_top.x = frame_size.width
                    frame_right_bottom.x = frame_size.width
                    diff_y = (expected_width - frame_size.width) / aspect_ratio
                    if move_event.edge == 0 and move_event.point.y > top_y:
                        frame_right_top.y -= diff_y
                        frame_left_top.y -= diff_y
                        red_line_top.y -= diff_y
                    elif move_event.edge == 1 and move_event.point.y < bottom_y:
                        frame_right_bottom.y += diff_y
                        frame_left_bottom.y += diff_y
                        red_line_bottom.y += diff_y
                else:
                    frame_right_top.x = expected_width
                    frame_right_bottom.x = expected_width
            else:
                frame_left_top.x = right_x - expected_width
                frame_left_bottom.x = right_x - expected_width
                frame_right_top.x = right_x
                frame_right_bottom.x = right_x

            cv2.line(rectified_frame,
                     frame_left_top.to_tuple(int),
                     frame_right_top.to_tuple(int), (255, 255, 0), 2)
            cv2.line(rectified_frame,
                     frame_right_top.to_tuple(int),
                     frame_right_bottom.to_tuple(int), (0, 255, 0), 2)
            cv2.line(rectified_frame,
                     frame_right_bottom.to_tuple(int),
                     frame_left_bottom.to_tuple(int), (255, 255, 0), 2)
            cv2.line(rectified_frame,
                     frame_left_bottom.to_tuple(int),
                     frame_left_top.to_tuple(int), (0, 255, 0), 2)
            cv2.line(rectified_frame,
                     red_line_top.to_tuple(int),
                     red_line_bottom.to_tuple(int), (0, 0, 255), 6)

            text = 'w/h: %6.3f' % aspect_ratio
            cv2.putText(rectified_frame, text, (0, frame_size.height), cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            cv2.imshow(self.title, rectified_frame)
            numpy.copyto(prev_frame, gray_frame)
        # end of for loop
        cv2.destroyWindow(self.title)
        return (not move_event.is_exit, self._config)

    def close(self):
        if not self._video_capture:
            self._video_capture.release()


def main():
    import sys

    previewer = VideoPreviewer(sys.argv[1], sys.argv[2])
    confirm, config = previewer.preview()
    if confirm:
        with open(sys.argv[2], 'wb') as fout:
            fout.write(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()