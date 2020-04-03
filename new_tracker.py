from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils
import time
import cv2
import numpy as np
from collections import deque



initBB = None
fps = None

vs_1 = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)
all_points = {}
# pts = [] # deque(maxlen=100)
write = True
counter = 0
color = (0, 0, 255)


def get_current_frame(vs):
    _, frame = vs.read()
    frame = cv2.flip(frame, 1)

    if frame is None:
        return None

    frame = imutils.resize(frame, width=800)
    return frame


def get_ranges(box, frame):

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    (X, Y, W, H) = [int(v) for v in box]
    box1 = [X+W//4, Y + H//4, W//2, H//2]
    box2 = [X+W//8, Y + H//8, 3*W//4, 3*H//4]

    hsv_box_frame_1 = hsv[box1[1]: box1[1]+box1[3], box1[0]: box1[0]+box1[2], :]
    hsv_box_frame_2 = hsv[box2[1]: box2[1] + box2[3], box2[0]: box2[0] + box2[2], :]

    # box1_hsv_lower = np.array(
    #     [np.percentile(hsv_box_frame_1[:, :, 0], 25), np.percentile(hsv_box_frame_1[:, :, 1], 25), np.percentile(hsv_box_frame_1[:, :, 2], 25)])
    # box1_hsv_upper = np.array(
    #     [np.percentile(hsv_box_frame_1[:, :, 0], 75), np.percentile(hsv_box_frame_1[:, :, 1], 75), np.percentile(hsv_box_frame_1[:, :, 2], 75)])

    box1_hsv_lower = np.array([hsv_box_frame_1[:, :, 0].min(), hsv_box_frame_1[:, :, 1].min(), hsv_box_frame_1[:, :, 2].min()])
    box1_hsv_upper = np.array([hsv_box_frame_1[:, :, 0].max(), hsv_box_frame_1[:, :, 1].max(), hsv_box_frame_1[:, :, 2].max()])

    # box2_hsv_lower = np.array(
    #     [np.percentile(hsv_box_frame_2[:, :, 0], 40), np.percentile(hsv_box_frame_2[:, :, 1], 40), np.percentile(hsv_box_frame_2[:, :, 2], 40)])
    # box2_hsv_upper = np.array(
    #     [np.percentile(hsv_box_frame_2[:, :, 0], 60), np.percentile(hsv_box_frame_2[:, :, 1], 60), np.percentile(hsv_box_frame_2[:, :, 2], 60)])

    box2_hsv_lower = np.array([hsv_box_frame_2[:, :, 0].min(), hsv_box_frame_2[:, :, 1].min(), hsv_box_frame_2[:, :, 2].min()])
    box2_hsv_upper = np.array([hsv_box_frame_2[:, :, 0].max(), hsv_box_frame_2[:, :, 1].max(), hsv_box_frame_2[:, :, 2].max()])

    # print(box1_hsv_lower, box1_hsv_upper)
    # print(box2_hsv_lower, box2_hsv_upper)

    return [{'lower': box1_hsv_lower, 'upper': box1_hsv_upper}, {'lower': box2_hsv_lower, 'upper': box2_hsv_upper}]


def get_countor(box, ranges, frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(hsv, ranges[0]['lower'], ranges[0]['upper'])
    mask_2 = cv2.inRange(hsv, ranges[1]['lower'], ranges[1]['upper'])
    mask = mask_1 + mask_2

    (X, Y, W, H) = [int(v) for v in box]
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        final_radius = min(max(W//2, H//2)*2, radius)
        # only proceed if the radius meets a minimum size
        if radius > 5:
            # 	# draw the circle and centroid on the frame,
            # 	# then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(final_radius),
                       (255, 255, 255), 2)
            # cv2.circle(frame, center, 8, (0, 0, 255), -1)
        return int(x), int(y), int(final_radius)
    return None, None, None


while vs.isOpened():
    frame = get_current_frame(vs)
    if frame is None:
        break
    (H, W) = frame.shape[:2]
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # print(frame.shape)
    # print(HSV.shape)

    if initBB is not None:
        (success, box) = tracker.update(frame)
        # print(box, success)
        new_x, new_y, new_radius = get_countor(box, ranges, frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
        else:
            # new_x, new_y, new_radius = get_countor(box, ranges, frame)
            # print(new_x, new_y, new_radius, success)
            if (new_radius is not None) & (new_x is not None) & (new_y is not None):

                new_x, new_y, new_radius = get_countor(box, ranges, frame)
                x, y, w, h = int(new_x - new_radius), int(new_y - new_radius), int(new_radius), int(new_radius)

                # counter += 1
                # cv2.imshow("Frame", frame)
                # continue
            else:
                print('skipping')
                write = False
                continue

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
        center = (x + w // 2, y + h // 2)

        fps.update()
        fps.stop()
        info = [
            ("Tracker", "Test"),
            ("color", color),
            # ("radius", "{:.2f}".format(new_radius)),
            ("Write", write)
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('t'):
            write = not write
            counter += 1

        if key == ord('b'):
            color = (255, 0, 0)

        if key == ord('g'):
            color = (0, 255, 0)

        if key == ord('r'):
            color = (0, 0, 255)

        if write:
            try:
                all_points[counter].append([center, color])
            except KeyError:
                all_points[counter] = [[center, color]]

        if key == ord('c'):
            all_points = {}
            write = False

        for ctr, pts_new in all_points.items():
            for i in range(1, len(pts_new)):
                try:
                    if pts_new[i - 1] is None or pts_new[i] is None:
                        continue
                except IndexError:
                    continue

                thickness = 5  # pts_new[i][2]  # / float(i + 1)
                cv2.line(frame, pts_new[i - 1][0], pts_new[i][0], pts_new[i][1], thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        tracker = cv2.TrackerCSRT_create()
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)

        # print(initBB, type(initBB), type(initBB[0]))

        ranges = get_ranges(initBB, frame)
        tracker.init(frame, initBB)
        fps = FPS().start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
