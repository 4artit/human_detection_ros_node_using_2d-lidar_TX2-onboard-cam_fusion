import rospy
from cv_bridge import CvBridge, CvBridgeError
from objdetection.msg import Detection, Object, Segmentation, Fusion
from sensor_msgs.msg import RegionOfInterest, Image, LaserScan
from realtime_object_detection.clustering import get_center_from_KNN, get_center_from_mid_points, get_center_from_DBSCAN
import numpy as np

### CONSTANT INDEX
INDICES = []
for i in range(180):
    INDICES.append(i)

### CONSTANT COS SIN VALUE
Angles = []
for i in range(360):
    angle = (i * np.pi)/180
    Angles.append(angle)
Angles = np.asarray(Angles)
Angles = np.hstack((Angles[0:90], Angles[270:360]))
Cos = np.cos(Angles)
Sin = np.sin(Angles)

### CONSTANT PHI DELTA K C F VALUE FOR LIDAR&CAMERA CALIBRATION - EDIT YOUR OWN PHI DELTA K C F
DELTA = np.matrix([-27.3456,-24.4341,-100.7541]).T
PHI = np.matrix([[0.9995, 0.0165, -0.0268], [0.0154,-0.9990,-0.0421], [-0.0275,0.0417,-0.9988]])
K = np.array([0.1335, -0.2579, -0.0037, 0.0005, 0])
C = np.array([326.405476764440830, 235.055227337685270])
F = np.array([472.684529357136970, 633.367424121568890])

class FusionPublisher(object):
    def __init__(self):
        self.FuPub = rospy.Publisher('fusion/human', Fusion, queue_size=10)
        self._bridge = CvBridge()
    
    def publish(self, boxes, scores, classes, num, category_index, pixels=[], points=[], masks=None, fps=0):
        msg = Fusion()
        dists = []
        angles= []
        for i in range(boxes.shape[0]):
            if category_index[classes[i]]['name'] == 'person':
                if scores[i] > 0.4:
                    class_name = 'person'
                    ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                    ymin = ymin*600 
                    xmin = xmin*600
                    ymax = ymax*600
                    xmax = xmax*600
                    box = RegionOfInterest()
                    box.x_offset = xmin + (xmax-xmin)/2.0
                    box.y_offset = ymin + (ymax-ymin)/2.0
                    box.height = ymax - ymin
                    box.width = xmax - xmin
                    box.do_rectify = True
                    # FUSION PART - calcualte distance and angle
                    j = 0
                    valid_points = []
                    for pixel in pixels:
                        if pixel[0] > xmin and pixel[0] < xmax and pixel[1] > ymin and pixel[1] < ymax:
                            valid_points.append([points[j][0], points[j][1]])
                        j = j + 1
                    ### KNN CODE
                    #if len(valid_points) >= 6:
                    #    dist, x, y = get_center_from_KNN(valid_points)
                    #    if dist[0] < dist[1]:
                    #        ang = np.arctan2(y[0], -x[0]) * (180 / np.pi)
                    #        msg.distance.append(dist[0])
                    #        msg.angle.append(ang)
                    #        dists.append(dist[0])
                    #        angles.append(ang)
                    #    else:
                    #        ang = np.arctan2(y[1], -x[1]) * (180 / np.pi)
                    #        msg.distance.append(dist[1])
                    #        msg.angle.append(ang)
                    #        dists.append(dist[1])
                    #        angles.append(ang)

                    ### N-points CODE(9)
                    #if len(valid_points) >= 9:
                    #    dist, ang = get_center_from_mid_points(valid_points)
                    #    msg.distance.append(dist)
                    #    msg.angle.append(ang)
                    #    dists.append(dist)
                    #    angles.append(ang)

                    ### DBSCAN CODE
                    if len(valid_points) >= 4:
                        dist, ang = get_center_from_DBSCAN(valid_points)
                        msg.distance.append(dist)
                        msg.angle.append(ang)
                        dists.append(dist)
                        angles.append(ang)

                    ###
                    else:
                        dists.append('None')
                        angles.append('None')
                    # fill detection message with objects
                    obj = Object()
                    obj.box = box
                    obj.class_name = class_name
                    obj.score = int(100*scores[i])
                    if masks is not None:
                        obj.mask = self._bridge.cv2_to_imgmsg(masks[i], encoding="passthrough")
                    msg.objects.append(obj)
            msg.fps = fps
        self.FuPub.publish(msg)
        return dists, angles

class DetectionPublisher(object):
    """
    Publish ROS detection messages
    """
    def __init__(self):
        self.DetPub = rospy.Publisher('dl/detection', Detection, queue_size=10)
        self._bridge = CvBridge()

    def publish(self, boxes, scores, classes, num, category_index, masks=None, fps=0):
        # init detection message
        msg = Detection()
        for i in range(boxes.shape[0]):
            if scores[i] > 0.5:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                ymin = ymin*600 
                xmin = xmin*600
                ymax = ymax*600
                xmax = xmax*600
                box = RegionOfInterest()
                box.x_offset = xmin + (xmax-xmin)/2.0
                box.y_offset = ymin + (ymax-ymin)/2.0
                box.height = ymax - ymin
                box.width = xmax - xmin
                # fill detection message with objects
                obj = Object()
                obj.box = box
                obj.class_name = class_name
                obj.score = int(100*scores[i])
                if masks is not None:
                    obj.mask = self._bridge.cv2_to_imgmsg(masks[i], encoding="passthrough")
                msg.objects.append(obj)
        msg.fps = fps
        # publish detection message
        self.DetPub.publish(msg)

class SegmentationPublisher(object):
    """
    Publish ROS detection messages
    """
    def __init__(self):
        self.SegPub = rospy.Publisher('Segmentation', Segmentation, queue_size=10)
        self._bridge = CvBridge()

    def publish(self, boxes, classes, labels, seg_map, fps=0):
        # init detection message
        msg = Segmentation()
        boxes = []
        for i in range(boxes.shape[0]):
            class_name = label[i]
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            box = RegionOfInterest()
            box.x_offset = xmin + (xmax-xmin)/2.0
            box.y_offset = ymin + (ymax-ymin)/2.0
            box.height = ymax - ymin
            box.width = xmax - xmin
            # fill segmentation message
            msg.boxes.append(box)
            msg.class_names.append(class_name)

        msg.seg_map = self._bridge.cv2_to_imgmsg(seg_map, encoding="passthrough")
        msg.fps = fps
        # publish detection message
        self.SegPub.publish(msg)

class ROSInput(object):
    """
    Capture video via ROS topic
    """
    def __init__(self, input, lidar_use):
        self._image = None
        self._lidar_image_pixels = None
        self._bridge = CvBridge()
        rospy.Subscriber(input, Image, self.imageCallback)
	if lidar_use is True:
            self._lidar_points = None
            rospy.Subscriber("/scan", LaserScan, self.lidarCallback)

    def imageCallback(self, data):
        try:
            image_raw = self._bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self._image = image_raw

    def lidarCallback(self, data):
        points = data.ranges
	points = np.asarray(points)
	points = np.hstack((points[0:90],points[270:360]))
        X =  - points * Cos
	Y = - points * Sin
	pts = np.matrix([Y, np.zeros(180), X])
	pts = pts * 1000

	invphi = np.linalg.inv(PHI)

	cpts = np.dot(invphi, pts) + DELTA

	xc = cpts[0]
	yc = cpts[1]
	zc = cpts[2]

	a = xc / zc
	b = yc / zc

	r = np.sqrt(np.power(a, 2) + np.power(b, 2))

	ad = np.multiply(a, (1 + np.multiply(K[0], np.power(r, 2)) + np.multiply(K[1], np.power(r, 4)) + np.multiply(K[4], np.power(r, 6)))) 
	+ 2 * K[2] * np.multiply(a, b) + K[3] * (np.power(r, 2) + 2 * np.power(a, 2))
	bd = np.multiply(b, (1 + np.multiply(K[0], np.power(r, 2)) + np.multiply(K[1], np.power(r, 4)) + np.multiply(K[4], np.power(r, 6))))
	+ K[2] * (np.power(r, 2) + 2 * np.power(b, 2)) + 2 * K[3] * np.multiply(a, b)

	x = (F[0] * a + C[0]) * (600./640.);
	y = (F[1] * b + C[1]) * (600./480.);
        pixel_list = []
        points = []
        for i in range(np.shape(x)[1]):
            if x[0, i] >= 0 and x[0, i] < 600 and y[0, i] >=0 and y[0, i] < 600:
                pixel_list.append((int(round(x[0, i])), int(round(y[0, i]))))
                points.append((X[i], Y[i]))
        self._lidar_points = points
        self._lidar_image_pixels = pixel_list

    def isActive(self):
        return True

    @property
    def image(self):
        return self._image

    def cleanup(self):
        pass

    def isEnabled(self):
        return self._enabled
