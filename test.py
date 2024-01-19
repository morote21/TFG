import numpy as np
import cv2
from ultralytics import YOLO
import copy

def resizeFrame(frame, height=1080):
    """
    Resizes the frame to a specific height
    :param frame: frame to resize
    :param height: height to resize to
    :return: resized frame
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
    width = int(height * aspect_ratio)
    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame


def getClassName(model, cls):
        """
        Returns the name of the class
        :param cls: class (int)
        :return: name of the class (string)
        """
        return model.names[int(cls)]

def getDistance(mean1, mean2):
    """
    Gets the distance between two means
    :param mean1: mean of the first player (tuple)
    :param mean2: mean of the second player (tuple)
    :return: distance between the two means (float)
    """
    return (mean1[0]-mean2[0])**2 + (mean1[1]-mean2[1])**2 + (mean1[2]-mean2[2])**2


def getMeanOfEachColorChannel(image):
    """
    Gets the mean of each color channel of the image
    :param image: image (np.array)
    :return: mean of each color channel (tuple)
    """
    # BGR
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    meanB = np.mean(b)
    meanG = np.mean(g)
    meanR = np.mean(r)

    return meanB, meanG, meanR

def drawBoundingBoxPlayer(image, boxes):
    """
    Draws a bounding box around the player and writes its identity
    :param image: image to draw (np.array)
    :param boxes: bounding boxes of the players (list)
    :return: image with the bounding box and the identity (np.array)
    """
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    return image

def getRoiOfPlayers(model_pose, image, boxes):
        """
        Gets the region of interest of the player
        :param image: image of the player (np.array)
        :return: region of interest of the player (np.array)
        """
        rois = []
        for box in boxes:
            player = copy.deepcopy(image[box[1]:box[3], box[0]:box[2]])
            results = model_pose.predict(player, show=False)
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]

            # if no keypoints detected, return None
            if len(keypoints) == 0:
                continue
            
            else:
                ls = keypoints[5]   # left shoulder
                rs = keypoints[6]   # right shoulder
                lh = keypoints[11]  # left hip
                rh = keypoints[12]  # right hip

                # crop image by left-most shoulder and right-most hip
                ls_x = int(ls[0])
                rs_x = int(rs[0])
                lh_x = int(lh[0])
                rh_x = int(rh[0])

                ls_y = int(ls[1])
                rh_y = int(rh[1])

                if ls_x < rs_x:
                    left_torax = ls_x
                    right_torax = rs_x
                else:
                    left_torax = rs_x
                    right_torax = ls_x

                if lh_x > rh_x:
                    right_hip = lh_x
                    #right_torax = lh_x
                    left_hip = rh_x
                else:
                    #right_torax = rh_x
                    right_hip = rh_x
                    left_hip = lh_x
                
                if left_hip < left_torax:
                    left_torax = left_hip
                
                if right_hip > right_torax:
                    right_torax = right_hip
                
                height = player.shape[0]
                width = player.shape[1]

                rois.append([left_torax + box[0], ls_y + box[1], right_torax + box[0], rh_y + box[1]])
                rois.append([int(left_hip - width*0.05) + box[0], int(rh_y - height*0.05) + box[1], int(right_hip + width*0.05) + box[0], int(rh_y + height*0.05) + box[1]])

                
        return rois

                # roiTorax = image[ls_y:rh_y, left_torax:right_torax]

                # roiHip = image[int(rh_y - height*0.10):int(rh_y + height*0.10), int(left_hip - width*0.10):int(right_hip + width*0.10)]

                # return roiTorax, roiHip
        
    
model = YOLO("./domainLayer/models/yolov8m.pt")
model_pose = YOLO("./domainLayer/models/yolov8m-pose.pt")

video = cv2.VideoCapture("/home/morote/Desktop/input_tfg/final_videos/3ptcontest_outdoor_4k_30fps.mp4")
if not video.isOpened():
    print("Error opening video stream or file")

while video.isOpened():
    ret, frame = video.read()
    if ret:
        frame = resizeFrame(frame)
        results = model.predict(source=frame, show=False, save=False, conf=0.3, device=0)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls

            indexes = [i for i, cls in enumerate(classes) if getClassName(model, cls) == "person"]    # cambiar por el cls directo de person para no buscar en self.model.names

            boxes = boxes[indexes]
            rois = getRoiOfPlayers(model_pose, frame, boxes)

            for roi in rois:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 255), 2)

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    else:
        break

# image1 = cv2.imread("/home/morote/Desktop/input_tfg/equipments/equipment_1_outdoor.png")
# image2 = cv2.imread("/home/morote/Desktop/input_tfg/equipments/equipment_2_outdoor.png")

# getRoiOfPlayer(model_pose, image1)
# getRoiOfPlayer(model_pose, image2)

# cv2.imshow("image1", image1)
# cv2.imshow("image2", image2)

# cv2.waitKey(0)
cv2.destroyAllWindows()

