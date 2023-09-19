import cv2
import sys
import copy
from court.homography import CoordinateStore, Homography
from ultralytics import YOLO

def main():

    print('Insert video path:')
    video_path = input()

    print('Insert top view image path:')
    topview_path = input()
    topview_image = cv2.imread(topview_path)

    print('Select 6 points in the same order on both images:')
    video = cv2.VideoCapture(video_path)

    # Read first frame for homography computation
    ret, frame = video.read()

    if not ret:
        print("Error reading video frame.\n")

    h = Homography(scene_image=frame, topview_image=topview_image)
    h.compute_homography()

    h.print_homography()
    
    model = YOLO('./tracking/yolov8m-pose.pt')


    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(source=frame, show=False, save=False, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        topview_image_copy = topview_image.copy()
        for box, id in zip(boxes, ids):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (box[0], box[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
            floor_point = ((box[0] + box[2]) // 2, box[3])
            cv2.circle(frame, floor_point, 3, (0, 255, 0), 2)

            floor_point_transformed = h.transform_point(floor_point)
            cv2.circle(topview_image_copy, (int(floor_point_transformed[0]), int(floor_point_transformed[1])), 3, (0, 255, 0), 2)

            cv2.imshow("frame", frame)
            cv2.imshow("topview", topview_image_copy)
            key = cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()