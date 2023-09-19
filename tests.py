import cv2

print('Insert video path:')
video_path = input()

video = cv2.VideoCapture(video_path)

ret, frame = video.read()

cv2.imshow('first frame', frame)
while(1):
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break