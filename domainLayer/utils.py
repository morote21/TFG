import cv2
import numpy as np

def getBorders(image):

    def clickEvent(event, x, y, flags, param):        
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
            param.append((x, y))
    
    borders = []
    cv2.namedWindow("borders")
    cv2.setMouseCallback("borders", clickEvent, param=borders)

    while 1:
        cv2.imshow("borders", image)
        if len(borders) == 8:
            cv2.destroyAllWindows()
            break
        
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return np.array(borders)