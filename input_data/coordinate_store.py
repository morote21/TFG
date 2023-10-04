import numpy as np
import cv2


class CoordinateStore:
    def __init__(self):
        self.points_src = np.zeros(shape=(6, 2))
        self.i_src = 0
        self.points_dst = np.zeros(shape=(6, 2))
        self.i_dst = 0

    def select_point(self,event,x,y,flags,param):
        image_src = param[1]
        image_dst = param[2]

        if event == cv2.EVENT_LBUTTONDBLCLK:
            if (param[0] == 0):
                cv2.circle(image_src,(x,y),3,(255,0,0),2)
                self.points_src[self.i_src] = [x, y]
                self.i_src += 1
                print(self.points_src)
                if (self.i_src == 6):
                    param[0] = 1

            elif (param[0] == 1):
                cv2.circle(image_dst,(x,y),3,(255,0,0),2)
                self.points_dst[self.i_dst] = [x, y]
                self.i_dst += 1
                print(self.points_dst)
                if (self.i_dst == 6):
                    param[0] = 2

    def get_points_scene(self):
        return self.points_src

    def get_points_topview(self):
        return self.points_dst


    """def track_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Before homography: ({x}, {y})")
            h = param[0]
            p = np.array([x, y, 1])
            p = np.matmul(h, p)
            q = np.array([p[0]/p[2], p[1]/p[2]])
            print(f"After homography: ({q[0]}, {q[1]})")
            cv2.circle(image_dst,(int(q[0]),int(q[1])),3,(255,0,0),2)"""