import numpy as np
import cv2


class CoordinateStore:
    def __init__(self):
        self.points_src = np.zeros(shape=(8, 2))
        self.i_src = 0
        self.points_dst = np.zeros(shape=(8, 2))
        self.i_dst = 0

    def select_point(self,event,x,y,flags,param):
        image_src = param[1]
        image_dst = param[2]

        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
            if param[0] == 0:
                cv2.circle(image_src,(x,y),3,(255,0,0),2)
                self.points_src[self.i_src] = [x, y]
                self.i_src += 1
                # print(self.points_src)
                if self.i_src == 8:
                    param[0] = 1

            elif param[0] == 1:
                cv2.circle(image_dst,(x,y),3,(255,0,0),2)
                self.points_dst[self.i_dst] = [x, y]
                self.i_dst += 1
                # print(self.points_dst)
                if self.i_dst == 8:
                    param[0] = 2


    def get_points_scene(self):
        return self.points_src

    def get_points_topview(self):
        return self.points_dst

