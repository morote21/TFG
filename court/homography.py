import cv2
import numpy as np  


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

    def track_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Before homography: ({x}, {y})")
            h = param[0]
            p = np.array([x, y, 1])
            p = np.matmul(h, p)
            q = np.array([p[0]/p[2], p[1]/p[2]])
            print(f"After homography: ({q[0]}, {q[1]})")
            cv2.circle(image_dst,(int(q[0]),int(q[1])),3,(255,0,0),2)


class Homography:
    def __init__(self, scene_image, topview_image):
        self.scene = scene_image
        self.topview = topview_image
        
    def compute_homography(self):
        cc = CoordinateStore()
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        param = [0, self.scene, self.topview]

        cv2.setMouseCallback('image',cc.select_point, param)

        while(1):

            if (param[0] == 0):
                cv2.imshow('image',self.scene)
            elif (param[0] == 1):
                cv2.imshow('image',self.topview)
            else:
                cv2.destroyAllWindows()
                break

            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

        self.h_matrix, status = cv2.findHomography(cc.points_src, cc.points_dst, cv2.RANSAC, 3.0)


    def transform_point(self, point):
        x = point[0]
        y = point[1]
        p = np.array([x, y, 1])
        p = np.matmul(self.h_matrix, p)
        q = np.array([p[0]/p[2], p[1]/p[2]])
        return tuple((q[0], q[1]))
    
    def print_homography(self):
        print(self.h_matrix)



if __name__ == "__main__":
    #instantiate class
    cc = CoordinateStore()


    # Create a black image, a window and bind the function to window
    image_src = cv2.imread('./images/basketballcourt.jpeg')
    image_dst = cv2.imread('./images/synthetic_court3.png')
    cv2.namedWindow('image')

    param = [0]

    cv2.setMouseCallback('image',cc.select_point, param)

    while(1):

        if (param[0] == 0):
            cv2.imshow('image',image_src)
        elif (param[0] == 1):
            cv2.imshow('image',image_dst)
        else:
            cv2.destroyAllWindows()
            break

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    
    print("Selected Coordinates Object: ")
    for i in cc.points_src:
        print(i) 

    print("Selected Coordinates Scene: ")
    for i in cc.points_dst:
        print(i) 

    h, status = cv2.findHomography(cc.points_src, cc.points_dst, cv2.RANSAC, 3.0)
    print(h)
    image_dst = cv2.warpPerspective(image_src, h, (image_dst.shape[1], image_dst.shape[0]))
    # cv2.imshow("Result", image_dst)
    
    while(1):

        cv2.namedWindow('scene')
        cv2.namedWindow('topview')

        param = [h]
        cv2.setMouseCallback('scene',cc.track_point, param)

        cv2.imshow('scene', image_src)
        cv2.imshow('topview', image_dst)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()



    



