
import numpy as np
from numpy.lib import angle
from GraphMaker import GraphMaker
import random
import cv2 

class CutUI:

    # constants
    lidar_offset = 0.08 # [m]
    angle_offset = 0.04 # [rad]

    def __init__(self, image_name, output_dir):
        self.image_name = image_name
        self.graph_maker = GraphMaker()
        print(f'Loading image: {image_name}')
        GraphMaker.load_image(self.graph_maker, filename= image_name)
        self.display_image = np.array(self.graph_maker.image)
        self.mode = self.graph_maker.foreground
        self.output_dir = output_dir
        self.tip = False

    def run(self, index, events):
        #drone is still adjusting
        if (int(events.events[index]["flight_phase"]) == 1 and events.drone_z[index] > events.drone_z[index+1] or int(events.events[index]["event_id"]) < 6):
            return

        # get left and right indexes of the edges of the blade in LiDAR coordinates 
        try:
            self.right_index, self.left_index = events.get_blade_edge_indexes(events.lidar[index]["sweep"])
        except:
            print(f'Could not find blade of id: {events.lidar[index]["event_id"]}')
            return

        # dirty
        # self.point_center_left = events.lidar_to_pixel(self.left_index, index, -self.angle_offset) 
        self.point_center_left = events.lidar_to_pixel(self.left_index, index, 0) 
        self.point_center_right = events.lidar_to_pixel(self.right_index, index, 0) 

        # generate polygon
        print('generating the polygon...')
        self.check_accuracy_of_lidar_readings(events)
        print(f'point left edge: {self.point_center_left}')
        print(f'point right edge: {self.point_center_right}')
 
        """
            GRAB CUT
        """
        """
        mask = np.zeros(events.image_size,np.uint8)
        
        # mark everything as background
        mask[:,:] = cv2.GC_BGD
        
        # original x coordinates of the edges
        left_orig = self.point_center_left[0]
        right_orig = self.point_center_right[0]
        # likely to be background
        self.point_center_left[0] = left_orig-200
        self.point_center_right[0] = right_orig+200
        pts = self.generate_points(events, index, self.point_center_left, self.point_center_right)
        cv2.fillPoly(mask, [pts], cv2.GC_PR_BGD)
        # likely to be foreground
        self.point_center_left[0] = left_orig-100
        self.point_center_right[0] = right_orig+100
        pts = self.generate_points(events, index, self.point_center_left, self.point_center_right)
        cv2.fillPoly(mask, [pts], cv2.GC_PR_FGD)   
        # foreground
        self.point_center_left[0] = left_orig
        self.point_center_right[0] = right_orig
        pts = self.generate_points(events, index, self.point_center_left, self.point_center_right)
        cv2.fillPoly(mask, [pts], cv2.GC_FGD)

        rect = (self.point_center_left[0]-300,0,abs(self.point_center_right[0]-self.point_center_left[0])+400,events.image_size[0])
        
        self.graph_maker.grab_cut(mask, rect, self.output_dir, self.image_name)
        """
        """
            GRAPH CUT
        """
        
        # line connecting left edge with right edge
        for pixel in range(0,self.point_center_right[0] - self.point_center_left[0], 2):
            self.graph_maker.add_seed(self.point_center_left[0]+pixel, self.point_center_left[1], self.mode)
        
        # if it is a tip the line only connects the center to the top
        if events.drone_z[index] < 1.1*np.min(events.drone_z):
            print('TIP DETECTED')
            print(f'id: {index} \t drone z: {events.drone_z[index]} \t min: {np.min(events.drone_z)}')
            self.tip = True
            for pixel in range(0,int(events.image_size[0]//2), 10):
                self.graph_maker.add_seed((self.point_center_left[0]+self.point_center_right[0])//2, self.point_center_left[1]-pixel, self.mode)
        else: 
            # line from the bottom to the top 
            self.tip = False
            for pixel in range(0,int(events.image_size[0]//2), 10):
                self.graph_maker.add_seed((self.point_center_left[0]+self.point_center_right[0])//2, self.point_center_left[1]+pixel, self.mode)
                self.graph_maker.add_seed((self.point_center_left[0]+self.point_center_right[0])//2, self.point_center_left[1]-pixel, self.mode)
        
        #line connecting left edge with top
        for pixel in range(0,(self.point_center_right[0] - self.point_center_left[0])//2, 10):
            self.graph_maker.add_seed(self.point_center_left[0]+pixel, self.point_center_left[1]-pixel, self.mode)
            self.graph_maker.add_seed(self.point_center_left[0]+pixel, self.point_center_left[1]+pixel, self.mode)
            self.graph_maker.add_seed(self.point_center_right[0]-pixel, self.point_center_right[1]-pixel, self.mode)
            self.graph_maker.add_seed(self.point_center_right[0]-pixel, self.point_center_right[1]+pixel, self.mode)
        
        # generate seeds for background
        self.mode = 1 - self.mode
        # deviate edges to the background part 
        
        for pixel in range(0, events.image_size[0], 10):
            # left line from top to bottom 
            self.graph_maker.add_seed(0, pixel, self.mode)
            self.graph_maker.add_seed(50, pixel, self.mode)
            self.graph_maker.add_seed(100, pixel, self.mode)

            #right line from top to bottom
            self.graph_maker.add_seed(events.image_size[1], pixel, self.mode)
            self.graph_maker.add_seed(events.image_size[1]-50, pixel, self.mode)
            self.graph_maker.add_seed(events.image_size[1]-100, pixel, self.mode)
            if int(events.events[index]["flight_phase"]) == 3:
                self.graph_maker.add_seed(3000, pixel, self.mode)
                self.graph_maker.add_seed(events.image_size[1]-2500, pixel, self.mode)

        # create graph
        self.graph_maker.create_graph(self.tip)
        # save result
        self.graph_maker.save_image(self.image_name, self.output_dir)
        
       

    def check_accuracy_of_lidar_readings(self, events):
        # check accuracy of lidar readings
        if self.point_center_left [0] > self.point_center_right[0]:
            self.aux = self.point_center_left
            self.point_center_left = self.point_center_right
            self.point_center_right = self.aux
        if self.point_center_left[0] < 0:
            self.point_center_left[0] = 0
        if self.point_center_right[0] > events.image_size[1]:
            self.point_center_right[0] = events.image_size[1]
        if abs(self.point_center_left[0] - self.point_center_right[0]) < 200:
            self.point_center_left[0] = self.point_center_left[0] - 160
            self.point_center_right[0] = self.point_center_right[0] + 160
    
    def generate_points(self, e, idx, left, right):
        tilt_angle = np.arctan(10)
        deviation = (e.image_size[0]//2)//np.tan(tilt_angle) 

        # high pressure side
        if int(e.events[idx]["flight_phase"]) == 1:
            x_tl = left[0] + deviation
            x_tr = right[0] + deviation
            x_br = right[0] - deviation
            x_bl = left[0] - deviation
            
        # leading edge 
        elif int(e.events[idx]["flight_phase"]) == 3:
            if (abs(left[0]- right[0] < 100)):
                left[0] = left[0]-250
                right[0] = right[0]+250
            x_tl = left[0] + deviation
            x_tr = right[0] - deviation
            x_br = right[0] - deviation*2
            x_bl = left[0] + deviation
            if (e.events[idx]["drone_position_z"] < np.mean(e.get_drone_z())):
                x_tl = x_bl = left[0] 
                x_tr = x_br = right[0] 

        # low pressure side        
        # trailing edge
        else: 
            x_tl = left[0] - deviation
            x_tr = right[0] - deviation
            x_br = right[0] + deviation
            x_bl = left[0] + deviation 

        y_bottom = e.image_size[0]

        # tip part of the blade
        if e.events[idx]["drone_position_z"] < np.min(e.get_drone_z())*1.10:
            x_bl = (left[0] + right[0])//2
            x_br = x_bl
            y_bottom = e.image_size[0]//2*(0.9+e.events[idx]["drone_position_z"]-np.min(e.get_drone_z()))     
            
        # compute points of the polygon
        pts = np.array([[int(x_tl), 0], 
                        [int(x_tr), 0], 
                        tuple(self.point_center_right),
                        [x_br, y_bottom],
                        [x_bl, y_bottom],
                        tuple(self.point_center_left),
                        ], dtype=np.int32)

        return pts.reshape((-1, 1, 2))

    