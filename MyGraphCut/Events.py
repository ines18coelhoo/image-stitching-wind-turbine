"""
    The 'Event' class stores information associated with an image that is taken 
    during a flight such as drone position and yaw, camera settings and 
    LIDAR readings. 
"""
import json
import numpy as np 
import ijson

class Events:
    """
        Represents an event (i.e., one image and state of the drone at the time
        when the image was taken during a drone flight)
    """
    # camera parameters
    focal_length = 0.085 # [m]
    sensor_width = 0.0359 # [m]
    sensor_height = 0.024 # [m]
    lidar_offset = 0.08 # [m]
    angle_offset = 0.04 # [rad]
    
    #img parameters y, x
    image_size = [5304, 7952]

    # convert focal length from [mm] to [pixel]
    pixel_size_x = sensor_width / image_size[1]
    focal_length_pix_x = focal_length / pixel_size_x

    pixel_size_y = sensor_height / image_size[0]
    focal_length_pix_y = focal_length / pixel_size_y

    camera_matrix = np.array([[focal_length_pix_x, 0, image_size[1], 0],
                              [0, focal_length_pix_y, image_size[0], 0],
                              [0, 0, 1, 0]])
    
    def __init__(self, lidar_filename, events_filename):
        
        with open(f'{events_filename}') as json_file:
            self.events = json.load(json_file)
            if self.events is None:
                print(f'[ERROR] could not read file {events_filename}' )
            json_file.close()

        with open(f'{lidar_filename}') as json_file:
            self.lidar = json.load(json_file)
            if self.lidar is None:
                print(f'[ERROR] could not read file {lidar_filename}' )
            json_file.close()
        
        
        self.offsets = []
        self.drone_z = []
        print('Calculating offsets..')
        self.get_offset(0)
        self.dif_altitudes = np.max(self.drone_z) - np.min(self.drone_z)
        self.from_least_squares(self.offsets, self.drone_z)

    def get_drone_z(self):
        return self.drone_z

    # Get the indexes of the deviation from the LiDAR center in the sweep vector
    def get_blade_edge_indexes(self, sweep_vector):
        self.blade_indexes = []
        for i in range(len(sweep_vector)):
            if sweep_vector[i] != "inf" and float(sweep_vector[i]) < 9 and (i > 500):
                self.blade_indexes.append(i)
            if i > 600:
                break

        return self.blade_indexes[0], self.blade_indexes[-1] 

    # [index] -> LiDAR index
    # [idx] -> event index
    def lidar_to_pixel(self, index, idx, offset):
        self.angle = float(self.lidar[idx]["angle_min"]) + float(self.lidar[idx]["inc_angle"]*index)
        #self.angle = float(self.lidar[idx]["inc_angle"]*(index-len(self.lidar[idx]["sweep"])//2))
        dist_lidar = float(self.lidar[idx]["sweep"][index])

        assert self.angle > -np.pi / 2. and self.angle < np.pi / 2 
        
        # Compute pixel values
        x = self.image_size[1] // 2 - np.tan(self.angle+offset)*self.focal_length_pix_x
        y = self.image_size[0] // 2 - self.focal_length_pix_y * self.lidar_offset / dist_lidar

        return np.round(np.array((x,y))).astype(np.int32)
    
    def from_least_squares(self, xs, ys): 
        """ 
            Returns the least squares line for the given values 
        """
        xs = np.atleast_2d(xs).T
        ys = np.atleast_2d(ys).T
        A = np.hstack((xs**0, xs**1))
        self.b, self.m = np.linalg.pinv(A).dot(ys).ravel()
    
    def eval(self, y, m, b): 
        """ 
            Returns the value of the polynomial evaluated at 'x'
        """
        if np.array(y).ndim == 0: 
            return (y - b) / m
        elif np.array(y).ndim == 1: 
            return np.array([(v-b)/m for v in y])
        else: 
            raise Exception("x must be 0D or 1D.")
    
    def get_offset(self, ref_idx):
        try:
            blade_left_index, blade_right_index = self.get_blade_edge_indexes(self.lidar[ref_idx]["sweep"])
        except:
            print('Could not find blade')
            ref_idx = ref_idx+1
            blade_left_index, blade_right_index = self.get_blade_edge_indexes(self.lidar[ref_idx]["sweep"])

        blade_right_angle = float(self.lidar[ref_idx]["inc_angle"]*(blade_right_index-len(self.lidar[ref_idx]["sweep"])//2))
        # Compute x and y coordinates of image center in world coordinates
        world_view_dir = np.array([np.cos(self.events[ref_idx]["drone_yaw"]), np.sin(self.events[ref_idx]["drone_yaw"])])# [m], unit vector
        world_orth_dir = np.array([world_view_dir[1], -world_view_dir[0]]) # [m], unit vector, corresponds to
                                                                        # `world_view_dir` rotated by 90 degrees
        world_image_center = np.array([self.events[ref_idx]["drone_position_x"], self.events[ref_idx]["drone_position_y"]]) + self.events[ref_idx]["distance_to_airfoil_center"] * world_view_dir # [m]

        for e, event in enumerate(self.events):
            # Vector from image center to position of drone (x and y world coordinates)
            world_offset_vec = np.array([event["drone_position_x"],event["drone_position_y"]]) - world_image_center # [m]
            # Inner product of inner of `world_offset_vec` and `world_orth_dir` corresponds to the offset between
            # LIDAR/image center of event `e` and the LIDAR/image center of the reference event, assuming that the yaw of
            # `e` and `event` is the same.
            offset = np.inner(world_orth_dir, world_offset_vec) # [m]

            # Yaw of `e` and `event` is usually not the same, attribute for that:
            d_yaw = event["drone_yaw"] - self.events[ref_idx]["drone_yaw"] # [m]
            offset += np.sin(-d_yaw) * event["distance_to_airfoil_center"] # [m]

            # Compute offset of blade center of `e` from `e`'s LIDAR/image center at blade distance
            try:
                blade_left_index, blade_right_index = self.get_blade_edge_indexes(self.lidar[e]["sweep"])
                self.drone_z.append(self.events[e]["drone_position_z"])
                #print(f'left sweep idx: {blade_left_index} \t value: {lidar[e]["sweep"][blade_left_index]}')
                #print(f'right sweep idx: {blade_right_index} \t value: {lidar[e]["sweep"][blade_right_index]}')
            except:
                print('Could not find blade')
                continue
            blade_left_angle = float(self.lidar[e]["inc_angle"]*(blade_left_index-len(self.lidar[e]["sweep"])//2))
            blade_left_pos = np.tan(np.tan(blade_left_angle)) * event["distance_to_airfoil_center"] # [m] deviation from LIDAR center
            blade_right_pos = np.tan(blade_right_angle) * self.events[ref_idx]["distance_to_airfoil_center"] # [m] deviation from LIDAR center
            blade_center_pos = (blade_left_pos + blade_right_pos) / 2 # [m] deviation from LIDAR center
            offset += blade_center_pos
            self.offsets.append(-offset)
        
