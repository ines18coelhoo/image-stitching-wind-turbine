
import cv2
import numpy as np
import maxflow
from statistics import stdev
import skimage
from skimage import measure
from skimage import morphology

class GraphMaker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000

    def __init__(self):
        self.image = None
        self.graph = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.mask = None
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

    def rescaleFrame(self, frame, scale):
        self.width = int(frame.shape[1] * scale)
        self.height = int(frame.shape[0] * scale)
        self.dimensions = (self.width, self.height)
        
        return cv2.resize(frame, self.dimensions, cv2.INTER_LANCZOS4)

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        # resize image
        #self.image = self.rescaleFrame(self.image, scale = 0.1)
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None
    
    def grab_cut(self, mask, rect, output_dir, image_name):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask, bgdModel, fgdModel = cv2.grabCut(self.image, mask , rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
        # Run connected components and only keep the largest foreground area.
        mask = self.remove_isolated_regions(mask)

        # Use morphology to get rid of smaller artifacts
        kernel = morphology.disk(radius=3)
        mask = cv2.erode(mask, kernel, iterations=10)
        mask = cv2.dilate(mask, kernel, iterations=10)
        
        #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        self.image = self.image*mask[:,:,np.newaxis]
        img_name = image_name.split('/')
        img_name = img_name[1].split('.JPG')
        cv2.imwrite(f'{output_dir}/{img_name[0]}_grab.JPG', self.image)

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        else:
            return self.segment_overlay

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.4, 0.1)

    def create_graph(self, tip):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        print("Making graph")
        print("Finding foreground and background averages")
        self.find_averages()

        print("Populating nodes and edges")
        self.populate_graph()

        print("Cutting graph")
        self.cut_graph(tip)

    def find_averages(self):
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))
        print(self.graph.shape)
        self.graph.fill(self.default)
        
        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0

        for coordinate in self.foreground_seeds:
            try:
                self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1
            except:
                self.graph[5304//2, coordinate[0] - 1] = 1
    
    def populate_graph(self):
        self.nodes = []
        self.edges = []

        # make all s and t connections for the graph
        # populate nodes
        for (y, x), value in np.ndenumerate(self.graph):
            # this is a background pixel
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), self.MAXIMUM, 0))

            # this is a foreground node
            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAXIMUM))

            else:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        sigma = np.std(self.image)
        # populate edges
        for (y, x), value in np.ndenumerate(self.graph):
            # if the point is in the borders
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue

            my_index = self.get_node_num(x, y, self.image.shape)

            neighbor_index = self.get_node_num(x+1, y, self.image.shape)
            """
            if self.image[y, x].all() > self.image[y, x+1].all():
                g = np.exp(-(np.power(self.image[y, x] - self.image[y, x+1], 2))/(2*sigma*sigma))
            else:
                g = 1
            """
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x+1], 2)))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y+1, self.image.shape)
            """
            if self.image[y, x].all() > self.image[y+1, x].all():
                g = np.exp(-(np.power(self.image[y, x] - self.image[y+1, x], 2))/(2*sigma*sigma))
            else:
                g = 1
            """
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y+1, x], 2)))
            self.edges.append((my_index, neighbor_index, g))

    def cut_graph(self, tip):
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        # returns the identifiers of the nodes added
        nodelist = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])


        # Perform the maxflow computation in the graph. 
        # Returns the capacity of the minimum cut or, equivalently, the maximum flow of the graph.
        flow = g.maxflow()
        left_edge_x = []
        y_prev = 0

        for index in range(len(self.nodes)):
            # If it is foreground
            if g.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                self.segment_overlay[xy[1], xy[0]] = (255, 0, 255)
                self.mask[xy[1], xy[0]] = (True, True, True)
                
                if not tip:
                    if xy[1] > self.image.shape[0] - 1000 and y_prev != xy[1]:
                        left_edge_x.append(xy[0])
                y_prev = xy[1]

        if tip:
            return

        try:
            b, m = self.from_least_squares(left_edge_x, np.arange(self.image.shape[0] - 999, self.image.shape[0], 1).tolist())
            print(f'm = {m} \t b = {b}')        
        except:
            return
        

        for y in range(self.image.shape[0] - 990):
            x = int((y-b)/m)
            self.mask[y, x] = (True, True, True)
            i = 1
            while 1:
                self.mask[y, x+i] = (True, True, True)
                i = i + 1
                if (self.mask[y, x+i]).any():
                    break
     

    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num

    def save_image(self, filename, filedir):
        if self.mask is None:
            print('Please segment the image before saving.')
            return

        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        image_name = filename.split('/')
        image_name = image_name[2].split('.JPG')
        print(f'Saving image: {filedir}/clean/{image_name[0]}.JPG')
        cv2.imwrite(f'{filedir}/clean/{image_name[0]}.JPG', to_save)

    

    def remove_isolated_regions(self, mask):
        """
        @brief Runs a connected components algorithm and removes small isloated regions.

        @param mask Input mask

        @return The output mask
        """
        all_labels = measure.label(mask)

        min_index = np.min(all_labels)
        max_index = np.max(all_labels)

        # Find biggest connected foreground area and set all else to background
        max_size = 0
        best_index = -1
        for i in range(min_index, max_index+1):
            indices =  np.where(all_labels == i)
            val = mask[indices[0][0], indices[1][0]] # whether foreground or background
            if val == 0: continue
            size = len(indices[0])
            if size > max_size:
                best_index = i
                max_size = size
        mask[all_labels != best_index] = 0

        return mask
            
    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        # [0] -> x
        # [1] -> y
        return (int(nodenum % array_shape[1])), (int(nodenum / array_shape[1]))
    
    @staticmethod
    def from_least_squares(xs, ys): 
        """ 
            Returns the least squares line for the given values 
        """
        xs = np.atleast_2d(xs).T
        ys = np.atleast_2d(ys).T
        A = np.hstack((xs**0, xs**1))
        # [0] -> b
        # [1] -> m
        return np.linalg.pinv(A).dot(ys).ravel() 