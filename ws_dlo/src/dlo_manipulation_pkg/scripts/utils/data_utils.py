import cv2
import numpy as np
# import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

import torch
import torch_geometric as pyg
from torch_geometric.nn import radius_graph

# get homogeneous points
def homo_points(pts):
    pts = np.array(pts,dtype=np.float32)
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def transform(pts,M):
    return np.dot(pts, M.T)

def invert_affine(M):
    A = M[:, :2]      # 2x2 linear part
    t = M[:, 2]       # 2x1 translation part
    A_inv = np.linalg.inv(A)
    t_inv = -np.dot(A_inv, t)
    M_inv = np.hstack([A_inv, t_inv[:, None]])  # shape (2, 3)
    return M_inv
def to_pixels(pos, width=1920, height=1080,
              x_range=(0, 1), y_range=(0, 1),
              boundary='clip'):
    """
    Convert coordinates in arbitrary ranges to pixel coordinates.

    Args:
        pos: (N,2) numpy array of [x, y] coordinates.
        width, height: image resolution.
        x_range: (min_x, max_x) range of input x values.
        y_range: (min_y, max_y) range of input y values.
        boundary:
            'clip'   -> clamp values to [0, width-1]/[0, height-1]
            'wrap'   -> wrap around image boundaries (modulo)
            'ignore' -> leave out-of-bounds values as is

    Returns:
        (x_pix, y_pix): int numpy arrays of pixel coordinates
    """
    x, y = pos[:, 0], pos[:, 1]

    # Normalize to [0, 1]
    x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])

    # Map to pixel coordinates (flip y)
    x_pix = x_norm * (width - 1)
    y_pix = (1 - y_norm) * (height - 1)

    # Handle boundaries
    if boundary == 'clip':
        x_pix = np.clip(x_pix, 0, width - 1)
        y_pix = np.clip(y_pix, 0, height - 1)
    elif boundary == 'wrap':
        x_pix = np.mod(x_pix, width)
        y_pix = np.mod(y_pix, height)
    elif boundary == 'ignore':
        pass
    else:
        raise ValueError("boundary must be one of ['clip', 'wrap', 'ignore']")

    return x_pix, y_pix

def concat_np_pos(fp_pos, end_pos):
    # concat end pose and fp pos
    end1_pos_xy, end2_pos_xy = end_pos[0,:2].reshape(1,-1), end_pos[1,:2].reshape(1,-1)
    return np.concatenate((end1_pos_xy,fp_pos,end2_pos_xy),axis=0)


def get_graph_data(fp_pos, fp_pos_d, end_pos, end_pos_d, delta_t, miss_idx=[], shift=True,
                            normalize_v=True, use_yaw_d = False, yaw_scale=20):
    # default torch.nn.linear dtype is torch.float 32, input data need to match
    fp_pos = torch.tensor(fp_pos, dtype=torch.float32)
    if fp_pos_d is not None:
        fp_pos_d = torch.tensor(fp_pos_d, dtype=torch.float32)
    end_pos = torch.tensor(end_pos, dtype=torch.float32)
    end_pos_d = torch.tensor(end_pos_d, dtype=torch.float32)
    delta_t = torch.tensor(delta_t, dtype=torch.float32)

    # shift pos, relative to the left most end pos
    if shift:
        shift_pos = end_pos[0:1, :2]  # left end pos
        fp_pos = fp_pos - shift_pos
        end_pos[:, :2] = end_pos[:, :2] - shift_pos

    if normalize_v:
        # --- Normalize velocity vector (per frame, per point) ---
        # Add small epsilon to avoid division by small magnitude and cause large velocity
        epsilon = 1e-2

        # Compute magnitude of total end-effector motion (average both ends)
        end_speed = np.linalg.norm(end_pos_d[:, :2], axis=1)  # [N-1, 2] (x, y only)
        end_avg_speed = np.mean(end_speed, axis=0, keepdims=True)  # [N-1, 1]  # only x, y
        # Apply floor
        end_avg_speed = np.maximum(end_avg_speed, epsilon)  # avoid divide very small values

        # normalize velocity and yaw? as well
        end_pos_d = end_pos_d / end_avg_speed  # unit direction vectors
        end_pos_d[:, 2] = end_pos_d[:, 2] / yaw_scale  # scale yaw, otherwise too large
        if fp_pos_d is not None:
            fp_pos_d = fp_pos_d / end_avg_speed

    # concat fp and end pos/velocity together
    pos_concat = torch.concat((end_pos[0, :2].unsqueeze(0), fp_pos, end_pos[1, :2].unsqueeze(0)), axis=0)

    if fp_pos_d is not None:
        velocity_concat = torch.concat((end_pos_d[0, :2].unsqueeze(0), fp_pos_d, end_pos_d[1, :2].unsqueeze(0)), axis=0)
        # y is velocity
        y = velocity_concat
    else:
        y = None

    edge_index = radius_graph(pos_concat, r=0.2, loop=False)  # loop=False to avoid self-loops

    # dim = node_feature.size(-1) #dim=2

    # get action
    if use_yaw_d:
        # get action, only for x,y
        action = torch.zeros((pos_concat.shape[0],end_pos_d.shape[1]))
        # Compute difference at the ends
        action[0, :] = end_pos_d[0, :].unsqueeze(0)
        action[-1, :] = end_pos_d[1, :].unsqueeze(0)
    else:
        # get action, only for x,y
        action = torch.zeros_like(pos_concat)
        # Compute difference at the ends
        action[0, :] = end_pos_d[0, :2].unsqueeze(0)
        action[-1, :] = end_pos_d[1, :2].unsqueeze(0)

    # node feature is the concatenation of [state, action]
    node_feature = torch.cat((pos_concat, action), dim=1)

    # edge_feature is [xi, xj, relative_movement]
    # sender_feature = node_feature[edge_index[0]]
    # receiver_feature = node_feature[edge_index[1]]
    relative_move = pos_concat[edge_index[0]] - pos_concat[edge_index[1]]
    edge_feature = relative_move

    # edge_feature = torch.cat([sender_feature,receiver_feature,relative_move],dim=1)

    # return the graph with features
    # last_pos is already shifted
    # action is velocity
    graph = pyg.data.Data(
        x=node_feature,
        edge_index=edge_index,
        edge_attr=edge_feature,
        last_pos=pos_concat,
        # next_pos = next_pos,
        y=y,
        delta_t=delta_t,
        rd_mag=end_avg_speed
    )
    return graph

class GraphDataset(pyg.data.Dataset):
    def __init__(self, dataset_array, args):
        super().__init__()

        self.dataset=dataset_array
        self.args = args

        try:
            if args.num_fp:
                self.num_fp = args.num_fp
        except:
            self.num_fp = 9

        # normalize_method: "standard" or "minmax" or None
        # self.fp_pose_idx = list(range(0, 18))
        # self.end_pos_idx = list(range(18, 24))
        # self.fp_pos_d_idx = list(range(24, 42))
        # self.end_pos_d_idx = list(range(42, 48))
        #
        # self.end1_xy_idx = list(range(18, 20))
        # self.end1_yaw_idx = 20
        # self.end2_xy_idx = list(range(21, 23))
        # self.end2_yaw_idx = 23
        #
        # # explicit end-effector deltas
        # self.end1_xy_d_idx = list(range(42, 44))  # [42, 43]
        # self.end1_yaw_d_idx = 44
        # self.end2_xy_d_idx = list(range(45, 47))  # [45, 46]
        # self.end2_yaw_d_idx = 47
        #
        # # timestep delta
        # self.delta_t_idx = 48

        self.fp_pose_idx = list(range(0, 2*self.num_fp))
        self.end_pos_idx = list(range(2*self.num_fp, 2*self.num_fp+6))
        self.fp_pos_d_idx = list(range(2*self.num_fp+6, 4*self.num_fp+6))
        self.end_pos_d_idx = list(range(4*self.num_fp+6, 4*self.num_fp+12))

        self.end1_xy_idx = list(range(2*self.num_fp, 2*self.num_fp+2))
        self.end1_yaw_idx = 2*self.num_fp+2
        self.end2_xy_idx = list(range(2*self.num_fp+3, 2*self.num_fp+5))
        self.end2_yaw_idx = 2*self.num_fp+5

        # explicit end-effector deltas
        self.end1_xy_d_idx = list(range(4*self.num_fp+6, 4*self.num_fp+8))  # [42, 43]
        self.end1_yaw_d_idx = 4*self.num_fp+8
        self.end2_xy_d_idx = list(range(4*self.num_fp+9, 4*self.num_fp+11))  # [45, 46]
        self.end2_yaw_d_idx = 4*self.num_fp+11

        # timestep delta
        self.delta_t_idx = 4*self.num_fp+12

        # self.args = args
    # how many windows(data sample) in one dataset
    def len(self):
        return self.dataset.shape[0]

    def extract_data_from_vector(self,data):
        # take X, Xd, R, Rd
        fp_pos = data[self.fp_pose_idx].reshape(self.num_fp, 2)
        end_pos = data[self.end_pos_idx].reshape(2, 3)
        fp_pos_d = data[self.fp_pos_d_idx].reshape(self.num_fp, 2)
        end_pos_d = data[self.end_pos_d_idx].reshape(2,3)

        return fp_pos, fp_pos_d, end_pos, end_pos_d

    def extract_delta_t(self,data):
        return data[self.delta_t_idx]

    def concat_np_pos(self,fp_pos, end_pos):
        # concat end pose and fp pos
        end1_pos_xy, end2_pos_xy = end_pos[0,:2].reshape(1,-1), end_pos[1,:2].reshape(1,-1)
        return np.concatenate((end1_pos_xy,fp_pos,end2_pos_xy),axis=0)

    def concat_tensor_pos(self,fp_pos, end_pos):
        return torch.concat((end_pos[0, :2].unsqueeze(0), fp_pos, end_pos[1, :2].unsqueeze(0)), axis=0)


    def get(self, idx):
        data_vector = self.dataset[idx]
        fp_pos, fp_pos_d, end_pos, end_pos_d = self.extract_data_from_vector(data_vector)
        delta_t = self.extract_delta_t(data_vector)
        # load corresponding data for this time slice
        # with torch.no_grad():
        #     graph = self.get_graph_data_all_miss(fp_pos, fp_pos_d, end_pos, end_pos_d, delta_t)
        # return graph

        try:
            yaw_scale = self.args.yaw_scale
        except:
            yaw_scale = 20

        graph = get_graph_data(fp_pos, fp_pos_d, end_pos, end_pos_d, delta_t, use_yaw_d=self.args.use_yaw_d, yaw_scale=yaw_scale)

        return graph

    def predict_next_pos(self,last_pos,pred_y,scale,delta_t):
        return last_pos+pred_y*scale*delta_t

def draw_points_2D(last_pos, cur_pos, size=(480, 640), x_range=(0, 1), y_range=(0, 1), normalized=True, radius=4, color=(0, 0, 255), thickness=-1, ):
    h, w = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)  # white canvas

    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
    y_span = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

    def draw_points(points, color):
        for x, y in points:
            if normalized:
                # Map x ∈ [x_min, x_max], y ∈ [y_min, y_max] → [0,1]
                xn = (x - x_min) / x_span
                yn = (y - y_min) / y_span

                # Clamp to [0,1]
                xn = 0.0 if xn < 0.0 else (1.0 if xn > 1.0 else xn)
                yn = 0.0 if yn < 0.0 else (1.0 if yn > 1.0 else yn)

                # Convert to pixels; flip y (image origin at top-left)
                px = int(round(xn * (w - 1)))
                py = int(round((1.0 - yn) * (h - 1)))
            else:
                px = int(round(x))
                py = int(round(y))
                # If your system uses origin bottom-left, flip here:
                # py = (h - 1) - py

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), radius, color, thickness)

    # Draw last pose in green
    draw_points(last_pos, (0, 255, 0))
    # Draw current pose in red
    draw_points(cur_pos, (0, 0, 255))

    return img

#process images and original robot data file to get transformed coordinates and saved to data vector
class SaveDataProcessor:
    def __init__(self, robot_data_path, config):

        self.config = config
        self.tf_img_rb1 = np.array(self.config["tf_img_rb1"]) #img --> robot 1
        self.tf_rb2_rb1 = np.array(self.config["tf_rb2_rb1"]) #robot --> robot 1

        with open(robot_data_path, "r") as file:
            lines = file.readlines()
        self.robot_data_lines = lines

        self.robot1_t_idx = 0
        self.robot1_tcp_idx = list(range(13,25))
        self.robot2_t_idx = 25
        self.robot2_tcp_idx = list(range(38,50))

    def transform_point(self,point,transform_matrix):
        return transform(homo_points(point), transform_matrix)

    def extract_frame_robot_data(self, frame_idx):
        # read txt and get data
        row = self.robot_data_lines[frame_idx].strip()  # Remove newline characters
        values = row.split(" ")[1:]  # Split by comma
        values = np.array([float(value) for value in values])

        # t1, 1_q, 1_qd, 1_tcp, 1_tcp_speed, t2, ...
        return values

    def extract_t_tcp_data(self,frame_idx):
        # only tacke time and tcp data
        values = self.extract_frame_robot_data(frame_idx)
        # extract from one row
        end1_t = values[self.robot1_t_idx]
        end1_tcp = values[self.robot1_tcp_idx]
        end2_t = values[self.robot2_t_idx]
        end2_tcp = values[self.robot2_tcp_idx]

        # shape: 1, 12, 1, 12
        return end1_t, end1_tcp, end2_t, end2_tcp

    def calculate_transformed_R_2D (self, end1_tcp, end2_tcp):
        end1_xy = end1_tcp[:2].reshape(1,-1)
        end2_xy = end2_tcp[:2].reshape(1,-1)
        end2_xy_transformed = self.transform_point(end2_xy, self.tf_rb2_rb1)

        # Rotation vector to RPY
        end1_rpy = end1_tcp[3:6]
        end2_rpy = end2_tcp[3:6]

        end1_yaw = (Rotation.from_rotvec(end1_rpy).as_euler('xyz', degrees=False))[2]
        end2_yaw = (Rotation.from_rotvec(end2_rpy).as_euler('xyz', degrees=False))[2]

        end1_pos = np.concatenate((end1_xy,end1_yaw.reshape(1,1)),axis=1)
        end2_pos = np.concatenate((end2_xy_transformed,end2_yaw.reshape(1,1)),axis=1)

        return np.concatenate((end1_pos,end2_pos),axis=0)

    def calculate_transformed_X_2D(self, fp_pixel):
        # convert pixel to robot 1 frame
        return self.transform_point(fp_pixel, self.tf_img_rb1)

    def calculate_delta (self, pos1, pos2):
        if pos2 is None:
            return np.zeros(pos1.shape)
        return pos1-pos2

    def concate_all_pose(self,fp_pos, end_pos):
        # concat end pose and fp pos
        end1_pos_xy, end2_pos_xy = end_pos[0,:2].reshape(1,-1), end_pos[1,:2].reshape(1,-1)
        return np.concatenate((end1_pos_xy,fp_pos,end2_pos_xy),axis=0)

    def generate_data_vector(self, last_fp_pos, last_end_pos, delta_fp_pos, delta_end_pos, delta_t):
        return np.concatenate([
            last_fp_pos.flatten(),
            last_end_pos.flatten(),
            delta_fp_pos.flatten(),
            delta_end_pos.flatten(),
            np.array([delta_t])
        ])

    def calculate_delta_R(self, frame_idx, interval=1): # just to check, in the data processing, just use calculate delta
        end1_t_cur, end1_tcp_cur, end2_t_cur, end2_tcp_cur = self.extract_t_tcp_data(frame_idx)
        if frame_idx-interval<0:
            return np.zeros(2,3)
        end1_t_last, end1_tcp_last, end2_t_last, end2_tcp_last = self.extract_t_tcp_data(frame_idx-interval)
        return self.calculate_transformed_R_2D(end1_tcp_cur, end2_tcp_cur)-self.calculate_transformed_R_2D(end1_tcp_last, end2_tcp_last)

    def draw_points_cv2(self, last_pos, cur_pos):
        return draw_points_2D(last_pos, cur_pos)