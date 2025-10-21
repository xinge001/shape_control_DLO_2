#!/usr/bin/env python

# our proposed controller

import numpy as np
from matplotlib import pyplot as plt
import time
import sys, os

import rospy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as sciR

import copy

from GNN_no_ros_new import JacobianPredictor
from utils.state_index import I
import torch


from torch_geometric.data import Data, Batch
import torch_geometric as pyg
from torch_geometric.nn import radius_graph
from GN_model_DLO import MySubEdge, MySubNode
from torch_geometric.nn import global_mean_pool


params_end_vel_max = 0.1
params_normalized_error_thres = 0.2/8
params_control_gain = 1.0
params_lambda_weight = 0.1
params_over_stretch_cos_angle_thres = 0.998

def get_graph_data(state_input, length, numFPs):
    # state_input dim is [1,44]
    left_end_pos = state_input[:, 3*numFPs : 3*numFPs + 3]
    left_end_quat = state_input[:,  3*numFPs + 3 : 3*numFPs +7]
    right_end_pos = state_input[:,  3*numFPs + 7 : 3*numFPs +10]
    right_end_quat = state_input[:,  3*numFPs +10 : 3*numFPs +14]
    fps_pos = state_input[:, 0 : 3*numFPs].reshape(numFPs, 3)

    # node_input dim is 7
    # [x,y,z,angle pos]
    # angle_pos is [0,0,0,0] for fps, and [q1, q2, q3, q4] for ends
    left_end_state = torch.concat((left_end_pos,left_end_quat),dim=-1)
    right_end_state = torch.concat((right_end_pos,right_end_quat),dim=-1)
    fps_state = torch.concat((fps_pos,torch.zeros(numFPs,4)),dim=-1)

    print("full state shape", fps_state.shape)

    # fps_state = fps_state[[1,3,5,7],:]
    #
    # print("partial state shape", fps_state.shape)

    node_input = torch.concat((left_end_state,fps_state,right_end_state),dim=0)

    # dim = [num_fps + 2, 3]
    all_pos_xyz = node_input[:,:3]

    # edge_index depends on redius
    edge_index = radius_graph(all_pos_xyz, r=length/3, loop=False)

    # edge feature input, relative distance
    edge_input = all_pos_xyz[edge_index[0]]-all_pos_xyz[edge_index[1]]

    # return the graph with features
    graph = pyg.data.Data(
        x=node_input.double(),
        edge_index=edge_index,
        edge_attr=edge_input.double(),
        all_pos_xyz = all_pos_xyz.double()
    )

    return graph

def get_graph_batch(state_input_tensor, length_tensor, numFPs=10):
    # get a list of graph
    return [get_graph_data(state_input_tensor[i:i+1,:], length_tensor[i], numFPs) for i in range(state_input_tensor.shape[0])]

class Controller(object):
    # --------------------------------------------------------------------
    def __init__(self):
        self.numFPs = rospy.get_param("DLO/num_FPs")
        self.env_dim = rospy.get_param("env/dimension")
        self.env = rospy.get_param("env/sim_or_real")
        self.bEnableEndRotation = rospy.get_param("controller/enable_end_rotation")
        self.b_left_arm = rospy.get_param("controller/enable_left_arm")
        self.b_right_arm = rospy.get_param("controller/enable_right_arm")
        self.targetFPsIdx = rospy.get_param("controller/object_fps_idx")
        self.project_dir = rospy.get_param("project_dir")
        self.offline_model_name = rospy.get_param("controller/offline_model")
        self.control_law = rospy.get_param("controller/control_law")
        self.controlRate = rospy.get_param("ros_rate/env_rate")

        # the non-zero dimension of the control input
        # e.g. for 2D is [0,1,5,6,7,11] just x,y,theta
        self.validJacoDim = self.getValidControlInputDim(self.env_dim, self.bEnableEndRotation, self.b_left_arm, self.b_right_arm)

        self.jacobianPredictor = JacobianPredictor()
        self.jacobianPredictor.LoadModelWeights() #default weights is /model/refWeights/2D/10*6_backup

        self.k = 0
        self.case_idx = 0
        self.state_save = []



    # -------------------------------------------------------------------- Try to avoid large
    # If the error is too large, Normalize the vector (unit vector): task_error / norm
    # Scale it to the threshold length: * thres
    def normalizeTaskError(self, task_error):
        norm =  np.linalg.norm(task_error)
        thres = params_normalized_error_thres * len(self.targetFPsIdx) #0.2
        if norm <= thres:
            return task_error
        else:
            return task_error / norm * thres


    # --------------------------------------------------------------------
    def optimizeControlInput(self, fps_vel_ref, J, lambd, v_max, C1, C2):
        fps_vel_ref = fps_vel_ref.reshape(-1, 1)

        def objectFunc(v):
            v = v.reshape(-1, 1)
            cost = 1/2 * (fps_vel_ref - J @ v).T @ (fps_vel_ref - J @ v) + lambd/2 * v.T @ v
            return cost[0, 0]

        def quad_inequation(v): # >= 0
            v = v.reshape(-1, 1)
            ieq = (v.T @ v) - v_max**2
            return -ieq.reshape(-1, )

        def linear_inequation(v): # >= 0
            v = v.reshape(-1, 1)
            ieq = C1 @ v
            return -ieq.reshape(-1, )

        def linear_equation(v):
            v = v.reshape(-1, 1)
            eq = C2 @ v
            return eq.reshape(-1, )

        def jacobian(v):
            v = v.reshape(-1, 1)
            jaco = - J.T @ fps_vel_ref   +   (J.T @ J + lambd * np.eye(J.shape[1])) @ v
            return jaco.reshape(-1, )

        quad_ineq = dict(type='ineq', fun=quad_inequation)
        constraints_list = [quad_ineq]
        if np.any(C1 != 0):
            linear_ineq = dict(type='ineq', fun=linear_inequation)
            constraints_list.append(linear_ineq)
        if np.any(C2 != 0):
            linear_eq = dict(type='eq', fun=linear_equation)
            constraints_list.append(linear_eq)

        v_init = np.zeros((J.shape[1], 1))
        res = minimize(objectFunc, v_init, method='SLSQP', jac=jacobian, constraints=constraints_list, options={'ftol':1e-10})

        return res.x.reshape(-1, 1)



    # --------------------------------------------------------------------
    def generateControlInput(self, state):
        self.state_save.append(copy.deepcopy(state))

        fpsPositions = state[I.fps_pos_idx]
        desiredPositions = state[I.desired_pos_idx]

        # [num_fps,3] e.g. 10*3
        full_task_error = np.zeros((self.numFPs, 3), dtype='float32')
        full_task_error[self.targetFPsIdx, :] = np.array(fpsPositions - desiredPositions).reshape(self.numFPs, 3)[self.targetFPsIdx, :]

        # 8*3, error is [24,1] 8 controlable points
        # self.targetFPsIdx = [1,2,3,4,5,6,7,8]
        target_task_error = np.zeros((3 * len(self.targetFPsIdx), 1))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_task_error[3*i : 3*i +3, :] = full_task_error[targetIdx, :].reshape(3, 1)

        normalized_target_task_error = self.normalizeTaskError(target_task_error) #[24,1]

        # calcualte the current Jacobian, and do the online learning
        # Jacobian = self.jacobianPredictor.OnlineLearningAndPredictJ(state, self.normalizeTaskError(full_task_error.reshape(-1, ))) # [30:1]

        length = state[I.length_idx]
        state_input = state[I.state_input_idx].reshape(1, -1) # one row matrix [1,44]

        # length_torch = torch.tensor(length).cuda()
        # state_input_torch = self.jacobianPredictor.relativeStateRepresentationTorch(torch.tensor(state_input)).cuda()
        # J_pred = self.jacobianPredictor.model_J(state_input_torch)
        # J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)  # no normalize 要改 # RBF_abs 要改
        # Jacobian = J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)
        # print(Jacobian.shape)

        length = torch.tensor(length)
        state_input = torch.tensor(state_input)

        # construct graph

        graph_lst = get_graph_batch(state_input, length, numFPs=self.numFPs)

        graph_batch = Batch.from_data_list(graph_lst)

        graph_batch = graph_batch.cuda()

        length = length.cuda()
        state_input = state_input.cuda()

        # J_pred = self.model_J(state_input)


        self.jacobianPredictor.model_J = self.jacobianPredictor.model_J.double()

        J_pred = torch.reshape(torch.reshape(self.jacobianPredictor.model_J(graph_batch), (-1, self.numFPs, 12, 3)).transpose(2, 3),
                               (-1, 3 * self.numFPs, 12))
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)
        Jacobian = J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)
        print("Graph Jacobian", Jacobian.shape)


        # get the target Jacobian
        # [24,6]
        target_J = np.zeros((3 * len(self.targetFPsIdx), len(self.validJacoDim)))
        for i, targetIdx in enumerate(self.targetFPsIdx):
            target_J[3*i : 3*i+3, :] = Jacobian[3*targetIdx : 3*targetIdx+3, self.validJacoDim]

        # calculate the ideal target point velocity
        alpha = params_control_gain
        fps_vel_ref = - alpha * normalized_target_task_error

        lambd = params_lambda_weight * np.linalg.norm(normalized_target_task_error)
        v_max = params_end_vel_max

        # get the matrix of the constraints for avoiding over-stretching
        C1, C2 = self.validStateConstraintMatrix(state)

        # calcuate the control input by solving the convex optmization problem
        u = self.optimizeControlInput(fps_vel_ref, target_J, lambd, v_max, C1, C2)

        u_12DoF = np.zeros((12, ))
        u_12DoF[self.validJacoDim] = u.reshape(-1, )

        # ensure safety
        if np.linalg.norm(u_12DoF) > v_max:
            u_12DoF = u_12DoF / np.linalg.norm(u_12DoF) * v_max

        self.k += 1

        return u_12DoF


    # --------------------------------------------------------------------
    def validStateConstraintMatrix(self, state):
        state = np.array(state)
        left_end_pos = state[I.left_end_pos_idx]
        right_end_pos = state[I.right_end_pos_idx]

        if self.env_dim == '3D':
            fps_pos = state[I.fps_pos_idx].reshape(self.numFPs, 3)
            left_end_pos = state[I.left_end_pos_idx]
            right_end_pos = state[I.right_end_pos_idx]
            C1 = np.zeros((1, 12))
            C2 = np.zeros((6, 12))
        elif self.env_dim == '2D':
            fps_pos = state[I.fps_pos_idx].reshape(self.numFPs, 3)[:, 0:2]
            left_end_pos = state[I.left_end_pos_idx][0:2]
            right_end_pos = state[I.right_end_pos_idx][0:2]
            C1 = np.zeros((1, 6))
            C2 = np.zeros((2, 6))

        # decide whether the current state is near over-stretched
        b_over_stretch = False
        segments = fps_pos.copy()
        segments[1:, :] = (fps_pos[1:, :] - fps_pos[0:-1, :]) 
        cos_angles = np.ones((self.numFPs - 2, ))
        for i in range(2, self.numFPs - 2):
            cos_angles[i-1] = np.dot(segments[i, :], segments[i+1, :]) / (np.linalg.norm(segments[i, :]) * np.linalg.norm(segments[i+1, :]))

        ends_distance =  (right_end_pos - left_end_pos).reshape(-1, 1)
        if np.all(cos_angles > params_over_stretch_cos_angle_thres):
            b_over_stretch = True

        # calculate the C1 and C2 matrix
        if b_over_stretch:  
            pd =  ends_distance
            if self.env_dim == '3D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 3)), pd.T, np.zeros((1, 3))], axis=1)
                C2_1 = np.concatenate([np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))], axis=1)
                C2_2 = np.concatenate([np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            elif self.env_dim == '2D':
                C1 = np.concatenate([-pd.T, np.zeros((1, 1)), pd.T, np.zeros((1, 1))], axis=1)
                C2_1 = np.concatenate([np.zeros((1, 2)), np.eye(1), np.zeros((1, 2)), np.zeros((1, 1))], axis=1)
                C2_2 = np.concatenate([np.zeros((1, 2)), np.zeros((1, 1)), np.zeros((1, 2)), np.eye(1)], axis=1)
                C2 = np.concatenate([C2_1, C2_2], axis=0)
            return C1, C2
        else:
            return C1, C2


    # --------------------------------------------------------------------
    def reset(self, state):
        self.state_save.append(state)
        
        result_dir = self.project_dir + "results/" + self.env + "/control/" + self.control_law + "/" + self.env_dim + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        np.save(result_dir + "state_" + str(self.case_idx) + ".npy", self.state_save)
        
        self.case_idx += 1
        self.state_save = []
        self.k = 0

        if (self.case_idx == 100):
            rospy.signal_shutdown("finish.")

        self.jacobianPredictor.LoadModelWeights()


    # --------------------------------------------------------------------
    def getValidControlInputDim(self, env_dim, bEnableEndRotation, b_left_arm, b_right_arm):
        if env_dim == '2D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 5, 6, 7, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 6, 7]
                elif b_left_arm:
                    validJacoDim = [0, 1]
                elif b_right_arm:
                    validJacoDim = [6, 7]
                else:
                    validJacoDim = np.empty()
        elif env_dim == '3D':
            if bEnableEndRotation:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2, 3, 4, 5]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8, 9, 10, 11]
                else:
                    validJacoDim = np.empty()
            else:
                if b_left_arm and b_right_arm:
                    validJacoDim = [0, 1, 2, 6, 7, 8]
                elif b_left_arm:
                    validJacoDim = [0, 1, 2]
                elif b_right_arm:
                    validJacoDim = [6, 7, 8]
                else:
                    validJacoDim = np.empty()
        else:
            print("Error: the environment dimension must be '2D' or '3D'.")

        return validJacoDim
