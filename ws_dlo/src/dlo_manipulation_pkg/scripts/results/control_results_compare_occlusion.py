#!/usr/bin/env python

# compare the control results of methods
# load the control results and calcualte the success rate, average task time, averaget task error

import os, sys
parrentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parrentdir)

import numpy as np
from matplotlib import pyplot as plt
import rospy
from utils.state_index import I


project_dir = rospy.get_param("project_dir")
num_fps = rospy.get_param("DLO/num_FPs")
env = rospy.get_param("env/sim_or_real")
target_points_idx = rospy.get_param("controller/object_fps_idx")


# ------------------------------------------------------------------------------
def evaluateControlResults(names, num_case=50, delta_t=0.1):

    env_dim = rospy.get_param("env/dimension")
    
    threshold = 0.10
    if env == 'sim':
        sequence_length = 30 * 10
        control_rate = 10

    target_dim = []
    for target_point_idx in target_points_idx:
        target_dim += [3*target_point_idx, 3*target_point_idx+1, 3*target_point_idx+2]

    all_success = np.zeros((num_case, len(names)))
    k = 0
    total_count = 0
    flip_rollout = []
    for name in names:
        task_error_success = []
        task_error_all = []
        task_time = []
        success = 0
        for i in range(num_case):
            # [2,7,11,15,18,25,28,33,39,41]
            # [2, 10, 11, 15, 24, 25, 27, 29, 31, 32, 33, 34, 39, 42, 43, 44, 48, 51, 54, 55, 60, 65, 71, 72, 74, 80, 82, 86, 89, 90, 92, 95, 96, 98]
            # [2, 3, 5, 7, 8, 9, 10, 11, 15, 16, 18, 20, 24, 25, 26, 27, 29, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45,48]
            # if i in [2, 10, 11, 15, 24, 25, 27, 29, 31, 32, 33, 34, 39, 42, 43, 44, 48, 51, 54, 55, 60, 65, 71, 72, 74, 80, 82, 86, 89, 90, 92, 95, 96, 98]:
            #     continue
            if env == 'sim':
                # state = np.load(project_dir + "results/" + env + "/control/" + name + "/" + env_dim  + "/state_" + str(i) + ".npy")
                state = np.load(
                    project_dir + "results/" + env + "/control/" + name + "/No_correct_15" + "/autoRate" + "/state_" + str(i) + ".npy")

            desired_positions = state[-1, I.desired_pos_idx] 
            positions = state[:, I.fps_pos_idx]

            # fp_true_vel = state[:, 48:72].reshape(-1,8,3)
            # print("max_vel", np.nanmax(fp_true_vel))
            #
            # if np.nanmax(fp_true_vel) > 0.3 and i not in flip_rollout:
            #     flip_rollout.append(i)
            # print("flip_rollout", flip_rollout)

            error = np.linalg.norm((positions - desired_positions)[:, target_dim], axis=1)

            # if doesn't overstretch and the final error is less than the threshold
            # if(state.shape[0] >= sequence_length - control_rate and np.all(error[-control_rate : -1] < threshold)):
            # if np.all(error[-control_rate:-1] < threshold):
            if np.all(error[-2:-1] < threshold):
                # if env == 'sim' and env_dim == '2D' and np.any(positions.reshape(-1, 10, 3)[1:, :, 2] > 0.005): # 2D if the DLO left the table
                #     continue
                success += 1
                print("success: ", i)
                all_success[i, k] = 1
                time = np.min(np.where(error < threshold)) * delta_t
                task_time.append(time)
                task_error_success.append(np.mean(error[-2 : -1]))

            # offline
            # [2, 3, 7, 9, 10, 11, 15, 18, 24, 25, 27, 28, 29, 31, 32, 33, 34, 38, 40, 41, 42, 43, 44, 45, 48]
            # autoRate
            # [2, 3, 5, 9, 10, 11, 15, 18, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 48]
            # Adam_0.1 [2, 3, 9, 10, 11, 15, 18, 24, 25, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48]
            # adam_1e4 [2, 3, 9, 10, 11, 13, 15, 18, 24, 25, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 44, 45, 48]
            # [2, 3, 5, 9, 10, 11, 13, 15, 18, 24, 25, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 48, 51, 54, 55, 58, 60, 64, 65, 71, 72, 74, 80, 82, 83, 86, 87, 89, 95, 96, 97, 98]
            # [2, 3, 7, 9, 10, 11, 15, 18, 24, 25, 27, 28, 29, 31, 32, 33, 34, 38, 40, 41, 42, 43, 44, 45, 48, 51, 54, 55, 60, 64, 71, 72, 74, 80, 82, 83, 86, 87, 89, 95, 96, 97, 98]
            # [2, 10, 11, 15, 24, 25, 27, 29, 31, 32, 33, 34, 39, 42, 43, 44, 48, 51, 54, 55, 60, 65, 71, 72, 74, 80, 82, 86, 89, 90, 92, 95, 96, 98]
            else:
                print("unsuccessful", i)
            total_count += 1

            # for all cases (not just successful cases)
            task_error_all.append( np.mean(error[-2 : -1]))

        if task_error_success == []:
              ave_task_error_success = 0
              ave_task_time = 0  
        else:
            ave_task_error_success = np.mean(np.array(task_error_success))
            ave_task_time = np.mean(np.array(task_time))
        ave_task_error_all = np.mean(np.array(task_error_all))
        
        print(name, " Success: ", success, ", Task time (s): ", ave_task_time, ", Success  Task error  (cm): ", ave_task_error_success * 100, ", All Task error  (cm): ", ave_task_error_all * 100)

        ave_result = np.array([success,  ave_task_time, ave_task_error_success * 100, ave_task_error_all * 100])
        np.save(project_dir + "results/" + env + "/control/" + name + "/" + env_dim + "/ave.npy", ave_result)

        k += 1
        print("total case", total_count)



# -------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    methods = ['ours']
    evaluateControlResults(methods)