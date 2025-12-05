import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from data_utils import GraphDataset, get_graph_data, concat_np_pos, to_pixels
from utils.state_index import I
import torch
import cv2

def load_rollout(npz_path):
    data = np.load(npz_path)
    observed_xy = data["observed_xy"]   # (T, N, 2)
    true_xy = data["true_xy"]           # (T, N, 2)
    missing_mask = data["missing_mask"] # (T, N)
    frames = data["frames"]             # (T,)
    return observed_xy, true_xy, missing_mask, frames

def plot_single_frame(true_xy_frame, obs_xy_frame, missing_mask_frame,
                      rollout_idx, frame_idx, save_dir=None):
    """
    true_xy_frame: (N,2)
    obs_xy_frame : (N,2)
    missing_mask_frame: (N,)
    """

    # Optionally pick first/last frame, here we assume you pass correct frame
    x_true, y_true = true_xy_frame[:, 0], true_xy_frame[:, 1]
    x_obs, y_obs   = obs_xy_frame[:, 0],  obs_xy_frame[:, 1]

    plt.figure(figsize=(4, 4), dpi=300)

    # Ground truth rope (blue, with line)
    plt.plot(x_true, y_true, "-o", label="True (Real State)", linewidth=1)

    # Observed/predicted rope (red)
    plt.plot(x_obs, y_obs, "-o", label="Observed/Predicted", linewidth=1)

    # Highlight missing points if you like (e.g. green crosses)
    miss_idx = np.where(missing_mask_frame)[0]
    if len(miss_idx) > 0:
        plt.scatter(x_obs[miss_idx], y_obs[miss_idx],
                    marker="x", s=30, label="Missing (Occluded) FPs")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Rollout {rollout_idx}, Frame {frame_idx}")
    plt.legend()
    plt.axis("equal")  # keep rope shape aspect ratio

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir,
                                f"rollout{rollout_idx}_frame{frame_idx}.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure to: {out_path}")

    plt.close()

def main():
    setting = "Paper_results"   # or args.setting
    tag = "offline"          # or args.tag
    num_fp = 8

    npz_dir = f"./results/{setting}/saved_DLO_prediction/{tag}"
    files = sorted(glob.glob(os.path.join(npz_dir, "fp_tracking_rollout*.npz")))
    print(files)
    if not files:
        raise SystemExit(f"No NPZ files found in {npz_dir}")

    print(f"Found {len(files)} rollout files.")

    for rollout_idx, fpath in enumerate(files):
        out_dir = f"./results/{setting}/figures_{tag}/{rollout_idx}"
        observed_xy, true_xy, missing_mask, frames = load_rollout(fpath)

        state_all = np.load("/home/xinge/shape_control_DLO_2/results/sim/control/ours/"+ setting + "/" + tag + "/state_" + str(
                rollout_idx) + ".npy")

        # desired_positions = state[-1, I.desired_pos_idx]
        # positions = state[:, I.fps_pos_idx]

        total_frame = len(frames)

        for frame_idx in range(total_frame):
            state = state_all[frame_idx]
            state_input = state[I.state_input_idx]
            # fp_pos = state_input[3:27].reshape(num_fp, 3)  # ground truth

            fp_pos = true_xy[frame_idx]

            observed_fp_pos = observed_xy[frame_idx]
            target_pos = state[I.desired_pos_idx].reshape(num_fp+2, 3)[:,:2]


            # Get midpoint between points 4 and 5
            mid_point = (target_pos[4, :2] + target_pos[5, :2]) / 2.0

            # Fixed rectangle size
            rect_width = 0.15
            rect_height = 0.15

            # Compute bounds centered at midpoint
            x0 = mid_point[0] - rect_width / 2
            x1 = mid_point[0] + rect_width / 2
            y0 = mid_point[1] - rect_height / 2
            y1 = mid_point[1] + rect_height / 2

            print(f"Rect bounds: x=[{x0:.3f}, {x1:.3f}], y=[{y0:.3f}, {y1:.3f}]")


            # 3. Check which fp_pos points fall inside the rectangle
            inside_mask = (
                    (fp_pos[:, 0] >= x0) & (fp_pos[:, 0] <= x1) &
                    (fp_pos[:, 1] >= y0) & (fp_pos[:, 1] <= y1)
            )

            # 4. Get indices of the covered / missing points
            missing_idx = np.where(inside_mask)[0]

            dropout_fp_idx = list(missing_idx)

            # dropout_fp_idx = []
            # print("missing_idx", dropout_fp_idx)

            lst = [0,1,2,3,4,5,6,7]
            see_fp_idx = [x for x in lst if x not in dropout_fp_idx]

            end_pos_data =  state_input[30:].reshape(2, 7)[:, [0, 1, 5]] # x,y,_ the last number is meaningless in 2D, just to make size consistent

            #ground truth pos
            # x_true, y_true = to_pixels(torch.tensor(concat_np_pos(fp_pos[:,:2], end_pos_data[:,:2])[1:-1, :2]), 1920,1080, x_range=(-1, 1), y_range=(-1, 1))
            x_true, y_true = to_pixels(torch.tensor(concat_np_pos(fp_pos[:,:2], end_pos_data[:,:2])), 1920,1080, x_range=(-1, 1), y_range=(-1, 1))
            x_target, y_target = to_pixels(torch.tensor(target_pos)[1:-1,:2], 1920,1080,
                                       x_range=(-1, 1), y_range=(-1, 1))

            #--------------------------------------- Visualize ----------------------- #


            x_miss, y_miss = to_pixels(torch.tensor(concat_np_pos(observed_fp_pos[:,:2], end_pos_data[:,:2])), 1920,1080,
                                       x_range=(-1, 1), y_range=(-1, 1))
            image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            overlay = image.copy()
            # Draw blue circles and lines (x_true, y_true)
            points_true = []
            for x, y in zip(x_true, y_true):
                pt = (int(x), int(y))
                cv2.circle(image, pt, 10, (255, 0, 0), -1)  # Blue
                points_true.append(pt)
            for i in range(1, len(points_true)):
                cv2.line(image, points_true[i - 1], points_true[i], (255, 0, 0), 2)

            # Blend overlay with original image (alpha = 0.5)
            alpha = 0.4  # 20% transparency
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Draw red circles and lines (x_miss, y_miss)
            # Draw red circles and lines only for points in drop_idx
            try:
                points_miss = []
                new_lst = [i+1 for i in dropout_fp_idx]
                for i, (x, y) in enumerate(zip(x_miss, y_miss)):
                    # if i not in new_lst:  # skip points not in drop_idx
                    #     continue

                    if i == 0 or i == num_fp+1:
                        radius = 15  # big
                    else:
                        radius = 10  # normal

                    pt = (int(x), int(y))
                    cv2.circle(image, pt, radius, (0, 0, 255), -1)  # Red
                    points_miss.append(pt)

                # Draw connecting lines only for the drawn points
                for i in range(1, len(points_miss)):
                    cv2.line(image, points_miss[i - 1], points_miss[i], (0, 0, 255), 2)

            except Exception as e:
                print("Error drawing points_miss:", e)

            # Draw green circles and lines (target)
            # points_target = []
            # for x, y in zip(x_target, y_target):
            #     pt = (int(x), int(y))
            #     cv2.circle(image, pt, 10, (0, 255, 0), -1)  # Green
            #     points_target.append(pt)
            # for i in range(1, len(points_target)):
            #     cv2.line(image, points_target[i - 1], points_target[i], (0, 255, 0), 2)

            # Draw green rectangles and connecting lines (target)
            points_target = []
            rect_w, rect_h = 10, 10  # rectangle width and height

            for x, y in zip(x_target, y_target):
                cx, cy = int(x), int(y)

                # Rectangle coordinates (centered at point)
                x1_tar = cx - rect_w // 2
                y1_tar = cy - rect_h // 2
                x2_tar = cx + rect_w // 2
                y2_tar = cy + rect_h // 2

                # cv2.rectangle(image, (x1_tar, y1_tar), (x2_tar, y2_tar), (0, 255, 0), -1)  # filled green rectangle

                radius = 10
                cv2.circle(image, (cx,cy), radius, (0, 255, 0), -1)  # Red

                points_target.append((cx, cy))

            # Draw connecting lines
            for i in range(1, len(points_target)):
                cv2.line(image, points_target[i - 1], points_target[i], (0, 255, 0), 2)

            # Draw the occluded rectangle (green border)
            # x_rect_pix, y_rect_pix = to_pixels(np.array([[x0, y0], [x1, y1]]), 1920,1080, x_range=(-1, 1),y_range=(-1, 1))
            # pt1 = tuple(np.array([x_rect_pix[0], y_rect_pix[0]]).astype(int))
            # pt2 = tuple(np.array([x_rect_pix[1], y_rect_pix[1]]).astype(int))
            # cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

            # Convert rectangle endpoints
            x_rect_pix, y_rect_pix = to_pixels(
                np.array([[x0, y0], [x1, y1]]),
                1920, 1080,
                x_range=(-1, 1), y_range=(-1, 1)
            )

            pt1 = tuple(np.array([x_rect_pix[0], y_rect_pix[0]]).astype(int))
            pt2 = tuple(np.array([x_rect_pix[1], y_rect_pix[1]]).astype(int))

            # --- Draw shaded rectangle ---
            overlay = image.copy()

            # Filled rectangle on overlay (green-ish)
            shade_color = (0, 255, 0)  # green
            cv2.rectangle(overlay, pt1, pt2, shade_color, -1)

            # Blend with transparency
            alpha = 0.3  # 30% opacity, adjust as needed
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # --- Optional: draw border line on top ---
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)


            cv2.putText(image, "Green: Target", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, "Blue: Real State", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(image, "Red: Prediction", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Text to be written on each frame
            text = f"Rollout {rollout_idx} - Frame {frame_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Put the text on the frame
            cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # out_rollout.write(image)
            # # Show video as it writes
            # cv2.imshow("Rollout", image)

            # save every image
            save_path = os.path.join(out_dir, f"{frame_idx}.png")
            highres_image = cv2.resize(
                image,
                None,
                fx=2.0, fy=2.0,  # 2× resolution
                interpolation=cv2.INTER_CUBIC  # smoother and clearer
            )

            cv2.imwrite(save_path, highres_image)

            # # Example: pick the last frame of this rollout
            # t = -1  # you can also pick np.argmax(...) based on error, etc.
            # true_xy_frame = true_xy[t]
            # obs_xy_frame  = observed_xy[t]
            # mask_frame    = missing_mask[t]
            # frame_idx     = int(frames[t])
            #
            # plot_single_frame(true_xy_frame,
            #                   obs_xy_frame,
            #                   mask_frame,
            #                   rollout_idx=rollout_idx,
            #                   frame_idx=frame_idx,
            #                   save_dir=out_dir)

if __name__ == "__main__":
    main()
