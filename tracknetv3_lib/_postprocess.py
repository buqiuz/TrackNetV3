import math
import cv2
import numpy as np
import pandas as pd
import torch
from collections import deque
from PIL import Image, ImageDraw
from torch import is_tensor

from ._preprocess import HEIGHT, WIDTH


def to_img(image):
    """ Convert the normalized image back to image format.

        Args:
            image (numpy.ndarray): Images with range in [0, 1]

        Returns:
            image (numpy.ndarray): Images with range in [0, 255]
    """
    image = image * 255
    image = image.astype('uint8')
    return image


def to_img_format(input, num_ch=1):
    """ Transform model input sequence format to image sequence format.

        Args:
            input (numpy.ndarray): model input with shape (N, L, H, W)
            num_ch (int): Number of channels of each frame

        Returns:
            (numpy.ndarray): Image sequences with shape (N, L, H, W)
    """
    assert len(input.shape) == 4, 'Input must be 4D tensor.'
    if num_ch == 1:
        return input
    else:
        raise NotImplementedError('to_img_format only supports num_ch=1 for inference.')


def get_ensemble_weight(seq_len, eval_mode):
    """ Get weight for temporal ensemble.

        Args:
            seq_len (int): Length of input sequence
            eval_mode (str): Mode of temporal ensemble
                Choices:
                    - 'average': Return uniform weight
                    - 'weight': Return positional weight

        Returns:
            weight (torch.Tensor): Weight for temporal ensemble
    """
    import torch
    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = (i + 1)
            weight[seq_len - i - 1] = (i + 1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')

    return weight


def predict_location(heatmap):
    """ Get coordinates from the heatmap.

        Args:
            heatmap (numpy.ndarray): A single heatmap with shape (H, W)

        Returns:
            x, y, w, h (Tuple[int, int, int, int]): bounding box of the bounding box with max area
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0, 0, 0
    else:
        # Find all respond area in the heatmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area among all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        return x, y, w, h


def generate_inpaint_mask(pred_dict, th_h=30):
    """ Generate inpaint mask from the predicted trajectory.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
            th_h (float): Height threshold (pixels) for y coordinate

        Returns:
            inpaint_mask (List): Inpaint mask
    """
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0 # index that ball start to disappear
    j = 0 # index that ball start to appear
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i - 1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view
            pass
        i = j

    return inpaint_mask.tolist()


def extract_predictions(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates.

        Port of predict.predict().

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if is_tensor(indices) else indices.numpy()

    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)

    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1])
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                    cx_pred, cy_pred = int(cx_pred * img_scaler[0]), int(cy_pred * img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break

    return pred_dict


def write_pred_csv(pred_dict, save_file):
    """ Write prediction result to csv file.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
            save_file (str): File path of the output csv file

        Returns:
            None
    """
    pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                            'Visibility': pred_dict['Visibility'],
                            'X': pred_dict['X'],
                            'Y': pred_dict['Y']})
    pred_df.to_csv(save_file, index=False)


def draw_traj(img, traj, radius=3, color='red'):
    """ Draw trajectory on the image.

        Args:
            img (numpy.ndarray): Image with shape (H, W, C)
            traj (deque): Trajectory to draw
            radius (int, optional): Radius of the ball dot
            color (str, optional): Color of the trajectory outline

        Returns:
            img (numpy.ndarray): Image with trajectory drawn
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox = (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


def write_pred_video(video_file, pred_dict, save_file, traj_len=8):
    """ Write a video with prediction result.

        Args:
            video_file (str): File path of the input video file
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
            save_file (str): File path of the output video file
            traj_len (int, optional): Length of trajectory to draw

        Returns:
            None
    """
    # Read video
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']

    # Video config
    out = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    # Create a queue for storing trajectory
    pred_queue = deque()

    # Draw prediction trajectory
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()

        # Push ball coordinates for each frame
        pred_queue.appendleft([x_pred[i], y_pred[i]]) if vis_pred[i] else pred_queue.appendleft(None)

        # Draw prediction trajectory
        frame = draw_traj(frame, pred_queue, color='yellow')

        out.write(frame)
        i += 1

    out.release()
    cap.release()
