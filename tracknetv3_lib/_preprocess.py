import cv2
import numpy as np
from PIL import Image

HEIGHT = 288
WIDTH = 512


def generate_frames(video_file):
    """ Sample all frames from the video.

        Args:
            video_file (str): File path of the video file

        Returns:
            frame_list (List[numpy.ndarray]): List of sampled frames in BGR order
    """
    assert video_file[-4:] == '.mp4', 'Invalid video file format.'

    cap = cv2.VideoCapture(video_file)
    frame_list = []
    success = True
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)
    cap.release()
    return frame_list


def get_video_info(video_file):
    """ Return the video properties.

        Args:
            video_file (str): File path of the video file

        Returns:
            info (Dict): {'w': int, 'h': int, 'fps': int, 'video_len': int, 'fourcc': int}
    """
    cap = cv2.VideoCapture(video_file)
    info = {'w': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'h': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'video_len': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))}
    cap.release()
    return info


def gen_median_from_all(frame_arr, bg_mode):
    """ Generate the median background image from the full frame array.

        Port of the median initialization in Shuttlecock_Trajectory_Dataset.

        Args:
            frame_arr (numpy.ndarray): Frame sequence with shape (N, H, W, 3) in RGB order
            bg_mode (str): Background mode ('', 'subtract', 'subtract_concat' or 'concat')

        Returns:
            median (numpy.ndarray): Background image
                'concat': (3, HEIGHT, WIDTH) resized
                'subtract' / 'subtract_concat': (H, W, 3) full resolution
                '': None
    """
    if not bg_mode:
        return None
    median = np.median(frame_arr, 0)
    if bg_mode == 'concat':
        median = Image.fromarray(median.astype('uint8'))
        median = np.array(median.resize(size=(WIDTH, HEIGHT)))
        median = np.moveaxis(median, -1, 0)
    return median


def gen_median_streaming(video_file, bg_mode, max_sample_num=1800, video_range=None):
    """ Generate the median background image by sampling the video.

        Port of Video_IterableDataset.__gen_median__.

        Args:
            video_file (str): File path of the video file
            bg_mode (str): Background mode
            max_sample_num (int): Maximum number of frames to sample for generating median image
            video_range (Tuple[int], optional): Range of start second and end second of the video

        Returns:
            median (numpy.ndarray): Background image, None if bg_mode is ''
    """
    if not bg_mode:
        return None
    print('Generate median image...')
    cap = cv2.VideoCapture(video_file)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if video_range is None:
        start_frame, end_frame = 0, video_len
    else:
        start_frame = max(0, video_range[0] * fps)
        end_frame = min(video_range[1] * fps, video_len)
    video_seg_len = end_frame - start_frame

    if video_seg_len > max_sample_num:
        sample_step = video_seg_len // max_sample_num
    else:
        sample_step = 1

    frame_list = []
    for i in range(start_frame, end_frame, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            break
        frame_list.append(frame)
    cap.release()
    median = np.median(frame_list, 0)[..., ::-1] # BGR to RGB
    if bg_mode == 'concat':
        median = Image.fromarray(median.astype('uint8'))
        median = np.array(median.resize(size=(WIDTH, HEIGHT)))
        median = np.moveaxis(median, -1, 0)
    print('Median image generated.')
    return median


def process_sequence(imgs, bg_mode='', median=None):
    """ Process a frame sequence into the model input format.

        Port of the frame processing in Shuttlecock_Trajectory_Dataset.__getitem__
        and Video_IterableDataset.__process__.

        Args:
            imgs (numpy.ndarray): Frame sequence with shape (L, H, W, 3) in RGB order
            bg_mode (str): Background mode
            median (numpy.ndarray, optional): Background image (required if bg_mode)

        Returns:
            frames (numpy.ndarray): Model input with shape (C, HEIGHT, WIDTH), float64 in [0, 1]
    """
    frames = np.array([]).reshape(0, HEIGHT, WIDTH)
    for i in range(imgs.shape[0]):
        img = Image.fromarray(imgs[i])
        if bg_mode == 'subtract':
            img = Image.fromarray(np.sum(np.absolute(img - median), 2).astype('uint8'))
            img = np.array(img.resize(size=(WIDTH, HEIGHT)))
            img = img.reshape(1, HEIGHT, WIDTH)
        elif bg_mode == 'subtract_concat':
            diff_img = Image.fromarray(np.sum(np.absolute(img - median), 2).astype('uint8'))
            diff_img = np.array(diff_img.resize(size=(WIDTH, HEIGHT)))
            diff_img = diff_img.reshape(1, HEIGHT, WIDTH)
            img = np.array(img.resize(size=(WIDTH, HEIGHT)))
            img = np.moveaxis(img, -1, 0)
            img = np.concatenate((img, diff_img), axis=0)
        else:
            img = np.array(img.resize(size=(WIDTH, HEIGHT)))
            img = np.moveaxis(img, -1, 0)
        frames = np.concatenate((frames, img), axis=0)

    if bg_mode == 'concat':
        frames = np.concatenate((median, frames), axis=0)

    # Normalization
    frames /= 255.
    return frames


def build_window_ids(num_frames, seq_len, sliding_step, padding=False):
    """ Construct the frame indices of each input window.

        Port of the id construction in Shuttlecock_Trajectory_Dataset._gen_input_from_frame_arr.

        Args:
            num_frames (int): Number of frames in the video
            seq_len (int): Length of input sequence
            sliding_step (int): Sliding step of the sliding window
            padding (bool): Whether to pad the last incomplete window by repeating the last frame

        Returns:
            id (numpy.ndarray): Window indices with shape (N, seq_len, 2), int32
    """
    id = np.array([], dtype=np.int32).reshape(0, seq_len, 2)
    last_idx = -1
    for i in range(0, num_frames, sliding_step):
        tmp_idx = []
        for f in range(seq_len):
            if i + f < num_frames:
                tmp_idx.append((0, i + f))
                last_idx = i + f
            else:
                # Padding the last sequence if imcompleted
                if padding:
                    tmp_idx.append((0, last_idx))
                else:
                    break
        if len(tmp_idx) == seq_len:
            id = np.concatenate((id, [tmp_idx]), axis=0)
    return id


def iter_video_windows(video_file, seq_len, sliding_step, bg_mode='', median=None):
    """ Iteratively yield processed window sequences from the video without loading it fully.

        Port of Video_IterableDataset.__iter__.

        Args:
            video_file (str): File path of the video file
            seq_len (int): Length of input sequence
            sliding_step (int): Sliding step of the sliding window
            bg_mode (str): Background mode
            median (numpy.ndarray, optional): Background image (required if bg_mode)

        Yields:
            data_idx (numpy.ndarray): Window indices with shape (seq_len, 2), int32
            frames (numpy.ndarray): Model input with shape (C, HEIGHT, WIDTH), float64 in [0, 1]
    """
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success = True
    start_f_id, end_f_id = 0, 0
    frame_list = []
    while success:
        # Sample frames
        while len(frame_list) < seq_len:
            success, frame = cap.read()
            if not success:
                break
            frame_list.append(frame)
            end_f_id += 1

        # Form a sequence
        data_idx = [(0, i) for i in range(start_f_id, end_f_id)]
        if len(data_idx) < seq_len:
            # Padding the last sequence if imcompleted
            data_idx.extend([(0, end_f_id - 1)] * (seq_len - len(data_idx)))
            frame_list.extend([frame_list[-1]] * (seq_len - len(frame_list)))
        data_idx = np.array(data_idx)
        frames = process_sequence(np.array(frame_list)[..., ::-1], bg_mode, median)
        yield data_idx, frames

        # Update the sliding window
        frame_list = frame_list[sliding_step:]
        start_f_id = start_f_id + sliding_step

    cap.release()


def build_coor_windows(x_pred, y_pred, vis_pred, inpaint, seq_len, sliding_step, padding=False):
    """ Construct the coordinate input windows for InpaintNet.

        Port of Shuttlecock_Trajectory_Dataset._gen_input_from_pred_dict.

        Args:
            x_pred (List[int]): Predicted x coordinates (original resolution)
            y_pred (List[int]): Predicted y coordinates (original resolution)
            vis_pred (List[int]): Predicted visibility
            inpaint (List[int]): Inpaint mask
            seq_len (int): Length of input sequence
            sliding_step (int): Sliding step of the sliding window
            padding (bool): Whether to pad the last incomplete window

        Returns:
            id (numpy.ndarray): Window indices with shape (N, seq_len, 2), int32
            coor_pred (numpy.ndarray): Predicted coordinates with shape (N, seq_len, 2), float32
            pred_vis (numpy.ndarray): Predicted visibility with shape (N, seq_len), float32
            inpaint_mask (numpy.ndarray): Inpaint mask with shape (N, seq_len), float32
    """
    assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint), \
        f'Length of x_pred, y_pred, vis_pred and inpaint are not equal.'

    id = np.array([], dtype=np.int32).reshape(0, seq_len, 2)
    coor_pred = np.array([], dtype=np.float32).reshape(0, seq_len, 2)
    pred_vis = np.array([], dtype=np.float32).reshape(0, seq_len)
    inpaint_mask = np.array([], dtype=np.float32).reshape(0, seq_len)

    # Sliding on the frame sequence
    last_idx = -1
    for i in range(0, len(inpaint), sliding_step):
        tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
        for f in range(seq_len):
            if i + f < len(inpaint):
                tmp_idx.append((0, i + f))
                tmp_coor_pred.append((x_pred[i + f], y_pred[i + f]))
                tmp_vis_pred.append(vis_pred[i + f])
                tmp_inpaint.append(inpaint[i + f])
                last_idx = i + f
            else:
                # Padding the last sequence if imcompleted
                if padding:
                    tmp_idx.append((0, last_idx))
                    tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                    tmp_vis_pred.append(vis_pred[last_idx])
                    tmp_inpaint.append(inpaint[last_idx])
                else:
                    break

        if len(tmp_idx) == seq_len:
            id = np.concatenate((id, [tmp_idx]), axis=0)
            coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
            pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
            inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)

    return id, coor_pred, pred_vis, inpaint_mask
