import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .model import TrackNet, InpaintNet
from ._preprocess import (HEIGHT, WIDTH, generate_frames, get_video_info,
                          gen_median_from_all, gen_median_streaming,
                          process_sequence, build_window_ids, iter_video_windows,
                          build_coor_windows)
from ._postprocess import (get_ensemble_weight, generate_inpaint_mask,
                           extract_predictions, write_pred_csv, write_pred_video)

DELTA_T = 1 / math.sqrt(HEIGHT ** 2 + WIDTH ** 2)
COOR_TH = DELTA_T * 50


def select_device():
    """ Select the best available PyTorch backend (CUDA, MPS or CPU). """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_checkpoint(path, device):
    """ Load a checkpoint onto the device, robust to torch>=2.6 weights_only default change. """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _build_tracknet(seq_len, bg_mode):
    """ Construct the TrackNet model matching the checkpoint configuration. """
    if bg_mode == 'subtract':
        return TrackNet(in_dim=seq_len, out_dim=seq_len)
    elif bg_mode == 'subtract_concat':
        return TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
    elif bg_mode == 'concat':
        return TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
    else:
        return TrackNet(in_dim=seq_len * 3, out_dim=seq_len)


class TrackNetV3:
    """ TrackNetV3 inference wrapper (TrackNet tracking + InpaintNet trajectory rectification).

        Args:
            tracknet_file (str): File path of the TrackNet checkpoint
            inpaintnet_file (str, optional): File path of the InpaintNet checkpoint
            device (torch.device, optional): Inference device, auto-selected if None

        Example:
            >>> model = TrackNetV3('ckpts/TrackNet_best.pt', 'ckpts/InpaintNet_best.pt')
            >>> df = model.predict_video('video.mp4')
    """
    def __init__(self, tracknet_file, inpaintnet_file=None, device=None):
        self.device = device if device is not None else select_device()
        print(f'Using PyTorch device: {self.device}')

        tracknet_ckpt = _load_checkpoint(tracknet_file, self.device)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        self.tracknet = _build_tracknet(self.tracknet_seq_len, self.bg_mode).to(self.device)
        self.tracknet.load_state_dict(tracknet_ckpt['model'])

        if inpaintnet_file:
            inpaintnet_ckpt = _load_checkpoint(inpaintnet_file, self.device)
            self.inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
            self.inpaintnet = InpaintNet().to(self.device)
            self.inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        else:
            self.inpaintnet_seq_len = None
            self.inpaintnet = None

    def _iter_frame_batches_arr(self, frame_arr, median, ids, batch_size):
        """ Yield (indices, frames) batches from an in-memory frame array (RGB). """
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            batch = np.stack([process_sequence(frame_arr[idx[:, 1], ...], self.bg_mode, median)
                              for idx in batch_ids])
            i = torch.from_numpy(np.stack(batch_ids))
            x = torch.from_numpy(batch).float()
            yield i, x

    def _iter_frame_batches_stream(self, video_file, seq_len, sliding_step, median, batch_size):
        """ Yield (indices, frames) batches by streaming windows from the video. """
        batch_ids, batch_frames = [], []
        for data_idx, frames in iter_video_windows(video_file, seq_len, sliding_step,
                                                   bg_mode=self.bg_mode, median=median):
            batch_ids.append(data_idx)
            batch_frames.append(frames)
            if len(batch_ids) == batch_size:
                yield torch.from_numpy(np.stack(batch_ids)), torch.from_numpy(np.stack(batch_frames)).float()
                batch_ids, batch_frames = [], []
        if batch_ids:
            yield torch.from_numpy(np.stack(batch_ids)), torch.from_numpy(np.stack(batch_frames)).float()

    def _iter_coor_batches(self, tracknet_pred_dict, seq_len, sliding_step, padding, img_shape, batch_size):
        """ Yield (indices, coor_pred, inpaint_mask) batches for InpaintNet.

            Returns:
                num_windows (int): Total number of windows
                generator: Iterator of (i, coor_pred, inpaint_mask) batches
        """
        ids, coor_pred, pred_vis, inpaint_mask = build_coor_windows(
            tracknet_pred_dict['X'], tracknet_pred_dict['Y'],
            tracknet_pred_dict['Visibility'], tracknet_pred_dict['Inpaint_Mask'],
            seq_len, sliding_step, padding=padding)
        w, h = img_shape
        # Normalization
        coor_pred[:, :, 0] = coor_pred[:, :, 0] / w
        coor_pred[:, :, 1] = coor_pred[:, :, 1] / h

        def _gen():
            for start in range(0, len(ids), batch_size):
                i = torch.from_numpy(ids[start:start + batch_size])
                c = torch.from_numpy(coor_pred[start:start + batch_size]).float()
                m = torch.from_numpy(inpaint_mask[start:start + batch_size]).reshape(-1, seq_len, 1).float()
                yield i, c, m

        return len(ids), _gen()

    @torch.no_grad()
    def predict_video(self, video_file, eval_mode='weight', large_video=False, batch_size=16,
                      max_sample_num=1800, video_range=None, output_video=False, traj_len=8,
                      save_dir=None, verbose=True):
        """ Predict the shuttlecock trajectory of a video.

            Args:
                video_file (str): File path of the video
                eval_mode (str): Temporal ensemble mode, 'weight' (default) or 'nonoverlap'
                large_video (bool): Stream the video to avoid memory errors on long videos (slower)
                batch_size (int): Batch size for inference
                max_sample_num (int): Maximum number of frames to sample for background estimation
                video_range (Tuple[int], optional): Start and end seconds of the video for background estimation
                output_video (bool): Whether to save a video with the predicted trajectory (requires save_dir)
                traj_len (int): Length of trajectory to draw on the output video
                save_dir (str, optional): Directory to save the prediction csv (and video)
                verbose (bool): Whether to show the progress bar

            Returns:
                pred_df (pandas.DataFrame): Prediction result with columns
                    Frame, Visibility, X, Y (original video resolution)
        """
        disable = not verbose
        info = get_video_info(video_file)
        w, h = info['w'], info['h']
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                              'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

        # Test on TrackNet
        self.tracknet.eval()
        seq_len = self.tracknet_seq_len
        if eval_mode == 'nonoverlap':
            # Non-overlap sampling
            if large_video:
                median = gen_median_streaming(video_file, self.bg_mode, max_sample_num, video_range)
                data_iter = self._iter_frame_batches_stream(video_file, seq_len, seq_len, median, batch_size)
                print(f'Video length: {info["video_len"]}')
            else:
                frame_list = generate_frames(video_file)
                frame_arr = np.array(frame_list)[:, :, :, ::-1]
                median = gen_median_from_all(frame_arr, self.bg_mode)
                ids = build_window_ids(len(frame_arr), seq_len, seq_len, padding=True)
                data_iter = self._iter_frame_batches_arr(frame_arr, median, ids, batch_size)

            for step, (i, x) in enumerate(tqdm(data_iter, disable=disable)):
                x = x.float().to(self.device)
                y_pred = self.tracknet(x).detach().cpu()

                # Predict
                tmp_pred = extract_predictions(i, y_pred=y_pred, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])
        else:
            # Overlap sampling for temporal ensemble
            if large_video:
                median = gen_median_streaming(video_file, self.bg_mode, max_sample_num, video_range)
                data_iter = self._iter_frame_batches_stream(video_file, seq_len, 1, median, batch_size)
                video_len = info['video_len']
                print(f'Video length: {video_len}')
            else:
                frame_list = generate_frames(video_file)
                frame_arr = np.array(frame_list)[:, :, :, ::-1]
                median = gen_median_from_all(frame_arr, self.bg_mode)
                ids = build_window_ids(len(frame_arr), seq_len, 1, padding=False)
                data_iter = self._iter_frame_batches_arr(frame_arr, median, ids, batch_size)
                video_len = len(frame_list)

            # Init prediction buffer params
            num_sample, sample_count = video_len - seq_len + 1, 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len - 1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
            weight = get_ensemble_weight(seq_len, eval_mode)
            for step, (i, x) in enumerate(tqdm(data_iter, disable=disable)):
                x = x.float().to(self.device)
                b_size, seq_len = i.shape[0], i.shape[1]
                y_pred = self.tracknet(x).detach().cpu()

                y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        y_pred = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                    else:
                        # General case
                        y_pred = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                    ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last batch
                        y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                        y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                        for f in range(1, seq_len):
                            # Last input sequence
                            y_pred = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

                # Predict
                tmp_pred = extract_predictions(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])

                # Update buffer, keep last predictions for ensemble in next iteration
                y_pred_buffer = y_pred_buffer[-buffer_size:]

        # Test on TrackNetV3 (TrackNet + InpaintNet)
        if self.inpaintnet is not None:
            self.inpaintnet.eval()
            seq_len = self.inpaintnet_seq_len
            tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
            inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}

            if eval_mode == 'nonoverlap':
                # Non-overlap sampling
                _, data_iter = self._iter_coor_batches(tracknet_pred_dict, seq_len, seq_len, True,
                                                       (w, h), batch_size)
                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_iter, disable=disable)):
                    with torch.no_grad():
                        coor_inpaint = self.inpaintnet(coor_pred.to(self.device), inpaint_mask.to(self.device)).detach().cpu()
                        coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask) # replace predicted coordinates with inpainted coordinates

                    # Thresholding
                    th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                    coor_inpaint[th_mask] = 0.

                    # Predict
                    tmp_pred = extract_predictions(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                    for key in tmp_pred.keys():
                        inpaint_pred_dict[key].extend(tmp_pred[key])
            else:
                # Overlap sampling for temporal ensemble
                num_windows, data_iter = self._iter_coor_batches(tracknet_pred_dict, seq_len, 1, False,
                                                                 (w, h), batch_size)
                weight = get_ensemble_weight(seq_len, eval_mode)

                # Init buffer params
                num_sample, sample_count = num_windows, 0
                buffer_size = seq_len - 1
                batch_i = torch.arange(seq_len) # [0, 1, 2, ..., 15]
                frame_i = torch.arange(seq_len - 1, -1, -1) # [15, 14, ..., 0]
                coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

                for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_iter, disable=disable)):
                    b_size = i.shape[0]
                    with torch.no_grad():
                        coor_inpaint = self.inpaintnet(coor_pred.to(self.device), inpaint_mask.to(self.device)).detach().cpu()
                        coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                    # Thresholding
                    th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                    coor_inpaint[th_mask] = 0.

                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                    ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                    ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)

                    for b in range(b_size):
                        if sample_count < buffer_size:
                            # Imcomplete buffer
                            coor_inpaint = coor_inpaint_buffer[batch_i + b, frame_i].sum(0)
                            coor_inpaint /= (sample_count + 1)
                        else:
                            # General case
                            coor_inpaint = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                        ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                        sample_count += 1

                        if sample_count == num_sample:
                            # Last input sequence
                            coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                            coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                            for f in range(1, seq_len):
                                coor_inpaint = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0)
                                coor_inpaint /= (seq_len - f)
                                ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                                ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                    # Thresholding
                    th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                    ensemble_coor_inpaint[th_mask] = 0.

                    # Predict
                    tmp_pred = extract_predictions(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                    for key in tmp_pred.keys():
                        inpaint_pred_dict[key].extend(tmp_pred[key])

                    # Update buffer, keep last predictions for ensemble in next iteration
                    coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        pred_dict = inpaint_pred_dict if self.inpaintnet is not None else tracknet_pred_dict

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            video_name = os.path.basename(video_file)[:-4]
            write_pred_csv(pred_dict, save_file=os.path.join(save_dir, f'{video_name}_ball.csv'))
            if output_video:
                write_pred_video(video_file, pred_dict,
                                 save_file=os.path.join(save_dir, f'{video_name}.mp4'),
                                 traj_len=traj_len)

        return pd.DataFrame({'Frame': pred_dict['Frame'],
                             'Visibility': pred_dict['Visibility'],
                             'X': pred_dict['X'],
                             'Y': pred_dict['Y']})
