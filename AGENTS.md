# AGENTS.md

Research code for TrackNetV3 (badminton shuttlecock tracking). Two modules trained separately: **TrackNet** (heatmap tracking) and **InpaintNet** (trajectory rectification). No test suite, lint, or CI — verification means running the scripts below.

## Standalone inference library

`tracknetv3_lib/` is a self-contained, dependency-light port of `predict.py` for secondary development (see its README). When changing inference-affecting logic in `dataset.py`, `utils/general.py`, or `test.py`, re-verify the lib with `tracknetv3_lib/verify.py` against a fresh `predict.py` baseline — its output must stay frame-identical.

## Setup

- `pip install -r requirements.txt` — pinned old deps (torch 1.10.0, Python 3.8 era); do not assume modern APIs.
- `data/` and `ckpts/` are gitignored. Dataset (Shuttlecock Trajectory Dataset) and pretrained checkpoints must be downloaded (links in README).
- `corrected_test_label/` IS checked in: `preprocess.py` copies it into `data/test/*/corrected_csv/` and `data/drop_frame.json`. Run `python preprocess.py` after arranging the dataset (also generates `frame/` dirs and the `val` split).

## Hardcoded paths (not CLI args)

- Data root is `data_dir = 'data'` in `dataset.py:12` — edit the file to change it; nearly every script imports it.
- Test-split rallies are read from `corrected_csv/` instead of `csv/` (see `dataset.py:224`, `utils/general.py:378`). If test evaluation can't find labels, this is why.

## Dataset cache gotcha

`dataset.py` silently reuses cached `.npz` files in the data root:
- `data_l{seq_len}_s{sliding_step}_{data_mode}_{split}.npz`
- `img_config_{HEIGHT}x{WIDTH}_{split}.npz`

If you change dataset logic (anything in `dataset.py` / `utils/general.py`), delete these caches or stale data is loaded. README says ".npy"; they are actually `.npz`.

## Pipeline order (matters)

1. `preprocess.py` — dataset prep (above).
2. `train.py --model_name TrackNet ...` — tracking module.
3. `python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16` — runs TrackNet over the data to produce InpaintNet training inputs (predicted trajectories + inpaint masks).
4. `train.py --model_name InpaintNet ...`
- Evaluation: `generate_mask_data.py --split_list test` then `test.py --inpaintnet_file ...` (full model). TrackNet-only eval skips step 3: `test.py --tracknet_file ...` directly.

## Quirks

- README's InpaintNet train command uses `--epoch 300`; the actual arg is `--epochs` (`train.py:184`).
- `--resume_training` resumes from `{model_name}_cur.pt` already in `--save_dir` (asserts it exists); hyperparams for the run come from that checkpoint.
- TensorBoard logs go to `{save_dir}/logs`.
- `predict.py --large_video` switches to IterableDataset to avoid OOM on long videos (slower); pair with `--video_range` / `--max_sample_num` for background estimation.
- `error_analysis.py` and `correct_label.py` are Dash apps with hardcoded file lists at the top of the file — edit them before running.
- Generated prediction coordinates are in model input space (512x288), not original video resolution; `Img_scaler` in the pred dict converts back.
