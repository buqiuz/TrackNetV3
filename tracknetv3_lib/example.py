""" Minimal usage example of tracknetv3_lib.

    Run from the repository root:
        python -m tracknetv3_lib.example --video_file <video.mp4> \
            --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt
"""
import argparse

from tracknetv3_lib import TrackNetV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, required=True)
    parser.add_argument('--tracknet_file', type=str, default='ckpts/TrackNet_best.pt')
    parser.add_argument('--inpaintnet_file', type=str, default='ckpts/InpaintNet_best.pt')
    parser.add_argument('--save_dir', type=str, default='pred_result')
    parser.add_argument('--output_video', action='store_true', default=False)
    args = parser.parse_args()

    model = TrackNetV3(args.tracknet_file, args.inpaintnet_file)
    pred_df = model.predict_video(args.video_file, save_dir=args.save_dir,
                                  output_video=args.output_video)

    print(pred_df.head(10))
    print(f'Total frames predicted: {len(pred_df)}')


if __name__ == '__main__':
    main()
