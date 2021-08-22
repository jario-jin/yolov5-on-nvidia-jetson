import glob 
import cv2 
import time 
import tqdm
import os
import argparse
import sys

import numpy as np 
from yolov5s import YoLov5TRT
from wedet.debug import FpsMe, DictMetric
from wedet.structures import SpireAnno

import matplotlib
matplotlib.use('Agg')


def get_parser():
    parser = argparse.ArgumentParser(description="Yolo TensorRT Demo")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="wedet-dataset name, e.g. coco",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam."
    )
    parser.add_argument(
        "--video-input",
        # default="/home/jario/Videos/uav-track/20210413155656.mp4",
        help="Path to video file."
    )
    parser.add_argument(
        "--input",
        default="/home/jario/dataset/val2017/*.jpg",
        help="A single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--gt",
        default="/home/jario/dataset/instances_val2017.json",
        help="Ground-Truth json file, e.g. instances_val2017.json",
    )
    parser.add_argument(
        "--output",
        default="/tmp/vis",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--engine",
        default="yolov5s.engine",
        help="Yolov5 tensorrt engine.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    yolo = YoLov5TRT(checkpoint_path='build/' + args.engine, device_num=0)
    fps_rec = FpsMe()
    spire_anno = SpireAnno(dataset=args.dataset_name, spire_dir=args.output)
    metric = DictMetric()

    if args.input:
        args.input = glob.glob(os.path.expanduser(args.input))
        assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=True):
            img = cv2.imread(path)

            t1 = time.time()
            results = yolo.infer(raw_image=img)
            metric.update({"infer_time": time.time() - t1})
            img = spire_anno.visualize_instances(
                img, results, output_spire_json=True, image_name=os.path.basename(path))

            updated, fps = fps_rec.spin()
            if updated:
                print("FPS: {}".format(fps))

            cv2.imshow('infer', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        spire_anno.cocoapi_eval(args.gt, os.path.join(args.output, "annotations"))
        print("MEAN Infer Time: {:.1f} ms (+- {:.1f} ms)".format(
            metric.mean()["infer_time"]*1000.,
            metric.std()["infer_time"]*1000.
        ))
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Camera {} can not opened!".format(camera_id))
            yolo.destroy()
            sys.exit(0)

        while True:
            ret, img = cap.read()
            if not ret:
                print('Can not read new frame!')
                break

            results = yolo.infer(raw_image=img)
            img = spire_anno.visualize_instances(img, results)

            updated, fps = fps_rec.spin()
            if updated:
                print("FPS: {}".format(fps))

            cv2.imshow('infer', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        while True:
            ret, img = video.read()
            if not ret:
                print('Can not read new frame!')
                break

            results = yolo.infer(raw_image=img)
            img = spire_anno.visualize_instances(img, results)

            updated, fps = fps_rec.spin()
            if updated:
                print("FPS: {}".format(fps))

            cv2.imshow('infer', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    yolo.destroy()
