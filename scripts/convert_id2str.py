import glob 
import cv2 
import time 
import tqdm
import os
import argparse
import sys
import json

import numpy as np
from wedet.debug import FpsMe, DictMetric
from wedet.structures import SpireAnno


def get_parser():
    parser = argparse.ArgumentParser(description="Yolo TensorRT Demo")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="wedet-dataset name, e.g. coco",
    )
    parser.add_argument(
        "--anno-dir",
        default="build/annotations",
        help="Path to spire annotations",
    )
    parser.add_argument(
        "--gt",
        default="instances_val2017.json",
        help="Ground-Truth json file, e.g. instances_val2017.json",
    )
    return parser


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


if __name__ == "__main__":
    args = get_parser().parse_args()

    spire_anno = SpireAnno(dataset=args.dataset_name)
    metric = DictMetric()

    # print(spire_anno.id_class)
    file_list = os.listdir(args.anno_dir)
    for f in file_list:
        if f.endswith('.json'):
            fn = os.path.join(args.anno_dir, f)
            # print(fn)
            with open(fn, 'r') as load_f:
                load_dict = json.load(load_f)

                metric.update({"infer_time": load_dict['infer_time']})

                for i in range(len(load_dict['annos'])):
                    cat = load_dict['annos'][i]['category_name']
                    if is_number(cat):
                        load_dict['annos'][i]['category_name'] = spire_anno.id_class[int(cat)]
                # print(load_dict)
            with open(fn, "w") as dump_f:
                json.dump(load_dict, dump_f)

    spire_anno.cocoapi_eval(args.gt, args.anno_dir)
    print("MEAN Infer Time: {:.1f} ms (+- {:.1f} ms)".format(
        metric.mean()["infer_time"],
        metric.std()["infer_time"]
    ))

