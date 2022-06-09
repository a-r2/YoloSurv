import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from settings import *

FILE = Path(__file__).resolve()
ROOT = "yolov5"  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

def loop(send_mail):
    @torch.no_grad()
    def run(
            weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        try:
            # Arguments
            weights = NN_TYPE
            conf_thres = CONF_THR
            classes = 0
            max_det = 1

            # Parameters
            SOURCE = str(0)
            PRED_FLAG_LEN = 10 #predictions buffer length
            RECORD_THR = 0.5 #buffered predictions percentage threshold above which recording is activated

            # Directories
            SAVE_DIR = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            (SAVE_DIR / 'labels' if save_txt else SAVE_DIR).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Dataloader
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(SOURCE, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
            pred_flag_b = [False] * PRED_FLAG_LEN
            prev_record_flag = False
            path = "0"
            for _, im, im0s, vid_cap, s in dataset:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                pred = model(im, augment=augment, visualize=visualize)

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                pred_flag_b[1:] = pred_flag_b[:-1]
                pred_flag_b[0] = True if len(pred[0].numpy()) > 0 else False
                record_flag = True if pred_flag_b.count(True) / PRED_FLAG_LEN > RECORD_THR else False

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    im0, frame = im0s[i].copy(), dataset.count
                    if record_flag and not prev_record_flag:
                        path = str(int(path) + 1)
                    save_path = str(SAVE_DIR / path)  # im.jpg

                    # Stream results
                    if view_img:
                        cv2.imshow("0", im0)
                        cv2.waitKey(1)  #1 milisecond

                    # Save results (image with detections)
                    if record_flag:
                        if vid_path[i] != save_path:  #new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  #release video writer
                            if vid_cap: #video capture
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else: #video stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                    elif prev_record_flag and not record_flag:
                        vid_writer[i].release() #release video writer
                        send_mail.set() #allow sending mail
                    prev_record_flag = record_flag
        except:
            vid_writer[i].release() #release video writer
            send_mail.set() #allow sending mail
            # Print message
            print("DETECTOR PROCESS ENDED SUCCESSFULLY!")
    run()
