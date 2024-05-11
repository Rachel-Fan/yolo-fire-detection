from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils.files import increment_path
from ultralytics.utils.checks import check_imgsz, check_imshow, check_yaml
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.trackers import track
from ultralytics import YOLO

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *

from collections import defaultdict
from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
from ui.resources_rc import *


import numpy as np
import threading
import traceback
import time
import json
import torch
import sys
import cv2
import os


class YoloPredictor(BasePredictor, QObject):
    # Signal definitions for communication with other parts of the application
    yolo2main_pre_img = Signal(np.ndarray)  # Signal for the original image
    yolo2main_res_img = Signal(np.ndarray)  # Signal for the result image
    yolo2main_status_msg = Signal(str)  # Signal for detection/pause/stop/completion/error messages
    yolo2main_fps = Signal(str)  # Frame rate signal
    yolo2main_labels = Signal(dict)  # Signal for detection results (number of targets per class)
    yolo2main_progress = Signal(int)  # Completion signal
    yolo2main_class_num = Signal(int)  # Signal for the number of detected classes
    yolo2main_target_num = Signal(int)  # Signal for the number of detected targets

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Call the initialization method of the parent class
        super(YoloPredictor, self).__init__()
        # Initialize PyQt's QObject
        QObject.__init__(self)

        # Parse the configuration file
        self.args = get_cfg(cfg, overrides)
        # Set the model save directory
        self.save_dir = get_save_dir(self.args)
        # Initialize a flag to mark whether the model has completed warmup
        self.done_warmup = False
        # Check whether to display images
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI-related properties
        self.used_model_name = None  # Name of the detection model to be used
        self.new_model_name = None  # Model name to be changed in real-time
        self.source = ""  # Input source
        self.stop_dtc = False  # Flag to terminate detection
        self.continue_dtc = True  # Flag to pause detection
        self.save_res = False  # Flag to save test results
        self.save_txt = False  # Flag to save labels (txt file)
        self.save_res_cam = False  # Flag to save webcam test results
        self.save_txt_cam = False  # Flag to save webcam labels (txt file)
        self.iou_thres = 0.45  # IoU threshold
        self.conf_thres = 0.25  # Confidence threshold
        self.speed_thres = 0  # Delay, in milliseconds
        self.labels_dict = {}  # Dictionary to return detection results
        self.progress_value = 0  # Value of the progress bar
        self.task = ""

        # Properties available once setup is complete
        self.model = None
        self.data = self.args.data  # data dictionary
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    # Main function for detection
    @smart_inference_mode()
    def run(self, *args, **kwargs):
        try:
            if self.args.verbose:
                LOGGER.info("Starting detection...")
            # Load model
            self.yolo2main_status_msg.emit("Loading model...")
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            with self._lock:  # for thread-safe inference
                # Set up the source each time predict is called
                self.setup_source(
                    self.source if self.source is not None else self.args.source
                )

                # Check save paths/labels
                if (self.save_res or self.save_txt or self.save_res_cam or self.save_txt_cam):
                    (self.save_dir / "labels" if (self.save_txt or self.save_txt_cam) else self.save_dir).mkdir(parents=True, exist_ok=True)

                # Model warmup
                if not self.done_warmup:
                    self.model.warmup(
                        imgsz=(
                            (1 if self.model.pt or self.model.triton else self.dataset.bs),
                            3,
                            *self.imgsz,
                        )
                    )
                    self.done_warmup = True

                self.seen, self.windows, self.batch, profilers = (
                    0,
                    [],
                    None,
                    (ops.Profile(), ops.Profile(), ops.Profile()),
                )
                # Begin detection
                count = 0  # frame count
                start_time = time.time()  # for calculating frame rate
                for batch in self.dataset:
                    if self.stop_dtc:
                        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                            self.vid_writer[-1].release()  # Release the last video writer
                        self.yolo2main_status_msg.emit("Detection stopped")
                        break

                    # Change model midway
                    if self.used_model_name != self.new_model_name:
                        self.setup_model(self.new_model_name)
                        self.used_model_name = self.new_model_name

                    # Pause toggle
                    if self.continue_dtc:
                        self.yolo2main_status_msg.emit("Detecting...")
                        self.batch = batch
                        path, im0s, vid_cap, s = batch
                        visualize = (
                            increment_path(self.save_dir / Path(path).stem, mkdir=True)
                            if self.args.visualize
                            else False
                        )

                        # Calculate completion and frame rate (to be optimized)
                        count += 1  # Increment frame count
                        if vid_cap:
                            all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frame count
                        else:
                            all_count = 1
                        self.progress_value = int(
                            count / all_count * 1000
                        )  # Progress bar (0~1000)
                        if count % 5 == 0 and count >= 5:  # Calculate frame rate every 5 frames
                            self.yolo2main_fps.emit(
                                str(int(5 / (time.time() - start_time)))
                            )
                            start_time = time.time()
                        # Preprocess
                        with profilers[0]:
                            if self.task == "Classify":
                                im = self.classify_preprocess(im0s)
                            else:
                                im = self.preprocess(im0s)

                        # Inference
                        with profilers[1]:
                            preds = self.inference(im, *args, **kwargs)

                        # Postprocess
                        with profilers[2]:
                            if self.task == "Classify":
                                self.results = self.classify_postprocess(preds, im, im0s)
                            elif self.task == "Detect":
                                self.results = self.postprocess(preds, im, im0s)
                            elif self.task == "Pose":
                                self.results = self.pose_postprocess(preds, im, im0s)
                            elif self.task == "Segment":
                                self.results = self.segment_postprocess(preds, im, im0s)

                            elif self.task == "Track":
                                model = YOLO(self.used_model_name)
                                self.results = model.track(
                                    source=self.source, tracker="bytetrack.yaml"
                                )
                                print(self.results)

                        self.run_callbacks("on_predict_postprocess_end")
                        # Visualize, save, write results
                        n = len(im0s)
                        for i in range(n):
                            self.seen += 1
                            self.results[i].speed = {
                                "preprocess": profilers[0].dt * 1e3 / n,
                                "inference": profilers[1].dt * 1e3 / n,
                                "postprocess": profilers[2].dt * 1e3 / n,
                            }
                            p, im0 = (
                                path[i],
                                (None if self.source_type.tensor else im0s[i].copy()),
                            )
                            p = Path(p)
                            label_str = self.write_results(
                                i, self.results, (p, im, im0)
                            )

                            # Labels and number dictionary
                            class_nums = 0
                            target_nums = 0
                            self.labels_dict = {}
                            if "no detections" in label_str:
                                im0 = im0
                                pass
                            else:
                                im0 = self.plotted_img
                                for ii in label_str.split(",")[:-1]:
                                    nums, label_name = ii.split("~")
                                    if ":" in nums:
                                        _, nums = nums.split(":")
                                    self.labels_dict[label_name] = int(nums)
                                    target_nums += int(nums)
                                    class_nums += 1
                            if self.save_res:
                                self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                            # Send test results
                            self.yolo2main_res_img.emit(im0)  # After detection
                            self.yolo2main_pre_img.emit(
                                im0s if isinstance(im0s, np.ndarray) else im0s[0]
                            )  # Before detection
                            self.yolo2main_class_num.emit(class_nums)
                            self.yolo2main_target_num.emit(target_nums)

                            if self.speed_thres != 0:
                                time.sleep(self.speed_thres / 1000)  # Delay, in milliseconds

                        self.yolo2main_progress.emit(self.progress_value)  # Progress bar

                    # Detection complete
                    if not self.source_type.webcam and count + 1 >= all_count:
                        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                            self.vid_writer[-1].release()  # Release the last video writer
                        self.yolo2main_status_msg.emit("Detection completed")
                        break

        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            self.yolo2main_status_msg.emit("%s" % e)


    def inference(self, img, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(
            img,
            augment=self.args.augment,
            visualize=visualize,
            embed=self.args.embed,
            *args,
            **kwargs,
        )

    def get_annotator(self, img):
        return Annotator(
            img, line_width=self.args.line_thickness, example=str(self.model.names)
        )

    def classify_preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack([self.transforms(im) for im in img], dim=0)
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(
            self.model.device
        )
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def classify_postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(
                Results(
                    orig_img=orig_img, path=img_path, names=self.model.names, probs=pred
                )
            )
        return results

    def preprocess(self, img):
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose(
                (0, 3, 1, 2)
            )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        ### important
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(
                    orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred
                )
            )
        return results

    def pose_postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=len(self.model.names),
        )

        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], orig_img.shape
            ).round()
            pred_kpts = (
                pred[:, 6:].view(len(pred), *self.model.kpt_shape)
                if len(pred)
                else pred[:, 6:]
            )
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    keypoints=pred_kpts,
                )
            )
        return results

    def segment_postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.conf_thres,
            self.iou_thres,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = (
            preds[1][-1] if len(preds[1]) == 3 else preds[1]
        )  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], orig_img.shape
                )
                masks = ops.process_mask_native(
                    proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2]
                )  # HWC
            else:
                masks = ops.process_mask(
                    proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True
                )  # HWC
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], orig_img.shape
                )
            results.append(
                Results(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    masks=masks,
                )
            )
        return results

    def pre_transform(self, img):
        same_shapes = all(x.shape == img[0].shape for x in img)
        letterbox = LetterBox(
            self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride
        )
        return [letterbox(image=x) for x in img]

    def setup_source(self, source):
        self.imgsz = check_imgsz(
            self.args.imgsz, stride=self.model.stride, min_dim=2
        )  # check image size
        self.transforms = (
            getattr(self.model.model, "transforms", classify_transforms(self.imgsz[0]))
            if self.task == "Classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            imgsz=self.imgsz,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.dataset.mode == "stream"  # streams
            or len(self.dataset) > 1000  # images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        if (
            self.source_type.webcam
            or self.source_type.from_img
            or self.source_type.tensor
        ):  # batch_size >= 1
            log_string += f"{idx}: "
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, "frame", 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / "labels" / p.stem) + (
            "" if self.dataset.mode == "image" else f"_{frame}"
        )
        # log_string += '%gx%g ' % im.shape[2:]  # print string

        result = results[idx]

        if self.task == "Classify":
            prob = results[idx].probs
            # for c in prob.top5:
            #     print(c)
        else:
            det = results[idx].boxes
            if len(det) == 0:
                return f"{log_string}(no detections), "  # if no, send this~~

            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                log_string += f"{n}~{self.model.names[int(c)]},"

        if (
            self.save_res or self.save_res_cam or self.args.save or self.args.show
        ):  # Add bbox to image
            plot_args = {
                "line_width": self.args.line_width,
                "boxes": self.args.show_boxes,
                "conf": self.args.show_conf,
                "labels": self.args.show_labels,
            }
            if not self.args.retina_masks:
                plot_args["im_gpu"] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        # if self.save_res_cam:
        #     result.save(str(self.save_dir / p.name))
        if self.save_txt or self.save_txt_cam:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(
                save_dir=self.save_dir / "crops",
                file_name=self.data_path.stem
                + ("" if self.dataset.mode == "image" else f"_{frame}"),
            )

        return log_string


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # Signal from the main window to the YOLO instance to start execution

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Basic UI setup
        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)  # Transparent rounded corners
        self.setWindowFlags(Qt.FramelessWindowHint)  # Set window flags: hide window border
        UIFuncitons.uiDefinitions(self)  # Custom interface definitions

        # Initial page setup
        self.task = ""
        self.PageIndex = 1
        self.content.setCurrentIndex(self.PageIndex)
        self.pushButton_detect.clicked.connect(self.button_detect)
        self.pushButton_pose.clicked.connect(self.button_pose)
        self.pushButton_classify.clicked.connect(self.button_classify)
        self.pushButton_segment.clicked.connect(self.button_segment)
        # self.pushButton_track.setEnabled(False)

        ####################################image or video####################################
        # Display module shadows
        UIFuncitons.shadow_style(self, self.Class_QF, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(64, 186, 193))

        # YOLO-v8 Thread
        self.yolo_predict = YoloPredictor()  # Create YOLO instance
        self.select_model = self.model_box.currentText()  # Default model

        self.yolo_thread = QThread()  # Create YOLO thread
        self.yolo_predict.yolo2main_pre_img.connect(
            lambda x: self.show_image(x, self.pre_video)
        )
        self.yolo_predict.yolo2main_res_img.connect(
            lambda x: self.show_image(x, self.res_video)
        )
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))
        self.yolo_predict.yolo2main_class_num.connect(
            lambda x: self.Class_num.setText(str(x))
        )
        self.yolo_predict.yolo2main_target_num.connect(
            lambda x: self.Target_num.setText(str(x))
        )
        self.yolo_predict.yolo2main_progress.connect(
            lambda x: self.progress_bar.setValue(x)
        )
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        self.Qtimer_ModelBox = QTimer(self)  # Timer: monitor changes in model files every 2 seconds
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Model parameters
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "iou_spinbox")
        )  # iou textbox
        self.iou_slider.valueChanged.connect(
            lambda x: self.change_val(x, "iou_slider")
        )  # iou slider
        self.conf_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "conf_spinbox")
        )  # conf textbox
        self.conf_slider.valueChanged.connect(
            lambda x: self.change_val(x, "conf_slider")
        )  # conf slider
        self.speed_spinbox.valueChanged.connect(
            lambda x: self.change_val(x, "speed_spinbox")
        )  # speed textbox
        self.speed_slider.valueChanged.connect(
            lambda x: self.change_val(x, "speed_slider")
        )  # speed slider

        # Initialize status window
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        self.fps_label.setText("--")
        self.Model_name.setText(self.select_model)

        # Select detection source
        self.src_file_button.clicked.connect(self.open_src_file)  # Select local file
        self.src_rtsp_button.clicked.connect(
            self.show_status("The function has not yet been implemented.")
        )  # Select RTSP

        # Start test buttons
        self.run_button.clicked.connect(self.run_or_continue)  # Pause/start
        self.stop_button.clicked.connect(self.stop)  # Terminate

        # Other function buttons
        self.save_res_button.toggled.connect(self.is_save_res)  # Save image option
        self.save_txt_button.toggled.connect(self.is_save_txt)  # Save label option
        ####################################image or video####################################

        ####################################camera####################################
        self.cam_data = np.array([])
        # Display camera module shadows
        UIFuncitons.shadow_style(self, self.Class_QF_cam, QColor(162, 129, 247))
        UIFuncitons.shadow_style(self, self.Target_QF_cam, QColor(251, 157, 139))
        UIFuncitons.shadow_style(self, self.Fps_QF_cam, QColor(170, 128, 213))
        UIFuncitons.shadow_style(self, self.Model_QF_cam, QColor(64, 186, 193))

        # YOLO-v8-cam Thread
        self.yolo_predict_cam = YoloPredictor()  # Create YOLO instance
        self.select_model_cam = self.model_box_cam.currentText()  # Default model

        self.yolo_thread_cam = QThread()  # Create YOLO thread
        self.yolo_predict_cam.yolo2main_pre_img.connect(
            lambda c: self.cam_show_image(c, self.pre_cam)
        )
        self.yolo_predict_cam.yolo2main_res_img.connect(
            lambda c: self.cam_show_image(c, self.res_cam)
        )
        self.yolo_predict_cam.yolo2main_status_msg.connect(
            lambda c: self.cam_show_status(c)
        )
        self.yolo_predict_cam.yolo2main_fps.connect(
            lambda c: self.fps_label_cam.setText(c)
        )
        self.yolo_predict_cam.yolo2main_class_num.connect(
            lambda c: self.Class_num_cam.setText(str(c))
        )
        self.yolo_predict_cam.yolo2main_target_num.connect(
            lambda c: self.Target_num_cam.setText(str(c))
        )
        self.yolo_predict_cam.yolo2main_progress.connect(
            self.progress_bar_cam.setValue(0)
        )
        self.main2yolo_begin_sgl.connect(self.yolo_predict_cam.run)
        self.yolo_predict_cam.moveToThread(self.yolo_thread_cam)

        self.Qtimer_ModelBox_cam = QTimer(self)  # Timer: monitor changes in model files every 2 seconds
        self.Qtimer_ModelBox_cam.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox_cam.start(2000)

        # cam model parameters
        self.model_box_cam.currentTextChanged.connect(self.cam_change_model)
        self.iou_spinbox_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "iou_spinbox_cam")
        )  # iou textbox
        self.iou_slider_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "iou_slider_cam")
        )  # iou slider
        self.conf_spinbox_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "conf_spinbox_cam")
        )  # conf textbox
        self.conf_slider_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "conf_slider_cam")
        )  # conf slider
        self.speed_spinbox_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "speed_spinbox_cam")
        )  # speed textbox
        self.speed_slider_cam.valueChanged.connect(
            lambda c: self.cam_change_val(c, "speed_slider_cam")
        )  # speed slider

        # Initialize cam status window
        self.Class_num_cam.setText("--")
        self.Target_num_cam.setText("--")
        self.fps_label_cam.setText("--")
        self.Model_name_cam.setText(self.select_model_cam)

        # Select detection source for camera
        self.src_cam_button.clicked.connect(self.cam_button)  # Choose camera

        # Start test buttons for camera
        self.run_button_cam.clicked.connect(self.cam_run_or_continue)  # Pause/start
        self.stop_button_cam.clicked.connect(self.cam_stop)  # Terminate

        # Other function buttons for camera
        self.save_res_button_cam.toggled.connect(self.cam_is_save_res)  # Save image option
        self.save_txt_button_cam.toggled.connect(self.cam_is_save_txt)  # Save label option
        ####################################camera####################################

        self.ToggleBotton.clicked.connect(
            lambda: UIFuncitons.toggleMenu(self, True)
        )  # Left navigation button

        # Initialization
        self.load_config()

        # Automatically trigger button_detect when the window opens
        self.button_detect()


    def button_classify(self):  # Event triggered after button_classify is pressed
        self.task = "Classify"
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task

        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  # Top right settings button

        # Read the model directory
        self.pt_list = os.listdir("./models/classify/")
        self.pt_list = [
            file for file in self.pt_list if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/classify/" + x)
        )  # Sort by file size
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/classify/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/classify/%s" % self.select_model_cam
        )

        # Read the cam model directory
        self.pt_list_cam = os.listdir("./models/classify/")
        self.pt_list_cam = [
            file
            for file in self.pt_list_cam
            if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list_cam.sort(
            key=lambda x: os.path.getsize("./models/classify/" + x)
        )  # Sort by file size
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)

        self.show_status("Current page: image or video detection page, Mode: Classify")

    def button_detect(self):  # Event triggered after button_detect is pressed
        self.task = "Detect"
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/detect/%s" % self.select_model_cam
        )
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  
       
        self.pt_list = os.listdir("./models/detect/")
        self.pt_list = [
            file for file in self.pt_list if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/detect/" + x)
        )  
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/detect/%s" % self.select_model_cam
        )

        self.pt_list_cam = os.listdir("./models/detect/")
        self.pt_list_cam = [
            file
            for file in self.pt_list_cam
            if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list_cam.sort(
            key=lambda x: os.path.getsize("./models/detect/" + x)
        )  
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Detect")

    def button_pose(self):  
        self.task = "Pose"
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/pose/%s" % self.select_model_cam
        )
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  

        self.pt_list = os.listdir("./models/pose/")
        self.pt_list = [
            file for file in self.pt_list if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/pose/" + x)
        )  
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/pose/%s" % self.select_model_cam
        )

        self.pt_list_cam = os.listdir("./models/pose/")
        self.pt_list_cam = [
            file
            for file in self.pt_list_cam
            if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list_cam.sort(
            key=lambda x: os.path.getsize("./models/pose/" + x)
        )  
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Pose")

    def button_segment(self):  
        self.task = "Segment"
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/segment/%s" % self.select_model_cam
        )
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(False)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  

        self.pt_list = os.listdir("./models/segment/")
        self.pt_list = [
            file for file in self.pt_list if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/segment/" + x)
        )  
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/segment/%s" % self.select_model_cam
        )

        self.pt_list_cam = os.listdir("./models/segment/")
        self.pt_list_cam = [
            file
            for file in self.pt_list_cam
            if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list_cam.sort(
            key=lambda x: os.path.getsize("./models/segment/" + x)
        )  
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Segment")

    def button_track(self):  
        self.task = "Track"
        self.yolo_predict.task = self.task
        self.yolo_predict_cam.task = self.task
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/track/%s" % self.select_model_cam
        )
        self.content.setCurrentIndex(0)
        self.src_file_button.setEnabled(True)
        self.src_cam_button.setEnabled(True)
        self.src_rtsp_button.setEnabled(True)
        self.settings_button.clicked.connect(
            lambda: UIFuncitons.settingBox(self, True)
        )  

        self.pt_list = os.listdir("./models/track/")
        self.pt_list = [
            file for file in self.pt_list if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list.sort(
            key=lambda x: os.path.getsize("./models/track/" + x)
        )  
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        self.yolo_predict_cam.new_model_name = (
            "./models/track/%s" % self.select_model_cam
        )

        self.pt_list_cam = os.listdir("./models/track/")
        self.pt_list_cam = [
            file
            for file in self.pt_list_cam
            if file.endswith((".pt", "onnx", "engine"))
        ]
        self.pt_list_cam.sort(
            key=lambda x: os.path.getsize("./models/track/" + x)
        )  # 按文件大小排序
        self.model_box_cam.clear()
        self.model_box_cam.addItems(self.pt_list_cam)
        self.show_status("Current page: image or video detection page, Mode: Track")

    ####################################image or video####################################
    # Select local files
    def open_src_file(self):
        if self.task == "Classify":
            self.show_status("Current page: image or video detection page, Mode: Classify")
        if self.task == "Detect":
            self.show_status("Current page: image or video detection page, Mode: Detect")
        if self.task == "Pose":
            self.show_status("Current page: image or video detection page, Mode: Pose")
        if self.task == "Segment":
            self.show_status("Current page: image or video detection page, Mode: Segment")
        if self.task == "Track":
            self.show_status("Current page: image or video detection page, Mode: Track")

        # Terminate camera thread to save resources
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()  # Terminate thread
            self.cam_stop()

        # Navigation between pages
        if self.PageIndex != 0:
            self.PageIndex = 0
            self.content.setCurrentIndex(self.PageIndex)
            self.settings_button.clicked.connect(
                lambda: UIFuncitons.settingBox(self, True)
            )  # Top right settings button

        if self.PageIndex == 0:
            # Set the configuration file path
            config_file = "config/fold.json"

            # Read the configuration file content
            config = json.load(open(config_file, "r", encoding="utf-8"))

            # Get the last opened folder path
            open_fold = config["open_fold"]

            # If the last opened folder does not exist, use the current working directory
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()

            # Let the user choose an image or video file through a file dialog
            if self.task == "Track":
                name, _ = QFileDialog.getOpenFileName(
                    self, "Video", open_fold, "Video Files (*.mp4 *.mkv *.avi *.flv)"
                )
            else:
                name, _ = QFileDialog.getOpenFileName(
                    self,
                    "Video/image",
                    open_fold,
                    "Video Files (*.mp4 *.mkv *.avi *.flv);;Image Files (*.jpg *.png)",
                )

            # If the user has selected a file
            if name:
                # Set the selected file's path as the source for yolo_predict
                self.yolo_predict.source = name

                # Display the file loading status
                self.show_status("File loaded: {}".format(os.path.basename(name)))

                # Update the last opened folder path in the configuration file
                config["open_fold"] = os.path.dirname(name)

                # Write the updated configuration back to the file
                config_json = json.dumps(config, ensure_ascii=False, indent=2)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(config_json)

                # Stop detection
                self.stop()

    # Main window displays the original image and detection results
    @staticmethod
    def show_image(img_src, label):
        try:
            # Get the original image's height, width, and channels
            ih, iw, _ = img_src.shape

            # Get the label's width and height
            w = label.geometry().width()
            h = label.geometry().height()

            # Maintain the original aspect ratio
            if iw / w > ih / h:
                scale = w / iw
                nw = w
                nh = int(scale * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scale = h / ih
                nw = int(scale * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # Convert the image to RGB format
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)

            # Convert image data to a Qt image object
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888,
            )

            # Display the image on the label
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # Handle exceptions, print error information
            print(repr(e))


    # Control start/pause for detection
    def run_or_continue(self):
        # Check if the source for YOLO prediction is empty
        if self.yolo_predict.source == "":
            self.show_status("Please select an image or video source before starting detection...")
            self.run_button.setChecked(False)
        else:
            # Set YOLO prediction stop flag to False
            self.yolo_predict.stop_dtc = False

            # If the start button is checked
            if self.run_button.isChecked():
                self.run_button.setChecked(True)  # Activate button
                self.save_txt_button.setEnabled(False)  # Disable saving options after starting detection
                self.save_res_button.setEnabled(False)
                self.show_status("Detection in progress...")
                self.yolo_predict.continue_dtc = True  # Control whether YOLO is paused
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            # If the start button is unchecked, it indicates a pause in detection
            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("Detection paused...")
                self.run_button.setChecked(False)  # Stop button

    # Display bottom status bar information
    def show_status(self, msg):
        # Set the text in the status bar
        self.status_bar.setText(msg)

        # Perform actions based on different status messages
        if msg == "Detection completed":
            # Enable buttons to save results and labels
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)

            # Set the detection toggle button to unchecked state
            self.run_button.setChecked(False)

            # Set the progress bar value to 0
            self.progress_bar.setValue(0)

            # If the YOLO thread is running, terminate it
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # End processing

        elif msg == "Detection terminated!":
            # Enable buttons to save results and labels
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)

            # Set the detection toggle button to unchecked state
            self.run_button.setChecked(False)

            # Set the progress bar value to 0
            self.progress_bar.setValue(0)

            # If the YOLO thread is running, terminate it
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # End processing

            # Clear the display images
            self.pre_video.clear()  # Clear the original image
            self.res_video.clear()  # Clear the detection result image
            self.Class_num.setText("--")  # Display number of classes
            self.Target_num.setText("--")  # Display number of targets
            self.fps_label.setText("--")  # Display frame rate information

    # Save test results button -- Images/Videos
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            # Display a message indicating that the run image results will not be saved
            self.show_status("NOTE: Run image results are not saved.")

            # Set the flag for saving results in the YOLO instance to False
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            # Display a message indicating that the run image results will be saved
            self.show_status("NOTE: Run image results will be saved.")

            # Set the flag for saving results in the YOLO instance to True
            self.yolo_predict.save_res = True


    # Save test results button -- Labels (txt)
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            # Display a message indicating that label results will not be saved
            self.show_status("NOTE: Label results are not saved.")

            # Set the flag for saving labels in the YOLO instance to False
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            # Display a message indicating that label results will be saved
            self.show_status("NOTE: Label results will be saved.")

            # Set the flag for saving labels in the YOLO instance to True
            self.yolo_predict.save_txt = True

    # Terminate button and related status handling
    def stop(self):
        # If the YOLO thread is running, terminate the thread
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # End thread

        # Set the termination flag of the YOLO instance to True
        self.yolo_predict.stop_dtc = True

        # Restore the state of the start button
        self.run_button.setChecked(False)

        # Enable the use of save buttons
        if self.task == "Classify":
            self.save_res_button.setEnabled(False)
            self.save_txt_button.setEnabled(False)
        else:
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)

        # Clear the image display areas
        self.pre_video.clear()  # Clear the original image
        self.res_video.clear()  # Clear the detection result image

        # Set the progress bar value to 0
        self.progress_bar.setValue(0)

        # Reset the display of class count, target count, and fps
        self.Class_num.setText("--")
        self.Target_num.setText("--")
        self.fps_label.setText("--")

    # Change detection parameters
    def change_val(self, x, flag):
        if flag == "iou_spinbox":
            # If the value of iou_spinbox changes, adjust the iou_slider value
            self.iou_slider.setValue(int(x * 100))

        elif flag == "iou_slider":
            # If the value of iou_slider changes, adjust the iou_spinbox value
            self.iou_spinbox.setValue(x / 100)
            # Display a message indicating the change in IOU threshold
            self.show_status("IOU Threshold: %s" % str(x / 100))
            # Set the IOU threshold in the YOLO instance
            self.yolo_predict.iou_thres = x / 100

        elif flag == "conf_spinbox":
            # If the value of conf_spinbox changes, adjust the conf_slider value
            self.conf_slider.setValue(int(x * 100))

        elif flag == "conf_slider":
            # If the value of conf_slider changes, adjust the conf_spinbox value
            self.conf_spinbox.setValue(x / 100)
            # Display a message indicating the change in Confidence threshold
            self.show_status("Conf Threshold: %s" % str(x / 100))
            # Set the Confidence threshold in the YOLO instance
            self.yolo_predict.conf_thres = x / 100

        elif flag == "speed_spinbox":
            # If the value of speed_spinbox changes, adjust the speed_slider value
            self.speed_slider.setValue(x)

        elif flag == "speed_slider":
            # If the value of speed_slider changes, adjust the speed_spinbox value
            self.speed_spinbox.setValue(x)
            # Display a message indicating the change in delay time
            self.show_status("Delay: %s ms" % str(x))
            # Set the delay threshold in the YOLO instance
            self.yolo_predict.speed_thres = x  # milliseconds

    # Change model
    def change_model(self, x):
        # Retrieve the current model name selected
        self.select_model = self.model_box.currentText()

        # Set the new model name in the YOLO instance
        if self.task == "Classify":
            self.yolo_predict.new_model_name = "./models/classify/%s" % self.select_model
        elif self.task == "Detect":
            self.yolo_predict.new_model_name = "./models/detect/%s" % self.select_model
        elif self.task == "Pose":
            self.yolo_predict.new_model_name = "./models/pose/%s" % self.select_model
        elif self.task == "Segment":
            self.yolo_predict.new_model_name = "./models/segment/%s" % self.select_model
        elif self.task == "Track":
            self.yolo_predict.new_model_name = "./models/track/%s" % self.select_model
        # Display a message indicating the model has been changed
        self.show_status("Change Model: %s" % self.select_model)

        # Display the new model name on the interface
        self.Model_name.setText(self.select_model)


        ####################################image or video####################################

        ####################################camera####################################
    # Camera button
    def cam_button(self):
        self.yolo_predict_cam.source = 0
        self.show_status("Current page: Webcam detection page")
        # Terminate the image or video thread to save resources
        if self.yolo_thread.isRunning():
            self.yolo_thread.quit()  # End thread
            self.stop()

        if self.PageIndex != 2:
            self.PageIndex = 2
            self.content.setCurrentIndex(self.PageIndex)
            self.settings_button.clicked.connect(
                lambda: UIFuncitons.cam_settingBox(self, True)
            )  # Top right settings button

    # Camera control start/pause detection
    def cam_run_or_continue(self):
        if self.yolo_predict_cam.source == "":
            self.show_status("No camera detected")
            self.run_button_cam.setChecked(False)

        else:
            # Set the stop flag for YOLO prediction to False
            self.yolo_predict_cam.stop_dtc = False

            # If the start button is checked
            if self.run_button_cam.isChecked():
                self.run_button_cam.setChecked(True)  # Start button
                self.save_txt_button_cam.setEnabled(False)  # Disable saving options after starting detection
                self.save_res_button_cam.setEnabled(False)
                self.cam_show_status("Detection in progress...")
                self.yolo_predict_cam.continue_dtc = True  # Control whether YOLO is paused

                if not self.yolo_thread_cam.isRunning():
                    self.yolo_thread_cam.start()
                    self.main2yolo_begin_sgl.emit()

            # If the start button is unchecked, indicating a pause in detection
            else:
                self.yolo_predict_cam.continue_dtc = False
                self.cam_show_status("Detection paused...")
                self.run_button_cam.setChecked(False)  # Stop button

    # Camera main window displays the original image and detection results
    @staticmethod
    def cam_show_image(img_src, label):
        try:
            # Get the original image's height, width, and channels
            ih, iw, _ = img_src.shape

            # Get the dimensions of the label
            w = label.geometry().width()
            h = label.geometry().height()

            # Maintain the original aspect ratio
            if iw / w > ih / h:
                scale = w / iw
                nw = w
                nh = int(scale * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scale = h / ih
                nw = int(scale * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            # Convert the image to RGB format
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)

            # Convert the image data to a Qt image object
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[2] * frame.shape[1],
                QImage.Format_RGB888,
            )

            # Display the image on the label
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            # Handle exceptions, print error information
            traceback.print_exc()
            print(f"Error: {e}")
            self.cam_show_status("%s" % e)

    # Change detection parameters for the camera
    def cam_change_val(self, c, flag):
        if flag == "iou_spinbox_cam":
            # If the value of iou_spinbox changes, adjust the iou_slider value
            self.iou_slider_cam.setValue(int(c * 100))

        elif flag == "iou_slider_cam":
            # If the value of iou_slider changes, adjust the iou_spinbox value
            self.iou_spinbox_cam.setValue(c / 100)
            # Display a message indicating the change in IOU threshold
            self.cam_show_status("IOU Threshold: %s" % str(c / 100))
            # Set the IOU threshold in the YOLO instance
            self.yolo_predict_cam.iou_thres = c / 100

        elif flag == "conf_spinbox_cam":
            # If the value of conf_spinbox changes, adjust the conf_slider value
            self.conf_slider_cam.setValue(int(c * 100))

        elif flag == "conf_slider_cam":
            # If the value of conf_slider changes, adjust the conf_spinbox value
            self.conf_spinbox_cam.setValue(c / 100)
            # Display a message indicating the change in Confidence threshold
            self.cam_show_status("Conf Threshold: %s" % str(c / 100))
            # Set the Confidence threshold in the YOLO instance
            self.yolo_predict_cam.conf_thres = c / 100

        elif flag == "speed_spinbox_cam":
            # If the value of speed_spinbox changes, adjust the speed_slider value
            self.speed_slider_cam.setValue(c)

        elif flag == "speed_slider_cam":
            # If the value of speed_slider changes, adjust the speed_spinbox value
            self.speed_spinbox_cam.setValue(c)
            # Display a message indicating the change in delay time
            self.cam_show_status("Delay: %s ms" % str(c))
            # Set the delay threshold in the YOLO instance
            self.yolo_predict_cam.speed_thres = c  # milliseconds

    # Change the model for the camera
    def cam_change_model(self, c):
        # Retrieve the current model name selected
        self.select_model_cam = self.model_box_cam.currentText()

        # Set the new model name in the YOLO instance
        if self.task == "Classify":
            self.yolo_predict_cam.new_model_name = "./models/classify/%s" % self.select_model_cam
        elif self.task == "Detect":
            self.yolo_predict_cam.new_model_name = "./models/detect/%s" % self.select_model_cam
        elif self.task == "Pose":
            self.yolo_predict_cam.new_model_name = "./models/pose/%s" % self.select_model_cam
        elif self.task == "Segment":
            self.yolo_predict_cam.new_model_name = "./models/segment/%s" % self.select_model_cam
        elif self.task == "Track":
            self.yolo_predict_cam.new_model_name = "./models/track/%s" % self.select_model_cam
        # Display a message indicating the model has been changed
        self.cam_show_status("Change Model: %s" % self.select_model_cam)

        # Display the new model name on the interface
        self.Model_name_cam.setText(self.select_model_cam)


    # Display bottom status bar information for camera
    def cam_show_status(self, msg):
        # Set the text in the status bar
        self.status_bar.setText(msg)

        # Perform actions based on different status messages
        if msg == "Detection completed" or msg == "检测完成":
            # Enable buttons to save results and labels
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)

            # Set the detection toggle button to unchecked state
            self.run_button_cam.setChecked(False)

            # Set the progress bar value to 0
            self.progress_bar_cam.setValue(0)

            # If the YOLO thread is running, terminate it
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # End processing

        elif msg == "Detection terminated!" or msg == "检测终止":
            # Enable buttons to save results and labels
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)

            # Set the detection toggle button to unchecked state
            self.run_button_cam.setChecked(False)

            # Set the progress bar value to 0
            self.progress_bar_cam.setValue(0)

            # If the YOLO thread is running, terminate it
            if self.yolo_thread_cam.isRunning():
                self.yolo_thread_cam.quit()  # End processing

            # Clear the display images
            self.pre_cam.clear()  # Clear the original image
            self.res_cam.clear()  # Clear the detection result image
            self.Class_num_cam.setText("--")  # Display number of classes
            self.Target_num_cam.setText("--")  # Display number of targets
            self.fps_label_cam.setText("--")  # Display frame rate information

    # Save test results button for camera -- Images/Videos
    def cam_is_save_res(self):
        if self.save_res_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Display a message indicating that run image results will not be saved
            self.show_status("NOTE: Run image results are not saved.")

            # Set the flag for saving results in the YOLO instance to False
            self.yolo_thread_cam.save_res = False
        elif self.save_res_button_cam.checkState() == Qt.CheckState.Checked:
            # Display a message indicating that run image results will be saved
            self.show_status("NOTE: Run image results will be saved.")

            # Set the flag for saving results in the YOLO instance to True
            self.yolo_thread_cam.save_res = True

    # Save test results button for camera -- Labels (txt)
    def cam_is_save_txt(self):
        if self.save_txt_button_cam.checkState() == Qt.CheckState.Unchecked:
            # Display a message indicating that label results will not be saved
            self.show_status("NOTE: Label results are not saved.")

            # Set the flag for saving labels in the YOLO instance to False
            self.yolo_thread_cam.save_txt_cam = False
        elif self.save_txt_button_cam.checkState() == Qt.CheckState.Checked:
            # Display a message indicating that label results will be saved
            self.show_status("NOTE: Label results will be saved.")

            # Set the flag for saving labels in the YOLO instance to True
            self.yolo_thread_cam.save_txt_cam = True

    # Camera terminate button and related status handling
    def cam_stop(self):
        # If the YOLO thread is running, terminate the thread
        if self.yolo_thread_cam.isRunning():
            self.yolo_thread_cam.quit()  # End thread

        # Set the termination flag of the YOLO instance to True
        self.yolo_predict_cam.stop_dtc = True

        # Restore the state of the start button
        self.run_button_cam.setChecked(False)

        # Enable the use of save buttons
        if self.task == "Classify":
            self.save_res_button_cam.setEnabled(False)
            self.save_txt_button_cam.setEnabled(False)
        else:
            self.save_res_button_cam.setEnabled(True)
            self.save_txt_button_cam.setEnabled(True)

        # Clear the image display areas
        self.pre_cam.clear()

        # Clear the detection result image display area
        self.res_cam.clear()

        # Set the progress bar value to 0
        # self.progress_bar.setValue(0)

        # Reset the display of class count, target count, and fps
        self.Class_num_cam.setText("--")
        self.Target_num_cam.setText("--")
        self.fps_label_cam.setText("--")


    ####################################camera####################################

    ####################################Shared Functionality####################################
    # Monitor changes to model files continuously
    def ModelBoxRefre(self):
        # Get all model files in the model directory
        if self.task == "Classify":
            pt_list = os.listdir("./models/classify")
            pt_list = [
                file for file in pt_list if file.endswith((".pt", "onnx", "engine"))
            ]
            pt_list.sort(key=lambda x: os.path.getsize("./models/classify/" + x))

            # If the model file list has changed, update the model dropdown content
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == "Detect":
            pt_list = os.listdir("./models/detect")
            pt_list = [
                file for file in pt_list if file.endswith((".pt", "onnx", "engine"))
            ]
            pt_list.sort(key=lambda x: os.path.getsize("./models/detect/" + x))
            # If the model file list has changed, update the model dropdown content
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)

        elif self.task == "Pose":
            pt_list = os.listdir("./models/pose")
            pt_list = [
                file for file in pt_list if file.endswith((".pt", "onnx", "engine"))
            ]
            pt_list.sort(key=lambda x: os.path.getsize("./models/pose/" + x))

            # If the model file list has changed, update the model dropdown content
            if pt_list != self.pt_list:
                self.pt_list = pt_list
                self.model_box.clear()
                self.model_box.addItems(self.pt_list)
                self.pt_list_cam = pt_list
                self.model_box_cam.clear()
                self.model_box_cam.addItems(self.pt_list_cam)


            elif self.task == "Segment":
                pt_list = os.listdir("./models/segment")
                pt_list = [
                    file for file in pt_list if file.endswith((".pt", "onnx", "engine"))
                ]
                pt_list.sort(key=lambda x: os.path.getsize("./models/segment/" + x))

                # If the model file list has changed, update the model dropdown content
                if pt_list != self.pt_list:
                    self.pt_list = pt_list
                    self.model_box.clear()
                    self.model_box.addItems(self.pt_list)
                    self.pt_list_cam = pt_list
                    self.model_box_cam.clear()
                    self.model_box_cam.addItems(self.pt_list_cam)

            elif self.task == "Track":
                pt_list = os.listdir("./models/track")
                pt_list = [
                    file for file in pt_list if file.endswith((".pt", "onnx", "engine"))
                ]
                pt_list.sort(key=lambda x: os.path.getsize("./models/track/" + x))

                if pt_list != self.pt_list:
                    self.pt_list = pt_list
                    self.model_box.clear()
                    self.model_box.addItems(self.pt_list)
                    self.pt_list_cam = pt_list
                    self.model_box_cam.clear()
                    self.model_box_cam.addItems(self.pt_list_cam)

        # Get mouse position (used for dragging the window by holding the title bar)
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # Optimize adjustments when resizing the window (specifically for dragging the bottom-right corner to resize)
    def resizeEvent(self, event):
        # Update the resize handles
        UIFuncitons.resize_grips(self)

    # Initialize configuration
    def load_config(self):
        config_file = "config/setting.json"

        # If the configuration file does not exist, create and write default settings
        if not os.path.exists(config_file):
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
                save_res_cam = 0
                save_txt_cam = 0
                new_config = {
                    "iou": iou,
                    "conf": conf,
                    "rate": rate,
                    "save_res": save_res,
                    "save_txt": save_txt,
                    "save_res": save_res_cam,
                    "save_txt": save_txt_cam,
                }
                new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write(new_json)
        else:
            # If the configuration file exists, read the configuration
            config = json.load(open(config_file, "r", encoding="utf-8"))

            # Check if the configuration content is complete; if incomplete, use default values
            if len(config) != 7:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
                save_res_cam = 0
                save_txt_cam = 0
            else:
                iou = config["iou"]
                conf = config["conf"]
                rate = config["rate"]
                save_res = config["save_res"]
                save_txt = config["save_txt"]
                save_res_cam = config["save_res_cam"]
                save_txt_cam = config["save_txt_cam"]

        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = False if save_res == 0 else True
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = False if save_txt == 0 else True
        self.run_button.setChecked(False)

        self.save_res_button_cam.setCheckState(Qt.CheckState(save_res_cam))
        self.yolo_predict_cam.save_res_cam = False if save_res_cam == 0 else True
        self.save_txt_button_cam.setCheckState(Qt.CheckState(save_txt_cam))
        self.yolo_predict_cam.save_txt_cam = False if save_txt_cam == 0 else True
        self.run_button_cam.setChecked(False)
        self.show_status("Welcome to YOLOv8 Detect System. Please choose Mode")

    # Close event, terminate threads, and save settings
    def closeEvent(self, event):
        # Save settings to configuration file
        config_file = "config/setting.json"
        config = dict()
        config["iou"] = self.iou_spinbox.value()
        config["conf"] = self.conf_spinbox.value()
        config["rate"] = self.speed_spinbox.value()
        config["save_res"] = (
            0 if self.save_res_button.checkState() == Qt.Unchecked else 2
        )
        config["save_txt"] = (
            0 if self.save_txt_button.checkState() == Qt.Unchecked else 2
        )
        config["save_res_cam"] = (
            0 if self.save_res_button_cam.checkState() == Qt.Unchecked else 2
        )
        config["save_txt_cam"] = (
            0 if self.save_txt_button_cam.checkState() == Qt.Unchecked else 2
        )
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_json)

        # Exit threads and application
        if self.yolo_thread.isRunning() or self.yolo_thread_cam.isRunning():
            # If YOLO threads are running, terminate them
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()

            self.yolo_predict_cam.stop_dtc = True
            self.yolo_thread_cam.quit()
            # Display exit prompt, wait for 3 seconds
            MessageBox(
                self.close_button,
                title="Note",
                text="Exiting, please wait...",
                time=3000,
                auto=True,
            ).exec()

            # Exit the application
            sys.exit(0)
        else:
            # If YOLO threads are not running, exit the application directly
            sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Home = MainWindow()
    
    # Uncomment the following lines if you need to handle camera input in separate threads
    # Create camera thread
    camera_thread = CameraThread()
    # Connect the camera's signal to receive captured images to a slot in Home
    camera_thread.imageCaptured.connect(Home.cam_data)
    # Start the camera thread
    camera_thread.start()
    
    # Display the main window
    Home.show()
    # Start the main application loop and exit when it ends
    sys.exit(app.exec())

