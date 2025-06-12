import gradio as gr
import cv2
from fastrtc import WebRTC
import time
import threading
import numpy as np
import mmcv
from time import sleep

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
import hashlib
from twilio.rest import Client
import os

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

DET_CFG = "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_WEIGHTS = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

POSE_CFG = "configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py"
POSE_WEIGHTS = "models/ProbPose-s.pth"

DEVICE = 'cuda:0'


client = Client(
  os.getenv("TWILIO_ACCOUNT_SID"),
  os.getenv("TWILIO_AUTH_TOKEN")
)
token = client.tokens.create()  # includes token.iceServers
rtc_configuration = {"iceServers": token.ice_servers}


# WebRTC configuration for webcam streaming
# rtc_configuration = None
webcam_constraints = {
    "video": {
        "width": {"exact": 320},
        "height": {"exact": 240},
        "sampleRate": {"ideal": 2, "max": 5}
    }
}


class AsyncFrameProcessor:
    """
    Asynchronous frame processor that handles real-time video stream processing.
    
    Maintains single-slot input and output queues to process only the latest frame,
    preventing queue buildup and ensuring real-time performance.
    """
    
    def __init__(self, processing_delay=0.5, startup_delay=0.0):
        """
        Initialize the async frame processor.
        
        Args:
            processing_delay (float): Simulated processing time in seconds
            startup_delay (float): Delay before processing starts
        """
        self.processing_delay = processing_delay
        self.startup_delay = startup_delay
        self.first_call_time = None
        self.frame_counter = 0
        self.runtime_start = time.time() + self.startup_delay

        # Thread-safe single-slot queues
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.latest_input_frame = None
        self.latest_output_frame = None
        
        # Threading components
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.new_frame_signal = threading.Event()
        
        # Detector and pose estimator models
        self.pose_model = None
        self.det_model = None
        self.visualizer = None
        self.init_models()

        # Start background processing
        self._start_processing_thread()
    
    def _start_processing_thread(self):
        """Start the background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_event.clear()
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()
    
    def _processing_worker(self):
        """Background thread that processes the latest frame"""
        while not self.stop_event.is_set():
            # Wait for a new frame to be available
            if self.new_frame_signal.wait(timeout=1.0):
                self.new_frame_signal.clear()
                
                # Get the latest input frame
                with self.input_lock:
                    if self.latest_input_frame is not None:
                        frame_to_process = self.latest_input_frame.copy()
                        frame_number = self.frame_counter
                        process_unique_hash = hashlib.md5(frame_to_process.tobytes()).hexdigest()
                        # print(f"Processing unique hash: {process_unique_hash}")
        
                    else:
                        continue
                
                # Process the frame
                processed_frame = self._process_frame(frame_to_process)

                # Write frame number in the top left corner
                current_runtime = time.time() - self.runtime_start
                hh = int(current_runtime // 3600)
                mm = int((current_runtime % 3600) // 60)
                ss = int(current_runtime % 60)
                if hh > 0:
                    print_str = "{:02d}:{:02d}:{:02d}".format(hh, mm, ss)
                else:
                    print_str = "{:02d}:{:02d}".format(mm, ss)

                processed_frame = cv2.putText(
                    processed_frame,
                    print_str,
                    [10, 50],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=2,
                )
                
                # Store the processed result
                with self.output_lock:
                    self.latest_output_frame = processed_frame
    
    def _process_frame(self, frame, bbox_thr=0.3, nms_thr=0.8, kpt_thr=0.3):
        # predict bbox
        processing_start = time.time()

        # Mirror the frame
        frame = frame[:, ::-1, :]  # Flip horizontally for webcam mirroring

        det_result = inference_detector(self.det_model, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > bbox_thr)]
        
        self.visualizer.set_image(frame)
        if len(bboxes) > 0:
            # Sort bboxes by confidence score (column 4) in descending order
            order = np.argsort(bboxes[:, 4])[::-1]
            bboxes = bboxes[order[0], :4].reshape((1, -1))


            # predict keypoints
            pose_start = time.time()
            pose_results = inference_topdown(self.pose_model, frame, bboxes)
            data_samples = merge_data_samples(pose_results)

            # breakpoint()  

            # Visualize results
            visualization_start = time.time()
            self.visualizer.add_datasample(
                'result',
                frame,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show_kpt_idx=False,
                show=False,
                kpt_thr=kpt_thr)
        
        stop_time = time.time()
        # print("Processing time: {:.3f}\tDetection time {:.3f}\tPose time: {:.3f}\tVisualization time: {:.3f}".format(
        #     stop_time - processing_start,
        #     pose_start - processing_start,
        #     visualization_start - pose_start,
        #     stop_time - visualization_start,
        # ))
        return self.visualizer.get_image()
 
    def process(self, frame):
        """
        Main processing function called by Gradio stream.
        Stores incoming frame and returns latest processed result.
        """
        current_time = time.time()
        if self.first_call_time is None:
            self.first_call_time = current_time

        # Store the new frame in the input slot (replacing any existing frame)
        with self.input_lock:
            self.latest_input_frame = frame.copy()
            self.frame_counter += 1
            input_unique_hash = hashlib.md5(frame.tobytes()).hexdigest()
            # print(f"Input unique hash: {input_unique_hash}")
        
        # Signal that a new frame is available for processing
        self.new_frame_signal.set()
        
        # Return the latest processed output, or original frame if no processing done yet
        with self.output_lock:
            if self.latest_output_frame is not None:
                output_unique_hash = hashlib.md5(self.latest_output_frame.tobytes()).hexdigest()
                # print(f"Output unique hash: {output_unique_hash}")
                return self.latest_output_frame
            else:
                # Add indicator that this is unprocessed
                temp_frame = frame.copy()
                cv2.putText(
                    temp_frame,
                    f"Waiting... {self.frame_counter}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),  # Red for unprocessed frames
                    2,
                )
                return temp_frame
    
    def stop(self):
        """Stop the processing thread"""
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def init_models(self):
        # Init detector
        if self.det_model is None:
            print("Initializing MMDetection detector...")
            self.det_model = init_detector(DET_CFG, DET_WEIGHTS, device=DEVICE)
            self.det_model.cfg = adapt_mmdet_pipeline(self.det_model.cfg)
            print("Detector initialized successfully!")

        # Init pose estimator
        if self.pose_model is None:
            print("Initializing MMPose estimator...")
            self.pose_model = init_pose_estimator(
                POSE_CFG,
                POSE_WEIGHTS,
                device=DEVICE,
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True)))
            )
            
            # Build visualizer
            self.pose_model.cfg.visualizer.radius = 4
            self.pose_model.cfg.visualizer.alpha = 0.8
            self.pose_model.cfg.visualizer.line_width = 2
            self.visualizer = VISUALIZERS.build(self.pose_model.cfg.visualizer)
            self.visualizer.set_dataset_meta(
                self.pose_model.dataset_meta, skeleton_style='mmpose'
            )
            print("Pose estimator initialized successfully!")


# CSS for styling the Gradio interface
css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

# Initialize the asynchronous frame processor
frame_processor = AsyncFrameProcessor(processing_delay=0.5)

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    ProbPose Webcam Demo (CVPR 2025)
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        See <a href="https://MiraPurkrabek.github.io/ProbPose/" target="_blank">https://MiraPurkrabek.github.io/ProbPose/</a> for details.
        </h3>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            webcam_stream = WebRTC(
                label="Webcam Stream",
                rtc_configuration=rtc_configuration,
                track_constraints=webcam_constraints,
                mirror_webcam=True,
            )

        # Stream processing: connects webcam input to frame processor
        webcam_stream.stream(
            fn=frame_processor.process, 
            inputs=[webcam_stream], 
            outputs=[webcam_stream], 
            time_limit=None
        )

if __name__ == "__main__":
    demo.launch(
        # server_name="0.0.0.0",
        # server_port=17860,
        share=True
    )