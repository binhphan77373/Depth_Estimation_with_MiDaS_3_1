# import cv2
# import torch
# import numpy as np
# import supervision as sv
# from enum import Enum
# from ultralytics import YOLO
# import time

# # ------------------------- YOLO Model Setup -------------------------
# class YOLOv11:
#     def __init__(self, model_path="yolo11n.pt"):
#         self.model = YOLO(model_path)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device).eval()

#     def predict(self, frame):
#         results = self.model(frame)[0]  # Run inference on frame
#         detections = sv.Detections.from_ultralytics(results)  # Convert results to supervision format
#         return detections

# # ------------------------- MiDaS Depth Estimation -------------------------
# class ModelType(Enum):
#     DPT_LARGE = "DPT_Large"     # High accuracy, slowest inference
#     DPT_HYBRID = "DPT_Hybrid"   # Medium accuracy, balanced speed
#     MIDAS_SMALL = "MiDaS_small" # Low accuracy, fastest inference

# class Midas:
#     def __init__(self, modelType=ModelType.DPT_LARGE):
#         self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
#         self.modelType = modelType
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.midas.to(self.device).eval()

#         # Load MiDaS Transformations
#         midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         if modelType.value in ["DPT_Large", "DPT_Hybrid"]:
#             self.transform = midas_transforms.dpt_transform
#         else:
#             self.transform = midas_transforms.small_transform

#     def predict(self, frame):
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = self.transform(img).to(self.device)

#         with torch.no_grad():
#             prediction = self.midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()
#         depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#         depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
#         return depth_map

# # ------------------------- Real-Time Processing -------------------------
# class RealTimeProcessing:
#     def __init__(self, yolo_model="yolo11n.pt", depth_model=ModelType.MIDAS_SMALL):
#         self.yolo = YOLOv11(yolo_model)
#         self.midas = Midas(depth_model)
#         self.bounding_box_annotator = sv.BoundingBoxAnnotator()
#         self.label_annotator = sv.LabelAnnotator()

#     def live_predict(self, video_source="demo.mp4", output_video="output.avi"):
#         cap = cv2.VideoCapture(video_source)  # Use 0 for webcam, or "demo.mp4" for video
#         if not cap.isOpened():
#             print("Error: Could not open video file.")
#             return

#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width * 2, frame_height))

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # YOLOv11 Object Detection
#             detections = self.yolo.predict(frame)

#             # MiDaS Depth Estimation
#             depth_map = self.midas.predict(frame)

#             # Annotate detections with bounding boxes and labels
#             annotated_frame = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
#             annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

#             # Stack images side-by-side
#             combined = np.hstack((annotated_frame, depth_map))

#             # Save frame to video
#             out.write(combined)

#             # Display the frame
#             cv2.imshow("YOLOv11 + MiDaS Depth Estimation", combined)

#             # Exit condition
#             if cv2.waitKey(25) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()

# # ------------------------- Run Application -------------------------
# if __name__ == "__main__":
#     print("Starting YOLOv11 + MiDaS real-time processing...")
#     processor = RealTimeProcessing(yolo_model="yolo11n.pt", depth_model=ModelType.MIDAS_SMALL)
#     processor.live_predict(video_source="demo.mp4", output_video="output.mp4")  # Change to 0 for live webcam

# import cv2
# import torch
# import numpy as np
# import supervision as sv
# from enum import Enum
# from ultralytics import YOLO
# import time
# import os

# # ------------------------- YOLO Model Setup -------------------------
# class YOLOv11:
#     def __init__(self, model_path="yolo11x.pt"):
#         self.model = YOLO(model_path)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device).eval()

#     def predict(self, frame):
#         results = self.model(frame)[0]  # Run inference on frame
#         detections = sv.Detections.from_ultralytics(results)  # Convert results to supervision format
#         return detections

# # ------------------------- MiDaS Depth Estimation -------------------------
# class ModelType(Enum):
#     DPT_LARGE = "DPT_Large"     # High accuracy, slowest inference
#     DPT_HYBRID = "DPT_Hybrid"   # Medium accuracy, balanced speed
#     MIDAS_SMALL = "MiDaS_small" # Low accuracy, fastest inference

# class Midas:
#     def __init__(self, modelType=ModelType.DPT_LARGE):
#         self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
#         self.modelType = modelType
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.midas.to(self.device).eval()

#         # Load MiDaS Transformations
#         midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         if modelType.value in ["DPT_Large", "DPT_Hybrid"]:
#             self.transform = midas_transforms.dpt_transform
#         else:
#             self.transform = midas_transforms.small_transform

#     def predict(self, frame):
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = self.transform(img).to(self.device)

#         with torch.no_grad():
#             prediction = self.midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()
#         depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#         depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
#         return depth_map

# # ------------------------- Real-Time Processing -------------------------
# class RealTimeProcessing:
#     def __init__(self, yolo_model="yolo11x.pt", depth_model=ModelType.MIDAS_SMALL):
#         self.yolo = YOLOv11(yolo_model)
#         self.midas = Midas(depth_model)
#         self.bounding_box_annotator = sv.BoundingBoxAnnotator()
#         self.label_annotator = sv.LabelAnnotator()

#     def live_predict(self, video_source="demo.mp4", output_video="output.avi", show_display=False):
#         cap = cv2.VideoCapture(video_source)  # Use 0 for webcam, or "demo.mp4" for video
#         if not cap.isOpened():
#             print("Error: Could not open video file.")
#             return

#         frame_width = int(cap.get(3))
#         frame_height = int(cap.get(4))
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width * 2, frame_height))

#         frame_count = 0
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # YOLOv11 Object Detection
#                 detections = self.yolo.predict(frame)

#                 # MiDaS Depth Estimation
#                 depth_map = self.midas.predict(frame)

#                 # Annotate detections with bounding boxes and labels
#                 annotated_frame = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
#                 annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)

#                 # Stack images side-by-side
#                 combined = np.hstack((annotated_frame, depth_map))

#                 # Save frame to video
#                 out.write(combined)

#                 # Optional display (only if GUI support is available)
#                 if show_display:
#                     try:
#                         cv2.imshow("YOLOv11 + MiDaS Depth Estimation", combined)
#                         if cv2.waitKey(25) & 0xFF == ord("q"):
#                             break
#                     except cv2.error:
#                         pass  # Silently continue if display fails

#                 frame_count += 1
#                 if frame_count % 30 == 0:  # Print progress every 30 frames
#                     print(f"Processed {frame_count} frames...")

#         except KeyboardInterrupt:
#             print("\nProcessing interrupted by user")
#         finally:
#             cap.release()
#             out.release()
#             if show_display:
#                 cv2.destroyAllWindows()
#             print(f"\nProcessing complete. Output saved to {output_video}")

# # ------------------------- Run Application -------------------------
# if __name__ == "__main__":
#     print("Starting YOLOv11 + MiDaS real-time processing...")
#     processor = RealTimeProcessing(yolo_model="yolo11x.pt", depth_model=ModelType.DPT_LARGE)
    
#     has_display = "DISPLAY" in os.environ or os.name == 'nt'
    
#     processor.live_predict(
#         video_source="demo.mp4", 
#         output_video="output.mp4",
#         show_display=has_display
#     )

import cv2
import torch
import numpy as np
import supervision as sv
from enum import Enum
from ultralytics import YOLO
import time
import os

# ------------------------- YOLO Model Setup -------------------------
class YOLOv11:
    def __init__(self, model_path="yolo11x.pt"):
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, frame):
        results = self.model(frame)[0]  # Run inference on frame
        detections = sv.Detections.from_ultralytics(results)  # Convert results to supervision format
        return detections

# ------------------------- MiDaS Depth Estimation -------------------------
class ModelType(Enum):
    DPT_LARGE = "DPT_Large"     # High accuracy, slowest inference
    DPT_HYBRID = "DPT_Hybrid"   # Medium accuracy, balanced speed
    MIDAS_SMALL = "MiDaS_small" # Low accuracy, fastest inference

class Midas:
    def __init__(self, modelType=ModelType.DPT_LARGE):
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device).eval()

        # Load MiDaS Transformations
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if modelType.value in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Lưu depth_map gốc cho tính toán khoảng cách
        self.original_depth = depth_map.copy()
        
        # Normalize depth map cho hiển thị
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        return depth_map

    def estimate_distance(self, bbox):
        """
        Ước tính khoảng cách đến đối tượng dựa trên depth map
        bbox: [x1, y1, x2, y2] - tọa độ bounding box
        """
        if not hasattr(self, 'original_depth'):
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        
        # Lấy vùng depth trong bounding box
        depth_roi = self.original_depth[y1:y2, x1:x2]
        
        if depth_roi.size == 0:
            return None
            
        # Tính trung bình độ sâu trong ROI
        avg_depth = np.median(depth_roi)
        
        # Chuyển đổi giá trị độ sâu thành meters
        # Hệ số này cần được điều chỉnh dựa trên calibration của camera
        distance_meters = avg_depth * 0.1  # Giả sử hệ số chuyển đổi
        
        return distance_meters

# ------------------------- Real-Time Processing -------------------------
class RealTimeProcessing:
    def __init__(self, yolo_model="yolo11x.pt", depth_model=ModelType.MIDAS_SMALL):
        self.yolo = YOLOv11(yolo_model)
        self.midas = Midas(depth_model)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def live_predict(self, video_source="demo.mp4", output_video="output.avi", show_display=False):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width * 2, frame_height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLOv11 Object Detection
                detections = self.yolo.predict(frame)

                # MiDaS Depth Estimation
                depth_map = self.midas.predict(frame)

                # Tính và hiển thị khoảng cách cho mỗi đối tượng
                annotated_frame = frame.copy()
                for i in range(len(detections.xyxy)):
                    bbox = detections.xyxy[i]
                    distance = self.midas.estimate_distance(bbox)
                    
                    if distance is not None:
                        # Cập nhật nhãn với khoảng cách
                        label = f"{detections.class_id[i]}: {distance:.2f}m"
                        
                        # Vẽ bounding box và nhãn
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Stack images side-by-side
                combined = np.hstack((annotated_frame, depth_map))

                # Save frame to video
                out.write(combined)

                # Optional display
                if show_display:
                    try:
                        cv2.imshow("YOLOv11 + MiDaS Depth Estimation", combined)
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        pass

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            cap.release()
            out.release()
            if show_display:
                cv2.destroyAllWindows()
            print(f"\nProcessing complete. Output saved to {output_video}")

# ------------------------- Run Application -------------------------
if __name__ == "__main__":
    print("Starting YOLOv11 + MiDaS real-time processing with distance estimation...")
    processor = RealTimeProcessing(yolo_model="yolo11x.pt", depth_model=ModelType.DPT_LARGE)
    
    has_display = "DISPLAY" in os.environ or os.name == 'nt'
    
    processor.live_predict(
        video_source="demo.mp4", 
        output_video="output.mp4",
        show_display=has_display
    )