# 导入必要的库
import os
import time
from typing import List, Optional, Tuple, Dict, Callable

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 解决matplotlib在无GUI环境下运行的问题
matplotlib.use('Agg')


class PoseEstimator:
    """
    负责加载YOLOv8姿态模型并进行关键点检测。
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        """
        初始化并加载YOLOv8姿态估计模型。

        Args:
            model_path (str): 预训练模型权重文件的路径。
                可选: 'yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', etc.
        """
        print(f"正在加载姿态估计模型: {model_path}...")
        self.model = YOLO(model_path)
        print("模型加载完毕。")

    def detect_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        对输入的图像帧进行姿态估计，并返回第一个检测到的人的关键点。

        Args:
            frame (np.ndarray): OpenCV格式的图像 (BGR)。

        Returns:
            np.ndarray: 第一个检测到的人的关键点坐标 (x, y)，格式为 numpy array。
                        如果没有检测到人，则返回一个空数组。
        """
        results = self.model.predict(frame, verbose=False)

        if results and results[0].keypoints and results[0].keypoints.shape[0] > 0:
            keypoints_xy = results[0].keypoints.xy[0]
            return keypoints_xy.cpu().numpy()
        else:
            return np.array([])


class PoseProcessor:
    """
    负责处理、增强和分析姿态关键点数据。
    """

    def __init__(self):
        # 定义关键点索引
        self.TORSO_UP_SLICE = np.array([[5, 6]])      # 肩膀中点
        self.TORSO_DOWN_SLICE = np.array([[11, 12]])  # 臀部中点

        # 用于计算角度的向量索引
        self.VECTOR_INDICES = np.array([
            [19, 17], [19, 18], [6, 12], [5, 11], [6, 8], [5, 7],
            [12, 14], [11, 13], [11, 12], [13, 15], [14, 16], [20, 21],
        ])

        # 用于计算角度的向量对索引
        self.PAIR_INDICES = np.array([
            [4, 2], [5, 3], [6, 10], [7, 9], [8, 6], [8, 7], [0, 11], [1, 11]
        ])

        # 用于与身体向量比较的垂直向量
        self.VERTICAL_VECTOR_COORDS = np.array([[1, 1], [1, 100]])

    def preprocess_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        对关键点进行预处理：处理无效值并添加额外的辅助点。

        Args:
            keypoints (np.ndarray): 原始关键点数组。

        Returns:
            np.ndarray: 经过处理和增强的关键点数组。
        """
        if keypoints.size == 0:
            return np.array([])

        # 1. 处理无效值 (将负坐标替换为NaN)
        processed_kps = np.where(keypoints < 0, np.nan, keypoints)

        # 2. 添加额外的辅助点
        # 计算肩膀中点
        torso_up = np.nanmean(processed_kps[self.TORSO_UP_SLICE], axis=1)
        # 计算臀部中点
        torso_down = np.nanmean(processed_kps[self.TORSO_DOWN_SLICE], axis=1)
        # 计算头部中心点
        head_center = np.nanmean(processed_kps[:5], axis=0, keepdims=True)

        # 将新点堆叠到原始关键点数组中
        return np.vstack([
            processed_kps,
            torso_up,         # index 17
            torso_down,       # index 18
            head_center,      # index 19
            self.VERTICAL_VECTOR_COORDS  # index 20, 21
        ])

    def calculate_angles_from_keypoints(self, keypoints: np.ndarray, angle_weights: np.ndarray) -> Optional[np.ndarray]:
        """
        从处理后的关键点计算身体角度。

        Args:
            keypoints (np.ndarray): 经过 preprocess_keypoints 处理后的关键点。
            angle_weights (np.ndarray): 角度权重。

        Returns:
            Optional[np.ndarray]: 计算出的角度数组，如果关键点不足则返回 None。
        """
        try:
            vector_pairs = keypoints[self.VECTOR_INDICES][self.PAIR_INDICES]

            # 计算向量
            vectors = np.subtract(vector_pairs[:, :, 0], vector_pairs[:, :, 1])

            # 计算点积
            dot_product = (vectors[:, 0, :] * vectors[:, 1, :]).sum(axis=-1)

            # 计算模的乘积
            norm_product = np.prod(np.linalg.norm(vectors, axis=2), axis=-1)

            # 防止除以零
            norm_product[norm_product == 0] = 1e-6

            cos_angle = np.divide(dot_product, norm_product)
            # 限制cos_angle在[-1, 1]范围内，避免计算错误
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            angles = np.arccos(cos_angle) * 180 / np.pi

            return angles.reshape(-1, 1) * angle_weights

        except (IndexError, TypeError) as e:
            print(f"计算角度时出错: {e}")
            return None

    def get_body_dimension(self, keypoints: np.ndarray) -> Optional[float]:
        """
        计算包裹所有可见关键点的边界框的对角线长度，作为身体尺寸的度量。

        Args:
            keypoints (np.ndarray): 原始关键点 (包含NaN)。

        Returns:
            Optional[float]: 对角线长度，如果可见关键点少于3个则返回 None。
        """
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_points) < 3:
            return None

        x_min, y_min = np.min(valid_points, axis=0)
        x_max, y_max = np.max(valid_points, axis=0)

        diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        return diagonal if diagonal > 10 else None

    def is_lying_down(self, keypoints: np.ndarray, tolerance_ratio: float = 0.3) -> bool:
        """
        检查姿态是否为躺倒。

        Args:
            keypoints (np.ndarray): 原始关键点 (包含NaN)。
            tolerance_ratio (float): 头脚垂直距离与身体对角线长度的比率阈值。

        Returns:
            bool: 如果是躺倒姿态，返回 True。
        """
        body_size = self.get_body_dimension(keypoints)
        if body_size is None:
            return False

        dynamic_tolerance = body_size * tolerance_ratio

        head_y_coords = keypoints[[0, 1, 2, 3, 4], 1]
        ankle_y_coords = keypoints[[15, 16], 1]

        if np.all(np.isnan(head_y_coords)) or np.all(np.isnan(ankle_y_coords)):
            return False

        head_avg_y = np.nanmean(head_y_coords)
        ankle_avg_y = np.nanmean(ankle_y_coords)
        vertical_distance = abs(head_avg_y - ankle_avg_y)

        if vertical_distance < dynamic_tolerance:
            print(
                f"[姿态确认] 检测到躺倒姿态: 头-脚垂直距离={vertical_distance:.1f} < "
                f"动态阈值={dynamic_tolerance:.1f} (身体尺寸={body_size:.1f})"
            )
            return True
        return False


class FallVisualizer:
    """
    负责所有可视化任务，如绘制文本、图表和拼接图像。
    """

    def __init__(self, font_path: str = "simhei.ttf", font_size: int = 30):
        try:
            self.font = ImageFont.truetype(
                font_path, font_size, encoding="utf-8")
        except IOError:
            print(f"警告: 字体文件 '{font_path}' 未找到，将使用默认字体。")
            self.font = ImageFont.load_default()

    def draw_status(self, frame: np.ndarray, text: str, color: Tuple[int, int, int], position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        在图像上绘制状态文本（支持中文）。

        Args:
            frame (np.ndarray): OpenCV BGR 格式图像。
            text (str): 要绘制的文本。
            color (Tuple[int, int, int]): BGR 格式的颜色。
            position (Tuple[int, int]): 文本左上角坐标 (x, y)。

        Returns:
            np.ndarray: 绘制了文本的图像。
        """
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 将 BGR 转换为 Pillow 使用的 RGB
        color_rgb = (color[2], color[1], color[0])
        draw.text(position, text, font=self.font, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_plot_image(self, costs: List[float], threshold: float, current_x: int,
                          plot_size: Tuple[int, int] = (500, 500)) -> np.ndarray:
        """
        创建一个成本曲线的图像。

        Args:
            costs (List[float]): 成本值列表。
            threshold (float): 阈值线。
            current_x (int): 当前图表的 x 轴中心点。
            plot_size (Tuple[int, int]): 输出图像的尺寸 (宽, 高)。

        Returns:
            np.ndarray: BGR 格式的图表图像。
        """
        fig = plt.figure(figsize=(plot_size[0] / 100, plot_size[1] / 100))
        plt.clf()
        plt.ylim(0, threshold + 50)
        plt.plot(costs, color='blue')
        plt.axhline(y=threshold, color='red', linestyle='--',
                    label=f'Threshold: {threshold:.2f}')
        plt.title("Fall Detection Cost")
        plt.xlabel("Frame Sequence")
        plt.ylabel("Cost")
        plt.legend()
        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def combine_frame_and_plot(self, frame: np.ndarray, plot_image: np.ndarray) -> np.ndarray:
        """
        将视频帧和图表图像水平拼接。

        Args:
            frame (np.ndarray): 视频帧。
            plot_image (np.ndarray): 图表图像。

        Returns:
            np.ndarray: 拼接后的图像。
        """
        h1, w1 = frame.shape[:2]
        h2, w2 = plot_image.shape[:2]

        output_height = max(h1, h2)
        output_width = w1 + w2

        merged_frame = np.full(
            (output_height, output_width, 3), 255, dtype=np.uint8)
        merged_frame[:h1, :w1] = frame
        merged_frame[:h2, w1:] = plot_image

        return merged_frame


class FallDetector:
    """
    跌倒检测系统的核心控制器。
    """

    def __init__(self,
                 model_path: str = "yolo11n-pose.pt",
                 target_fps: int = 6,
                 lying_tolerance_ratio: float = 0.3):

        self.target_fps = target_fps
        self.lying_tolerance_ratio = lying_tolerance_ratio

        # 初始化模块
        self.estimator = PoseEstimator(model_path)
        self.processor = PoseProcessor()
        self.visualizer = FallVisualizer()

        # 初始化权重
        self.angle_weights = np.ones((8, 1))
        self.cache_weights = np.ones((1, 6))

        # 定义成本计算方法
        self.cost_methods: Dict[str, Callable] = {
            "DifferenceMean": self._cost_difference_mean,
            "DifferenceSum": self._cost_difference_sum,
            "MeanDifference": self._cost_mean_difference,
            "Mean": lambda _, angles2: np.nanmean(angles2),
            "Division": self._cost_division
        }

        # 定义不同方法的阈值
        self.thresholds: Dict[str, float] = {
            "DifferenceMean": 58,
            "DifferenceSum": 55,
            "MeanDifference": 5,
            "Mean": 37,
            "Division": 8.5
        }

        self.reset()

    def reset(self):
        """重置检测器的内部状态。"""
        self.costs: List[float] = []
        self.cost_cache: List[float] = []
        self.previous_keypoints: Optional[np.ndarray] = None
        self.previous_angles: Optional[np.ndarray] = None
        self.potential_fall_frames: int = 0
        self.fall_detected: bool = False

    # --- 成本计算方法 ---
    def _cost_difference_mean(self, angles1: np.ndarray, angles2: np.ndarray) -> float:
        return np.nanmean(np.abs(angles1 - angles2)) * self.target_fps

    def _cost_difference_sum(self, angles1: np.ndarray, angles2: np.ndarray) -> float:
        return np.nansum(np.abs(angles1 - angles2))

    def _cost_mean_difference(self, angles1: np.ndarray, angles2: np.ndarray) -> float:
        return np.abs(np.nanmean(angles1) - np.nanmean(angles2))

    def _cost_division(self, angles1: np.ndarray, angles2: np.ndarray) -> float:
        angles1_safe = np.where(angles1 == 0, 1e-6, angles1)
        return np.nansum(np.divide(angles2, angles1_safe))

    def _calculate_cost(self, method: str, angles1: np.ndarray, angles2: np.ndarray) -> Optional[float]:
        """根据指定方法计算成本。"""
        if method not in self.cost_methods:
            print(f"错误: 无效的成本方法 '{method}'。")
            return None
        return self.cost_methods[method](angles1, angles2)

    def process_video(self, video_path: str, cost_method: str, save_output: bool = True):
        """
        处理视频文件以进行跌倒检测。

        Args:
            video_path (str): 输入视频文件的路径。
            cost_method (str): 用于计算的成本方法。
            save_output (bool): 是否保存带可视化的输出视频。
        """
        self.reset()
        threshold = self.thresholds.get(cost_method)
        if threshold is None:
            print(f"错误: 成本方法 '{cost_method}' 的阈值未定义。")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        step_size = max(1, round(video_fps / self.target_fps))
        print(
            f"视频FPS: {video_fps:.2f}, 目标处理FPS: {self.target_fps}, 帧步长: {step_size}")

        out = None
        if save_output:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_width = frame_width + 500  # 视频帧 + 图表宽度
            output_height = max(frame_height, 500)

            output_filename = f"assets/outputs/FallDetection_{os.path.basename(video_path)}"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_filename, fourcc,
                                  self.target_fps, (output_width, output_height))
            print(f"结果将保存到: {output_filename}")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step_size == 0:
                current_kps_raw = self.estimator.detect_keypoints(frame)

                if current_kps_raw.size == 0:
                    frame_idx += 1
                    continue

                current_kps = self.processor.preprocess_keypoints(
                    current_kps_raw)
                current_angles = self.processor.calculate_angles_from_keypoints(
                    current_kps, self.angle_weights)

                if self.previous_angles is not None and current_angles is not None:
                    # 如果有太多NaN值，则跳过此帧
                    if np.count_nonzero(np.isnan(self.previous_angles)) >= 6 or \
                       np.count_nonzero(np.isnan(current_angles)) >= 6:
                        self.previous_keypoints = current_kps
                        self.previous_angles = current_angles
                        frame_idx += 1
                        continue

                    # 计算成本
                    cost = self._calculate_cost(
                        cost_method, self.previous_angles, current_angles)
                    if cost is not None and not np.isnan(cost):
                        self.cost_cache.append(cost)

                        if len(self.cost_cache) >= 6:
                            weighted_cost = np.dot(
                                self.cache_weights, self.cost_cache) / 6
                            self.costs.append(float(weighted_cost.item()))
                            self.cost_cache.pop(0)  # 维持缓存大小

                            # --- 跌倒检测逻辑 ---
                            if float(weighted_cost.item()) > threshold:
                                print(
                                    f"[初步触发] 成本值超限: {weighted_cost.item():.2f} > {threshold}")
                                self.potential_fall_frames = 5  # 设置姿态确认窗口

                            if self.potential_fall_frames > 0:
                                if self.processor.is_lying_down(current_kps_raw, self.lying_tolerance_ratio):
                                    print("!!! 跌倒确认 !!! 剧烈运动后检测到躺倒姿态。")
                                    self.fall_detected = True
                                    self.potential_fall_frames = 0  # 确认后重置
                                else:
                                    self.potential_fall_frames -= 1

                # 更新状态
                self.previous_keypoints = current_kps
                self.previous_angles = current_angles

                # --- 可视化 ---
                if save_output and out:
                    # 确定状态和颜色
                    if self.fall_detected:
                        status_text, color = "状态: 检测到跌倒!", (0, 0, 255)  # 红
                    elif self.potential_fall_frames > 0:
                        # 黄
                        status_text, color = "状态: 可能跌倒 (姿态检测中...)", (0,
                                                                     255, 255)
                    else:
                        status_text, color = "状态: 正常", (0, 255, 0)  # 绿

                    frame_with_status = self.visualizer.draw_status(
                        frame, status_text, color)
                    plot_img = self.visualizer.create_plot_image(
                        self.costs, threshold, len(self.costs))
                    combined_frame = self.visualizer.combine_frame_and_plot(
                        frame_with_status, plot_img)

                    out.write(combined_frame)
                    # cv2.imshow("Fall Detection", combined_frame) # 可选：实时显示处理结果
                    # if cv2.waitKey(1) & 0xFF == 27:
                    #     break

            frame_idx += 1

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"处理完成。最终检测结果 - 是否跌倒: {self.fall_detected}")


if __name__ == '__main__':
    # --- 使用示例 ---
    # 1. 创建FallDetector实例
    # 可以指定模型路径，例如 "yolov8m-pose.pt" 以获得更高精度
    detector = FallDetector(model_path="yolov8n-pose.pt", target_fps=6)

    # 2. 定义输入视频和要使用的成本计算方法
    # 确保你的视频文件路径正确
    video_file = "assets/inputs/fall.mp4"  # <--- 修改为你的视频路径
    # 可选的成本方法: "DifferenceMean", "DifferenceSum", "MeanDifference", "Mean", "Division"
    selected_cost_method = "DifferenceMean"

    # 3. 运行处理
    if os.path.exists(video_file):
        detector.process_video(
            video_path=video_file,
            cost_method=selected_cost_method,
            save_output=True  # 设置为True来保存带图表的视频，False则只进行分析
        )
    else:
        print(f"错误: 视频文件未找到 -> {video_file}")
