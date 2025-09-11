# Importing the necessary libraries
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

matplotlib.use('Agg')


class KeyPoints:
    """
    使用YOLOv11-Pose模型来运行并找到图像中的人体关键点。
    function(model) - 加载模型
    function(detectPoints) - 找到图像中的关键点
    """

    def __init__(self):
        self.model_pose = None

    def model(
        self, checkpoint="yolo11n-pose.pt"
    ):  # 加载指定预训练权重的YOLOv8姿态估计模型
        """
        加载YOLOv8姿态估计模型。
        可用的模型包括: 'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt', 等。
        """
        self.model_pose = YOLO(checkpoint)

    def detectPoints(self, frame):  # 检测图像中的关键点
        """
        对输入的图像帧进行姿态估计。
        :param frame: OpenCV格式的图像 (BGR)
        :return: 第一个检测到的人的关键点坐标 (x, y)，格式为 numpy array。如果没有检测到人，则返回空列表。
        """
        # YOLOv8的predict函数可以直接处理BGR格式的Numpy数组
        results = self.model_pose.predict(
            frame, verbose=False)  # verbose=False可以关闭控制台输出

        # 检查是否检测到了任何姿态
        if results[0].keypoints.shape[0] > 0:
            # results[0].keypoints.xy 返回一个tensor，形状为 (检测到的人数, 关键点数量, 2)
            # 我们模仿原始代码的行为，只返回第一个检测到的人的关键点
            keypoints_xy = results[0].keypoints.xy[0]

            # 将结果从PyTorch Tensor转换为Numpy Array
            predict = keypoints_xy.cpu().numpy()

        else:
            # 如果没有检测到人，返回空列表
            predict = []

        return predict


class FeatureExtractor:
    """
    Used to extract features from generated keypoints.
    """

    def __init__(self):
        self.torso_up = np.array(
            [[5, 6]]
        )  # The slice used for generating the midpoint of the shoulders

        self.torso_down = np.array(
            [[11, 12]]
        )  # The slice used for generating the midpoint of the hips

        self.vector_indices = np.array(
            [
                [19, 17],
                [19, 18],
                [6, 12],
                [5, 11],
                [6, 8],
                [5, 7],
                [12, 14],
                [11, 13],
                [11, 12],
                [13, 15],
                [14, 16],
                [20, 21],
            ]
        )  # Vectors to be considered for calculating angles

        self.pair_indices = np.array(
            [[4, 2], [5, 3], [6, 10], [7, 9], [8, 6], [8, 7], [0, 11], [1, 11]]
        )  # The pairs of vectors for angle computation

        self.vertical_coordinates = np.array(
            [[1, 1], [1, 100]]
        )  # A vertical vector for comparing with other vectors

        self.angle_weights = np.ones((8, 1))  # Weights for angles
        self.cache_weights = np.ones((1, 6))  # Weights for the cache
        self.keypoints = KeyPoints()  # Initialize the keypoints class
        # Call the model method of the keypoints class to load the openpifpaf model
        self.keypoints.model()
        self.fps = 6  # Number of frames to consider in every second
        self.threshold = 10  # The threshold for fall detection

    def draw_box_string(self, img, x, y, string, color):
        """
        在图片上绘制中文字符。
        img: imread读取的图片 (numpy array);
        x,y: 字符起始绘制的位置;
        string: 显示的文字;
        color: 字体颜色，注意这里是OpenCV的BGR格式 (例如：(0, 255, 0) for green);
        return: 绘制了文字的图片 (numpy array)
        """
        # 将OpenCV的BGR图像转换为Pillow的RGB图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 加载字体文件（请确保'simhei.ttf'在您的项目路径下）
        # 字体大小可以根据需要调整
        try:
            font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
        except IOError:
            print("字体文件'simhei.ttf'未找到，请检查路径。")
            font = ImageFont.load_default()

        # Pillow使用的颜色是RGB格式，所以需要将BGR转换为RGB
        color_rgb = (color[2], color[1], color[0])

        # 在指定位置绘制文字
        draw.text((x, y), string, font=font, fill=color_rgb)

        # 将Pillow图像转换回OpenCV的BGR格式
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    def get_body_dimension(self, keypoints):
        """
        计算包裹所有可见关键点的边界框的对角线长度。
        :param keypoints: 未处理的关键点 (Numpy array)
        :return: 对角线长度，如果可见关键点少于3个则返回 None
        """
        # 过滤掉无效的关键点 (NaN 值)
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]

        # 如果可见的关键点太少，无法可靠计算尺寸
        if len(valid_points) < 3:
            return None

        x_min, y_min = np.min(valid_points, axis=0)
        x_max, y_max = np.max(valid_points, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        # 计算对角线长度
        diagonal = np.sqrt(width**2 + height**2)

        # 避免尺寸为0的情况
        return diagonal if diagonal > 10 else None

    def is_lying_down(self, keypoints, tolerance_ratio=0.3):
        """
        检查姿态是否为躺倒。
        使用身体边界框对角线长度作为动态标尺。
        :param keypoints: 包含 NaN 的原始关键点
        :param tolerance_ratio: 容忍的头脚垂直距离与身体对角线长度的比率。
                                这个比率需要比之前的小，因为对角线通常比躯干长。
                                例如 0.3 表示头脚垂直距离小于身体尺寸的30%，则认为是躺倒。
        :return: 如果是躺倒姿态，返回 True，否则返回 False
        """
        # 步骤1: 使用新的、更稳健的方法计算身体尺寸标尺
        body_size_metric = self.get_body_dimension(keypoints)

        if body_size_metric is None:
            # 如果无法计算身体尺寸，则无法进行可靠判断
            return False

        dynamic_tolerance = body_size_metric * tolerance_ratio

        # 步骤2: 计算头脚垂直距离 (这部分逻辑不变)
        head_y_coords = keypoints[[0, 5, 6], 1]
        if np.all(np.isnan(head_y_coords)):
            return False
        head_avg_y = np.nanmean(head_y_coords)

        ankle_y_coords = keypoints[[15, 16], 1]
        if np.all(np.isnan(ankle_y_coords)):
            return False
        ankle_avg_y = np.nanmean(ankle_y_coords)

        vertical_distance = abs(head_avg_y - ankle_avg_y)

        # 步骤3: 使用新的动态 tolerance 进行判断
        if vertical_distance < dynamic_tolerance:
            print(
                f"[姿态确认] 检测到躺倒姿态: 头-脚垂直距离={vertical_distance:.1f} < 动态阈值={dynamic_tolerance:.1f} (身体尺寸={body_size_metric:.1f})")
            return True

        return False

    def angleCalculation(self, vectors):
        """
        Used to calculate the angles between given pairs of vectors
        Takes as input the list of vector pairs, which represent two vectors with two coordinates each
        Returns the list of angles between them
        """

        difference = np.subtract(
            vectors[:, :, 0], vectors[:, :, 1]
        )  # Subtracts the coordinates to obtain the vectors

        dot = (difference[:, 0, :] * difference[:, 1, :]).sum(
            axis=-1
        )  # Calculates the dot product between the pairs of vectors

        norm = np.prod(
            np.linalg.norm(difference[:, :, :], axis=2), axis=-1
        )  # Calculates the norm of the vectors and multiplies them, same as |a|*|b|

        cos_angle = np.divide(dot, norm)  # cos(x) = dot(a,b)/|a|*|b|

        angle = (
            np.arccos(cos_angle) * 180 / np.pi
        )  # Take arccos of the result to get the angle

        angle = angle.reshape(-1, 1)  # Correct the shape of the output

        return angle

    def collectData(self, keypoints):
        """
        Calls handleMissingValues and addExtraPoints functions
        Used for handling negative predictions and adding extra points to the keypoints
        Takes as input the list of keypoints
        Returns the list of handled keypoints and added extra points
        """

        keypoints = self.handleMissingValues(keypoints)
        keypoints = self.addExtraPoints(keypoints)

        return keypoints

    def differenceMean(self, vector1_angles, vector2_angles):
        """
        Used for calculating the feature using differenceMean method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)
        """

        angle_difference = np.abs(
            vector1_angles - vector2_angles
        )  # Absolute difference of previous frame's angles and current frame's angles

        return (
            np.nanmean(angle_difference) * self.fps
        )  # Returns the mean of the difference multiplied by fps

    def meanDifference(self, vector1_angles, vector2_angles):
        """
        Used for calculating the feature using meanDifference method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)
        """

        return np.abs(np.nanmean(vector1_angles) - np.nanmean(vector2_angles))

    def differenceSum(self, vector1_angles, vector2_angles):
        """
        Used for calculating the feature using differenceSum method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)
        """

        angle_difference = np.abs(
            vector1_angles - vector2_angles
        )  # Absolute difference of previous and current frame angles

        # Returns the sum of angle differences
        return np.nansum(angle_difference)

    def costMean(self, vector_angles):
        """
        Used for calculating the feature using costMean method
        Takes as input the current frame angles
        Returns a scalar (the cost)
        """

        # Return the mean of the angles for the frame
        return np.nanmean(vector_angles)

    def divisionCost(self, vector1_angles, vector2_angles):
        """
        Used for calculating the feature using divisionCost method
        Takes as input the previous and current frames' angles
        Returns a scalar (the cost)
        """

        vector1_angles = np.where(
            vector1_angles == 0, np.nan, vector1_angles
        )  # If the angle is 0, replace it with nan to avoid division by zero

        angle_division = np.divide(
            vector2_angles, vector1_angles + 1e-6
        )  # Divide the current frame angles with previous frame angles

        return np.nansum(angle_division)  # Sum the result

    def handleMissingValues(self, keypoints):
        """
        Used for replacing negative predictions with NaNs
        Takes as input the list of the keypoints for the current frame
        Returns corrected list of keypoints with NaNs instead of negative values
        """

        if keypoints != []:
            keypoints = np.where(
                keypoints < 0, np.nan, keypoints
            )  # Where the points is negative replace it with NaN

        return keypoints

    def addExtraPoints(self, keypoints):
        """
        Used for adding extra points to the keypoints list
        Takes as input the keypoints for the frame
        Returns the list of keypoints with added extra points
        """
        if keypoints != []:
            torso_up = keypoints[self.torso_up].mean(
                axis=1
            )  # Get the midpoint of the shoulders using the mean of left and right shoulders

            torso_down = keypoints[self.torso_down].mean(
                axis=1
            )  # Get the midpoint of the hips using the mean of left and right shoulders

            head_coordinate = np.nanmean(
                keypoints[:5], axis=0
            )  # Get the mean of the head coordinate as one points instead of the five points

            keypoints = np.vstack(
                (
                    keypoints,
                    torso_up,
                    torso_down,
                    head_coordinate,
                    self.vertical_coordinates,
                )
            )  # Stack all the points with each other

        return keypoints

    def clip_from_to(self, costs):
        """
        Used for bounding the cost list using previously defined bounds
        Takes as input the cost list
        Returns the list of the bounded costs
        """

        sorted_ = np.sort(costs.reshape((len(costs))),
                          axis=-1)  # Sort the costs

        mean_start = np.mean(sorted_[:int(len(sorted_) * 0.1)])

        mean_end = np.mean(sorted_[len(sorted_) - int(len(sorted_) * 0.1):])

        # Bound the list with that values
        result = np.clip(costs, mean_start, mean_end)

        normalized = (result - mean_start) / (
            mean_end - mean_start
        )  # Normalize the costs using MinMaxScaling

        return normalized.reshape((len(normalized), 1))

    def chooseThreshold(self, cost_method):
        """
        Used for choosing the threshold based on the method for cost computation
        Takes as input the cost method
        Returns the threshold for that cost method
        """

        if cost_method == "DifferenceMean":
            self.threshold = 58
        elif cost_method == "DifferenceSum":
            self.threshold = 55
        elif cost_method == "MeanDifference":
            self.threshold = 5
        elif cost_method == "Mean":
            self.threshold = 37
        elif cost_method == "Division":
            self.threshold = 8.5

        return self.threshold

    def processVideo(self, video, cost_method):
        """
        Used for computing the cost for the entire video
        Takes as input the video and cost method
        Returns the list of the costs computed
        """

        camera_video = cv2.VideoCapture(video)  # Capture the video
        camera_video.set(3, 1280)  # Width of the video
        camera_video.set(4, 960)  # Height of the video
        # Get the fps of the video
        video_fps = camera_video.get(cv2.CAP_PROP_FPS)

        if video_fps != 30.0:  # If not 30 fps
            print(f"警告：视频FPS为 {video_fps}，不是30fps")
            print("尝试自动转换视频...")

            # 尝试转换视频
            try:
                from video_converter import convert_video
                converted_path = convert_video(video)
                if converted_path and os.path.exists(converted_path):
                    print(f"视频转换成功: {converted_path}")
                    # 重新打开转换后的视频
                    camera_video.release()
                    camera_video = cv2.VideoCapture(converted_path)
                    camera_video.set(3, 1280)
                    camera_video.set(4, 960)
                    video_fps = camera_video.get(cv2.CAP_PROP_FPS)
                    print(f"转换后FPS: {video_fps}")
                else:
                    print("视频转换失败，使用原始视频继续处理")
            except Exception as e:
                print(f"视频转换出错: {e}")
                print("使用原始视频继续处理")

        frame_index = 0  # Frame Index
        previous_keypoints = 0  # Variable for storing the previous keypoints
        previous_cost = 0  # Variable for storing the previous cost
        step_size = (
            video_fps // self.fps
        )  # Step size of the frames (If 5, we consider 0th frame, then fifth, then tenth, etc.)

        self.costlist = []  # List for storing costs
        cache = []  # List for storing the cache of the costs
        while camera_video.isOpened():  # While video is running
            condition, frame = camera_video.read()  # Read every frame
            if condition is False:  # If no frames left break the loop
                break
            if (
                frame_index % step_size == 0
            ):  # If the frame_index is divisible by step_size
                current_keypoints = self.keypoints.detectPoints(
                    frame
                )  # Find the keypoints for the current frame

                if frame_index == 0:  # If frame index is 0
                    previous_keypoints = (
                        current_keypoints  # Make the previous keypoints the current one
                    )
                    previous_cost = 0  # Make the previous cost 0
                    frame_index += 1  # Add 1 to frame_index and continue
                    continue

                previous_keypoints = self.collectData(
                    previous_keypoints
                )  # Handle missing values and add extra ones for previous frame

                current_keypoints = self.collectData(
                    current_keypoints
                )  # Handle missing values and add extra ones for current frame

                # 检查关键点是否有效
                if len(previous_keypoints) == 0 or len(current_keypoints) == 0:
                    continue

                try:
                    vector1_pairs = np.array(
                        previous_keypoints[self.vector_indices][self.pair_indices]
                    )  # Get vector pairs for previous keypoints

                    vector2_pairs = np.array(
                        current_keypoints[self.vector_indices][self.pair_indices]
                    )  # Get vector pairs for current keypoints
                except (IndexError, TypeError) as e:
                    print(f"关键点索引错误: {e}")
                    continue

                vector1_angles = (
                    self.angleCalculation(vector1_pairs) * self.angle_weights
                )  # Calculate the angles for previous frame and multiply with weights

                vector2_angles = (
                    self.angleCalculation(vector2_pairs) * self.angle_weights
                )  # Calculate the angles for current frame and multiply with weights

                if (
                    np.count_nonzero(np.isnan(vector1_angles)) >= 6
                    or np.count_nonzero(np.isnan(vector2_angles)) >= 6
                ):  # If more than six vectors are NaNs drop the frame and continue

                    continue

                start = time.time()  # Calculate the time for the cost computation

                if cost_method == "DifferenceMean":
                    cost = self.differenceMean(vector1_angles, vector2_angles)
                elif cost_method == "DifferenceSum":
                    cost = self.differenceSum(vector1_angles, vector2_angles)
                elif cost_method == "Division":
                    cost = self.divisionCost(vector1_angles, vector2_angles)
                elif cost_method == "Mean":
                    cost = self.costMean(vector2_angles)
                elif cost_method == "MeanDifference":
                    cost = self.meanDifference(vector1_angles, vector2_angles)
                else:
                    print(
                        'Not Valid Method!! Use "DifferenceMean", "MeanDifference", "DifferenceSum", "Division" or "Mean" as cost method!!!!'
                    )
                    return False

                end = time.time()

                if np.isnan(cost):  # If cost is NaN, take previous cost instead of NaN
                    cost = previous_cost

                cache.append(cost)  # Append the cost to cache

                if (
                    frame_index >= step_size * 6
                ):  # If the cache contains more than 5 elements
                    weighted_cost = (
                        np.dot(self.cache_weights, cache) / 6
                    )  # Calculate the cost based on previous 6 costs

                    cache = cache[
                        1:
                    ]  # Remove the last element of the cache to append the current cost

                    self.costlist.append(
                        weighted_cost
                    )  # Append the weighted cost to the cost list

                # Assign the current keypoints to the previous keypoints for the next frame
                previous_keypoints = current_keypoints

                previous_cost = (
                    cost  # Assign current cost to the previous cost for the next frame
                )

            frame_index += 1  # Add 1 to frame index
            k = cv2.waitKey(1) & 0xFF

            if k == 27:  # If esc is pressed break
                break

        camera_video.release()
        cv2.destroyAllWindows()

        return np.array(self.costlist)

    def realTimeVideo(self, video, cost_method, save=False):
        """
        Used for computing the cost for the entire video
        Takes as input the video and cost method
        Returns the list of the costs computed
        """
        falled = False
        potential_fall_frames = 0
        plot = plt.figure(figsize=(5, 5))
        camera_video = cv2.VideoCapture(video)  # Capture the video
        camera_video.set(3, 1280)  # Width of the video
        camera_video.set(4, 960)  # Height of the video

        video_fps = round(
            camera_video.get(cv2.CAP_PROP_FPS)
        )  # Get the fps of the video

        if video_fps != 30.0:  # If not 30 fps
            print(f"警告：视频FPS为 {video_fps}，不是30fps")
            print("尝试自动转换视频...")

            # 尝试转换视频
            try:
                from video_converter import convert_video
                converted_path = convert_video(video)
                if converted_path and os.path.exists(converted_path):
                    print(f"视频转换成功: {converted_path}")
                    # 重新打开转换后的视频
                    camera_video.release()
                    camera_video = cv2.VideoCapture(converted_path)
                    camera_video.set(3, 1280)
                    camera_video.set(4, 960)
                    video_fps = round(camera_video.get(cv2.CAP_PROP_FPS))
                    print(f"转换后FPS: {video_fps}")
                else:
                    print("视频转换失败，使用原始视频继续处理")
            except Exception as e:
                print(f"视频转换出错: {e}")
                print("使用原始视频继续处理")

        if save:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            frame_width = int(camera_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算合成后的图像尺寸
        plot_width = 500
        plot_height = 500
        output_width = frame_width + plot_width
        output_height = max(frame_height, plot_height)

        output_filename = f"assets/outputs/FallDetection-{os.path.basename(video).split('.')[0]}.mp4"
        out = cv2.VideoWriter(
            output_filename, fourcc, self.fps, (output_width, output_height)
        )
        print(
            f"输出视频已创建: {output_filename}, 尺寸: {output_width}x{output_height}, FPS: {self.fps}")

        frame_index = 0  # Frame Index
        previous_keypoints = 0  # Variable for storing the previous keypoints
        previous_cost = 0  # Variable for storing the previous cost
        step_size = (
            video_fps // self.fps
        )  # Step size of the frames (If 5, we consider 0th frame, then fifth, then tenth, etc.)

        self.costlist = []  # List for storing costs
        cache = []  # List for storing the cache of the costs
        while camera_video.isOpened():  # While video is running
            condition, frame = camera_video.read()  # Read every frame
            plot.canvas.draw()

            if condition is False:  # If no frames left break the loop
                break

            if (
                frame_index % step_size == 0
            ):  # If the frame_index is divisible by step_size
                current_keypoints = self.keypoints.detectPoints(
                    frame
                )  # Find the keypoints for the current frame

                if frame_index == 0:  # If frame index is 0
                    previous_keypoints = (
                        current_keypoints  # Make the previous keypoints the current one
                    )
                    previous_cost = 0  # Make the previous cost 0
                    frame_index += 1  # Add 1 to frame_index and continue
                    continue

                previous_keypoints = self.collectData(
                    previous_keypoints
                )  # Handle missing values and add extra ones for previous frame

                current_keypoints = self.collectData(
                    current_keypoints
                )  # Handle missing values and add extra ones for current frame

                # 检查关键点是否有效
                if len(previous_keypoints) == 0 or len(current_keypoints) == 0:
                    continue

                try:
                    vector1_pairs = np.array(
                        previous_keypoints[self.vector_indices][self.pair_indices]
                    )  # Get vector pairs for previous keypoints

                    vector2_pairs = np.array(
                        current_keypoints[self.vector_indices][self.pair_indices]
                    )  # Get vector pairs for current keypoints
                except (IndexError, TypeError) as e:
                    print(f"关键点索引错误: {e}")
                    continue

                vector1_angles = (
                    self.angleCalculation(vector1_pairs) * self.angle_weights
                )  # Calculate the angles for previous frame and multiply with weights

                vector2_angles = (
                    self.angleCalculation(vector2_pairs) * self.angle_weights
                )  # Calculate the angles for current frame and multiply with weights

                if (
                    np.count_nonzero(np.isnan(vector1_angles)) >= 6
                    or np.count_nonzero(np.isnan(vector2_angles)) >= 6
                ):  # If more than six vectors are NaNs drop the frame and continue

                    continue

                start = time.time()  # Calculate the time for the cost computation

                if cost_method == "DifferenceMean":
                    cost = self.differenceMean(vector1_angles, vector2_angles)
                elif cost_method == "DifferenceSum":
                    cost = self.differenceSum(vector1_angles, vector2_angles)
                elif cost_method == "Division":
                    cost = self.divisionCost(vector1_angles, vector2_angles)
                elif cost_method == "Mean":
                    cost = self.costMean(vector2_angles)
                elif cost_method == "MeanDifference":
                    cost = self.meanDifference(vector1_angles, vector2_angles)
                else:
                    print(
                        'Not Valid Method!! Use "DifferenceMean", "MeanDifference", "DifferenceSum", "Division" or "Mean" as cost method!!!!'
                    )
                    return False

                end = time.time()

                if np.isnan(cost):  # If cost is NaN, take previous cost instead of NaN
                    cost = previous_cost

                cache.append(cost)  # Append the cost to cache

                if (
                    frame_index >= step_size * 6
                ):  # If the cache contains more than 5 elements
                    weighted_cost = (
                        np.dot(self.cache_weights, cache) / 6
                    )  # Calculate the cost based on previous 6 costs
                    # print(weighted_cost)
                    weighted_cost_value = float(weighted_cost.item())
                    if weighted_cost_value > self.threshold:
                        print(
                            f"[初步触发] 成本值超限: {weighted_cost_value:.2f} > {self.threshold}")
                        # 设置一个时间窗口（例如5帧），在接下来的5帧内检查姿态
                        potential_fall_frames = 5

                    # 步骤 2: 姿态确认 - 在时间窗口内检查是否躺倒
                    if potential_fall_frames > 0:
                        # 使用当前帧的关键点进行姿态检查
                        # 注意：current_keypoints 此时已经经过 collectData 处理
                        if len(current_keypoints) > 17:  # 确保关键点有效
                            # 调用我们新写的函数，可以调整 tolerance
                            is_lying = self.is_lying_down(
                                current_keypoints, tolerance_ratio=0.15)

                            if is_lying:
                                print("!!! 跌倒确认 !!! 剧烈运动后检测到躺倒姿态。")
                                falled = True
                                # 可以在这里添加报警、变色等视觉提示
                                potential_fall_frames = 0  # 确认后重置，避免重复报警

                    # 递减时间窗口
                    potential_fall_frames -= 1

                    cache = cache[
                        1:
                    ]  # Remove the last element of the cache to append the current cost

                    self.costlist.append(
                        weighted_cost
                    )  # Append the weighted cost to the cost list

                # Assign the current keypoints to the previous keypoints for the next frame
                previous_keypoints = current_keypoints

                previous_cost = (
                    cost  # Assign current cost to the previous cost for the next frame
                )

                threshold = self.chooseThreshold(cost_method)
                # 初始化状态和颜色
                status_text = "状态: 正常"
                status_color = (0, 255, 0)  # 绿色 (BGR)

                # 根据条件更新状态和颜色
                if potential_fall_frames > 0:
                    status_text = "状态: 可能跌倒 (姿态检测中...)"
                    status_color = (0, 255, 255)  # 黄色 (BGR)
                elif falled:
                    status_text = "状态: 检测到跌倒!"
                    status_color = (0, 0, 255)  # 红色 (BGR)

                # 使用我们新的函数来绘制中文状态
                # (0, 110) 是文字的起始坐标，您可以根据需要调整
                frame = self.draw_box_string(
                    frame, 0, 110, status_text, status_color)

                plt.clf()  # Clear the plot
                plt.xlim(
                    frame_index / 5 - 15, frame_index / 5 + 15
                )  # Define the limit of x axis
                plt.ylim(0, self.threshold + 50)  # Define the limit of y axis
                plt.plot(self.costlist)  # Plot the costlist
                x_cord = [
                    frame_index / 5 - 15,
                    frame_index / 5 + 15,
                ]  # The threshold x cord
                y_cord = [threshold, threshold]  # The threshold y cord
                # Plot the threshold line
                plt.plot(x_cord, y_cord, color="red")
                plot.canvas.flush_events()  # Clears the old figure

                img = np.fromstring(
                    plot.canvas.tostring_rgb(), dtype=np.uint8, sep=""
                )  # Used to convert plot to image

                img = img.reshape(
                    plot.canvas.get_width_height()[::-1] + (3,)
                )  # Used to convert plot to image

                # Convert the image to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h1, w1 = frame.shape[:2]
                h2, w2 = img.shape[:2]
                merged = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
                merged[:, :] = (255, 255, 255)
                merged[:h1, :w1, :3] = frame
                merged[:h2, w1: w1 + w2, :3] = img

                if save:
                    out.write(merged)

                    # cv2.imshow("plot", merged)

            frame_index += 1  # Add 1 to frame index
            # k = cv2.waitKey(1) & 0xFF

            # if k == 27:  # If esc is pressed break
            #     break

        camera_video.release()

        if save:
            out.release()

        cv2.destroyAllWindows()
        print("Falled: ", falled)

    def plot(self, axis, cost, costmethod, fall_start, fall_end):
        """
        Used for plotting the cost list
        Takes as input the cost, starting frame of the fall and ending frame
        Returns the plot
        """

        threshold = self.chooseThreshold(costmethod)
        axis.plot(cost, label="cost")
        axis.set_title(f"Cost method is: {costmethod}")
        axis.axhline(y=threshold, label="Threshold", color="black")
        axis.axvspan(fall_start, fall_end, alpha=0.25,
                     color="red", label="Fall Frames")
        axis.legend(loc="upper right")

    def separatePlot(self, cost, costmethod, save=False):
        """
        Used for plotting the cost list
        Takes as input the cost, starting frame of the fall and ending frame
        Returns the plot
        """

        threshold = self.chooseThreshold(costmethod)
        plot = plt.figure(figsize=(10, 10))
        plt.plot(cost, label="cost")
        plt.title(f"Cost method is: {costmethod}")
        plt.legend(loc="upper right")
        plot.canvas.draw()

        img = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        img = img.reshape(plot.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if save is True:
            cv2.imwrite("FallDetection.png", img)

        cv2.imshow("plot", img)
        k = cv2.waitKey(10000) & 0xFF
        if k == 27:  # If esc is pressed break
            cv2.destroyAllWindows()
