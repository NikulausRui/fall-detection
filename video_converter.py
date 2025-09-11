#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频转换工具：将任意fps和方向的视频转换为30fps的1280x960格式
"""

import os
import subprocess
import tempfile

import cv2
import numpy as np


def detect_video_orientation(video_path):
    """检测视频方向"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    is_landscape = width > height
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'is_landscape': is_landscape,
        'aspect_ratio': width / height
    }


def convert_video_with_ffmpeg(input_path, output_path, target_fps=30, target_width=1280, target_height=960):
    """
    使用ffmpeg转换视频，支持横屏和竖屏

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_fps: 目标fps (默认30)
        target_width: 目标宽度 (默认1280)
        target_height: 目标高度 (默认960)
    """

    # 检测视频信息
    video_info = detect_video_orientation(input_path)
    print("原始视频信息:")
    print(f"  尺寸: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']}")
    print(f"  方向: {'横屏' if video_info['is_landscape'] else '竖屏'}")

    # 构建ffmpeg命令
    if video_info['is_landscape']:
        # 横屏视频：直接缩放
        scale_filter = f"scale={target_width}:{target_height}"
    else:
        # 竖屏视频：先缩放保持比例，再填充
        # 计算缩放比例
        scale_w = target_width / video_info['width']
        scale_h = target_height / video_info['height']
        scale = min(scale_w, scale_h)

        new_width = int(video_info['width'] * scale)
        new_height = int(video_info['height'] * scale)

        # 使用scale和pad滤镜
        scale_filter = f"scale={new_width}:{new_height},pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"

    # 构建完整的ffmpeg命令
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', scale_filter,
        '-r', str(target_fps),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # 覆盖输出文件
        output_path
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        # 执行ffmpeg命令
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        print("✅ 视频转换成功")

        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"输出文件大小: {file_size} 字节")

            # 验证输出视频信息
            cap = cv2.VideoCapture(output_path)
            out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            print("转换后视频信息:")
            print(f"  尺寸: {out_width}x{out_height}")
            print(f"  FPS: {out_fps}")

            return True
        else:
            print("❌ 输出文件不存在")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ 未找到ffmpeg，请先安装ffmpeg")
        print("安装方法:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: 下载并安装 https://ffmpeg.org/download.html")
        return False


def convert_video_python_fallback(input_path, output_path, target_fps=30, target_width=1280, target_height=960):
    """
    使用Python OpenCV作为ffmpeg的备选方案
    """
    print("使用Python OpenCV进行视频转换...")

    cap = cv2.VideoCapture(input_path)
    video_info = detect_video_orientation(input_path)

    # 设置输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, target_fps,
                          (target_width, target_height))

    if not out.isOpened():
        print("❌ 无法创建输出视频文件")
        return False

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_info['is_landscape']:
            # 横屏视频：直接缩放
            resized_frame = cv2.resize(frame, (target_width, target_height))
        else:
            # 竖屏视频：先缩放保持比例，再填充
            scale = min(
                target_width / video_info['width'], target_height / video_info['height'])
            new_width = int(video_info['width'] * scale)
            new_height = int(video_info['height'] * scale)

            # 缩放视频
            resized = cv2.resize(frame, (new_width, new_height))

            # 创建目标尺寸的黑色画布
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            # 计算居中位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            # 将缩放后的视频放在画布中央
            canvas[y_offset:y_offset+new_height,
                   x_offset:x_offset+new_width] = resized
            resized_frame = canvas

        out.write(resized_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()
    out.release()

    print(f"✅ Python转换完成，处理了 {frame_count} 帧")
    return True


def convert_video(input_path, output_path=None, use_ffmpeg=True):
    """
    主转换函数

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径，如果为None则自动生成
        use_ffmpeg: 是否优先使用ffmpeg
    """

    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return None

    if output_path is None:
        # 自动生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_converted_30fps.mp4"

    print(f"开始转换视频: {input_path} -> {output_path}")

    # 首先尝试使用ffmpeg
    if use_ffmpeg:
        success = convert_video_with_ffmpeg(input_path, output_path)
        if success:
            return output_path
        else:
            print("ffmpeg转换失败，尝试使用Python OpenCV...")

    # 如果ffmpeg失败，使用Python OpenCV
    success = convert_video_python_fallback(input_path, output_path)
    if success:
        return output_path
    else:
        print("❌ 所有转换方法都失败了")
        return None


if __name__ == "__main__":
    # 测试转换功能
    test_video = "./assets/videos/fall.mp4"
    if os.path.exists(test_video):
        result = convert_video(test_video)
        if result:
            print(f"转换成功: {result}")
        else:
            print("转换失败")
    else:
        print(f"测试视频不存在: {test_video}")
