"""
简单的框选水印位置，移除水印
    1、代码首先获取视频的第一个有效帧，用于选择水印的ROI（感兴趣区域）。
    2、然后用户可以通过交互式界面在视频中选择水印的ROI，以便后续处理。
    3、接着，通过自适应的方法检测水印并生成水印的遮罩。
    4、最后，利用生成的水印遮罩，对视频进行修复，去除水印。
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm


def get_first_valid_frame(video_clip, threshold=10, num_frames=10):
    """
    获取视频的第一个有效帧，用于选择水印的ROI
    :param video_clip: 视频剪辑对象
    :param threshold: 判断帧是否有效的阈值
    :param num_frames: 用于选择的帧的数量
    :return: 第一个有效帧的图像数据
    """
    total_frames = int(video_clip.fps * video_clip.duration)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx in frame_indices:
        frame = video_clip.get_frame(idx / video_clip.fps)
        if frame.mean() > threshold:
            return frame
    # 注意：不一定第一帧就有水印
    return video_clip.get_frame(0)


def select_roi_for_mask(video_clip):
    """
    从视频剪辑中选择水印的ROI
    :param video_clip: 视频剪辑对象
    :return: 水印ROI的坐标和尺寸 (x, y, w, h)
    """

    frame = get_first_valid_frame(video_clip)

    # 将视频帧调整为720p显示
    display_height = 720
    scale_factor = display_height / frame.shape[0]
    display_width = int(frame.shape[1] * scale_factor)
    display_frame = cv2.resize(frame, (display_width, display_height))

    instructions = "Select ROI and press SPACE or ENTER"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_frame, instructions, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    r = cv2.selectROI(display_frame)
    cv2.destroyAllWindows()

    r_original = (
    int(r[0] / scale_factor), int(r[1] / scale_factor), int(r[2] / scale_factor), int(r[3] / scale_factor))

    return r_original


def detect_watermark_adaptive(frame, roi):
    """
    自适应检测水印并生成遮罩。
    :param frame: 视频帧的图像数据
    :param roi: 水印的ROI坐标和尺寸 (x, y, w, h)。
    :return: 水印的遮罩图像数据。
    """
    roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = binary_frame

    return mask


def generate_watermark_mask(video_clip, num_frames=10, min_frame_count=7):
    """
    生成水印的遮罩
    :param video_clip: 视频剪辑对象
    :param num_frames: 用于生成遮罩的帧的数量
    :param min_frame_count: 水印像素点在至少多少帧中出现才被认为是水印
    :return: 水印的遮罩图像数据
    """
    total_frames = int(video_clip.duration * video_clip.fps)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = [video_clip.get_frame(idx / video_clip.fps) for idx in frame_indices]
    r_original = select_roi_for_mask(video_clip)

    masks = [detect_watermark_adaptive(frame, r_original) for frame in frames]

    final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
    # 根据像素点在至少min_frame_count张以上的帧中的出现来生成最终的遮罩
    final_mask = np.where(final_mask >= min_frame_count, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(final_mask, kernel)


def process_video(video_clip, output_path, apply_mask_func):
    """
    处理视频并保存结果
    :param video_clip: 视频剪辑对象
    :param output_path: 输出视频路径
    :param apply_mask_func: 应用遮罩的函数
    :return:
    """
    total_frames = int(video_clip.duration * video_clip.fps)
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")

    def process_frame(frame):
        result = apply_mask_func(frame)
        progress_bar.update(1000)
        return result

    processed_video = video_clip.fl_image(process_frame, apply_to=["each"])
    processed_video.write_videofile(f"{output_path}.mp4", codec="libx264")


if __name__ == "__main__":

    input_video_path = "Videos/CatRun_Wartermark.mp4"
    output_video_path = "Videos/CatRun_Wartermark_ToRemove.mp4"

    watermark_mask = None

    video_clip = VideoFileClip(input_video_path)
    if watermark_mask is None:
        watermark_mask = generate_watermark_mask(video_clip)

    mask_func = lambda frame: cv2.inpaint(frame, watermark_mask, 3, cv2.INPAINT_NS)
    video_name = os.path.basename(input_video_path)
    process_video(video_clip, output_video_path, mask_func)
    print(f"Successfully processed {video_name}")
