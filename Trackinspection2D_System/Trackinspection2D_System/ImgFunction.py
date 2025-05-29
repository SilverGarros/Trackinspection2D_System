import os
import sys
from threading import Lock
import struct
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from RailHead.PreprogressingMagic import Function as F

#
CROP_WIDE = 1024

"""
针对图像数据的扩展函数
"""

"""
图像信息
"""


# 获取 BMP 图像的位深度
def get_bmp_bit_depth_by_file(image_path):
    with open(image_path, 'rb') as f:
        # 读取 BMP 文件头（14字节）
        file_header = f.read(14)

        # 读取 BMP 信息头（通常是40字节，但也有其他类型的头）
        info_header = f.read(40)

        # 从信息头中提取 biBitCount 字段（位深度），biBitCount 位于第 15 和 16 字节
        bit_depth = struct.unpack('<H', info_header[14:16])[0]  # 'H'表示无符号短整型（2字节）

        return bit_depth


#  获取图像的色深
def get_img_mode(image_path):
    with Image.open(image_path) as img:
        # 获取图像的色深
        return img.mode


# 获取图像的大小和类型
def check_image_size_and_type(img):
    try:
        # 获取图像的大小和通道数
        shape = img.shape

        # 判断图像类型
        if len(shape) == 2:
            # 单通道灰度图 (512, 512)
            # print(f"图像大小: {shape}, 类型: 灰度图")
            return 1
        elif len(shape) == 3 and shape[2] == 3:
            # 三通道彩色图 (512, 512, 3)
            # print(f"图像大小: {shape}, 类型: 彩色图")
            return 2
        else:
            # print(f"图像大小: {shape}, 类型: 其他类型")
            return 0

    except Exception as e:
        print(f"发生错误: {e}")


"""
图像处理
"""


# 单图纵向拉伸函数

def image_vertical_stretch(image_path='', stretch_ratio=2, save_or_not=True, stretch_path=None):
    """
    拉伸图像并纵向切分为多张图。如果拉伸倍数为整数，每张图高度与原图一致，
    如果为非整数倍，则最后一张图高度为剩余部分。

    :param image_path: 输入图像路径
    :param stretch_ratio: 纵向拉伸倍数（可为非整数）
    :param save_or_not: 是否保存切分后图像
    :param stretch_path: 保存目录，若未指定，则使用原图所在目录
    :return: 若保存，则返回所有切分图的保存路径；否则返回切分后的图像列表
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 读取图像（若bmp判断位深度）
    if image_path.lower().endswith('.bmp'):
        if get_bmp_bit_depth_by_file(image_path) == 8:
            img = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {os.path.abspath(image_path)}")

    orig_h, orig_w = img.shape[:2]
    new_h = int(orig_h * stretch_ratio)
    stretched_img = cv2.resize(img, (orig_w, new_h), interpolation=cv2.INTER_LINEAR)

    images = []
    # 计算完整的切分张数，就是整数部分
    num_full = int(stretch_ratio)
    for i in range(num_full):
        piece = stretched_img[i * orig_h: (i + 1) * orig_h, :]
        images.append(piece)
    # 如果伸展后剩余部分高度不够整张原图，则作为最后一张图
    if num_full * orig_h < new_h:
        remainder = stretched_img[num_full * orig_h: new_h, :]
        images.append(remainder)

    if save_or_not:
        target_dir = stretch_path if stretch_path else os.path.dirname(image_path)
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        saved_paths = []
        for idx, img_piece in enumerate(images, start=1):
            # 保存文件名中添加序号
            new_name = f"{name}_part{idx}{ext}"
            new_path = os.path.join(target_dir, new_name)
            F.imwrite_unicode(new_path, img_piece)
            saved_paths.append(new_path)
        return saved_paths
    else:
        return images


# 批量重新切分文件夹内的图像（一次命名）
def folder_image_vertical_stretch_and_split(folder_path, stretch_ratio=2, save_or_not=True, output_folder="./output"):
    """
    对文件夹内的所有图片进行纵向拉伸，横向不变，拉伸后拼接并切分。

    :param folder_path: 输入文件夹路径
    :param stretch_ratio: 纵向拉伸倍数（可为非整数）
    :param save_or_not: 是否保存切分后的图像
    :param output_folder: 保存目录，若未指定，则使用输入文件夹路径
    :return: 若保存，则返回所有切分图的保存路径；否则返回切分后的图像列表
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # 获取文件夹内所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        raise ValueError("No image files found in the folder.")

    stretched_images = []
    orig_width = None

    # 对每张图片进行纵向拉伸
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue

        orig_h, orig_w = img.shape[:2]
        if orig_width is None:
            orig_width = orig_w  # 记录原始横向尺寸
        elif orig_width != orig_w:
            raise ValueError("All images must have the same width for horizontal concatenation.")

        new_h = int(orig_h * stretch_ratio)
        stretched_img = cv2.resize(img, (orig_w, new_h), interpolation=cv2.INTER_LINEAR)
        stretched_images.append(stretched_img)

    # 将所有拉伸后的图片按纵向拼接
    concatenated_image = np.vstack(stretched_images)

    # 按照原始横向尺寸切分拼接后的大图
    total_height = concatenated_image.shape[0]
    split_images = []
    num_splits = total_height // orig_h

    for i in range(num_splits):
        split_img = concatenated_image[i * orig_h: (i + 1) * orig_h, :]
        split_images.append(split_img)

    # 如果有剩余部分，作为最后一张图
    if num_splits * orig_h < total_height:
        remainder = concatenated_image[num_splits * orig_h:, :]
        split_images.append(remainder)

    if save_or_not:
        target_dir = output_folder if output_folder else folder_path
        os.makedirs(target_dir, exist_ok=True)
        saved_paths = []
        for idx, img_piece in enumerate(split_images, start=1):
            new_name = f"stitched_part{idx}.png"
            new_path = os.path.join(target_dir, new_name)
            F.imwrite_unicode(new_path, img_piece)
            saved_paths.append(new_path)
        return saved_paths
    else:
        return split_images

# 批量重新切分文件夹内的图像（不切图）
def folder_image_vertical_stretch_and_split_without_stitch(folder_path, stretch_ratio=2, save_or_not=True, output_folder="./output"):
    """
    对文件夹内的每个图像进行纵向拉伸，并按照原图高度切分。
    切分后的图像命名规则：原图名_切分总数_当前张数.原图格式
    若拉伸后余高存在，则该余高单独作为一个切分块，所有切分块编号从0开始
    """
    import os
    import cv2
    import numpy as np

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        raise ValueError("No image files found in the folder.")

    results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue

        orig_h, orig_w = img.shape[:2]
        new_h = int(orig_h * stretch_ratio)
        stretched_img = cv2.resize(img, (orig_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算完整切分块数量及余高
        num_full = new_h // orig_h
        remainder_height = new_h % orig_h

        pieces = []
        # 提取完整切分块
        for i in range(num_full):
            piece = stretched_img[i * orig_h: (i + 1) * orig_h, :]
            pieces.append(piece)
        # 如果余高存在，作为单独的一块
        if remainder_height > 0:
            remainder = stretched_img[num_full * orig_h:, :]
            pieces.append(remainder)

        total_splits = len(pieces)
        indices = list(range(total_splits))  # 编号从 0 开始

        saved_info = []
        for idx, piece in zip(indices, pieces):
            base, ext = os.path.splitext(image_file)
            new_name = f"{base}_{total_splits}of{idx}{ext}"
            if save_or_not:
                target_path = os.path.join(output_folder, new_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # 使用项目中的 F.imwrite_unicode 接口保存图片
                F.imwrite_unicode(target_path, piece)
                saved_info.append(target_path)
            else:
                saved_info.append(piece)
        results.append({image_file: saved_info})
    return results#

#批量重新切分文件夹内的图像（拼接余量）
# 该函数在处理时自动拼接上一张的余量
# 拼接余量后，命名规则：原图名_切分总数_当前张数.原图格式
# 若拉伸后余高存在，则该余高单独作为一个切分块，所有切分块编号从0开始
def folder_image_vertical_stretch_and_split_with_stitch(folder_path, stretch_ratio=2, save_or_not=True, output_folder="./output"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        raise ValueError("No image files found in the folder.")

    results = []
    prev_remainder = None
    remainder = None
    total_files = len(image_files)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        # print("image",image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image at {image_path}, skipping.")
            continue
        orig_h, orig_w = img.shape[:2]
        new_h = int(orig_h * stretch_ratio)
        stretched_img = cv2.resize(img, (orig_w, new_h), interpolation=cv2.INTER_LINEAR)
        num_full = new_h // orig_h
        print("num_full_1",num_full)
        pieces = [stretched_img[i * orig_w:(i + 1) * orig_w, :] for i in range(num_full)]
        if prev_remainder is not None:
            # 在切分图前面拼接上一张的余量，如果有的话
            stretched_img = np.vstack([prev_remainder, stretched_img])
            new_h = stretched_img.shape[0]
            print("new_h",new_h)
            num_full = new_h // orig_h
            pieces = [stretched_img[i * orig_w:(i + 1) * orig_w, :] for i in range(num_full)]
            concat_flag = True  # 表示当前图有拼接余量


        else:
            concat_flag = False
        print("num_full2", num_full)

        # 如果余高存在，作为单独的一块
        remainder = stretched_img[num_full * orig_w:] if new_h % orig_w else None
        # 当前图如果存在余量，且不是最后一张，则更新prev_remainder暂存预备给下一张拼接；最后一张则单独保存
        if remainder is not None:
            if idx == total_files - 1:
                # 如果是最后一张，则单独另外储存余量
                pieces.append(remainder)
                prev_remainder = None

            else:
                prev_remainder = remainder
        else: prev_remainder = None

        total_splits = len(pieces)
        if concat_flag:
            indices = list(range(0, total_splits))
        else:
            indices = list(range(1, total_splits + 1))

        saved_info = []
        for idx, piece in zip(indices, pieces):
            base, ext = os.path.splitext(image_file)
            new_name = f"{base}_{total_splits}of{idx}{ext}"
            if save_or_not:
                target_path = os.path.join(output_folder, new_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # 使用项目中的 F.imwrite_unicode 接口保存图片
                # 检查piece的尺寸，如果高度大于orig_w，打印报错信息
                if piece.shape[0] > orig_w:
                    print(f"Warning: Image piece {new_name} has height {piece.shape[0]} greater than orig_w{orig_w}.")
                F.imwrite_unicode(target_path, piece)
                saved_info.append(target_path)
            else:
                saved_info.append(piece)
        results.append({image_file: saved_info})

    return results


def folder_image_vertical_stretch_and_split_with_stitch_tqdm(folder_path, stretch_ratio=2, save_or_not=True, output_folder="./output",max_workers=32):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        raise ValueError("No image files found in the folder.")

    results = []
    failed = []
    total = len(image_files)

    def _process(file_name):
        image_path = os.path.join(folder_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"Could not read image: {image_path}")

        orig_h, orig_w = img.shape[:2]
        new_h = int(orig_h * stretch_ratio)
        stretched = cv2.resize(img, (orig_w, new_h), interpolation=cv2.INTER_LINEAR)

        pieces = [stretched[i*orig_h:(i+1)*orig_h, :] for i in range(new_h // orig_h)]
        rem = stretched[(new_h // orig_h)*orig_h:] if new_h % orig_h else None
        if rem is not None:
            pieces.append(rem)

        saved_info = []
        base, ext = os.path.splitext(file_name)
        count = len(pieces)
        for idx, piece in enumerate(pieces, start=1):
            if save_or_not:
                name = f"{base}_{count}of{idx}{ext}"
                path = os.path.join(output_folder, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                F.imwrite_unicode(path, piece)
                saved_info.append(path)
            else:
                saved_info.append(piece)
        return saved_info

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_process, fn): fn for fn in image_files}
        for future in tqdm(as_completed(futures), total=total, desc="处理进度", unit="张"):
            fn = futures[future]
            try:
                info = future.result()
                results.append({fn: info})
            except Exception as e:
                safe_print(f"❌ 处理失败: {fn}, 错误信息: {e}")
                failed.append(fn)

    return results
# 图像镜像处理函数
def image_mirror(image_path='', flipcode=0, save_or_not=True, mirror_path=None):
    """
    镜像处理图像函数
    author = silver
    version = 1.0
    :param image_path: 要镜像处理的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param flipcode: cv2.flip(img, flipcode) 镜像模式：
                     - 0: 水平镜像
                     - 1: 上下镜像
                     - -1: 水平垂直翻转
    :type flipcode: int
    :param mirror_path: 处理后的图像保存地址，默认为 None，表示保存到原始目录。
    :type mirror_path: str, optional
    :param save_or_not: 是否保存图片：
                        - True: 保存并返回保存信息
                        - False: 返回镜像后的 OpenCV 数组
    :type save_or_not: bool, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回处理后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 判断图像是否为灰度图像
    # if image_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # 假设图像是常见格式
    if image_path.lower().endswith('.bmp'):
        if get_bmp_bit_depth_by_file(image_path) == 8:
            img = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_GRAYSCALE)  # 如果是灰度图像或需要灰度处理
        else:
            img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)  # 如果是彩色图像

    if img is None:
        raise FileNotFoundError(f"Could not read image: {os.path.abspath(image_path)}")
    # 水平镜像
    mirrored_img = cv2.flip(img, flipcode)

    # 保存控制器
    if save_or_not is True:
        if mirror_path is None:
            mirror_path = f"{os.path.splitext(image_path)[0]}_mirrored_{flipcode}_{os.path.splitext(image_path)[1]}"
        else:
            mirror_path = f"{mirror_path}/{os.path.splitext(os.path.basename(image_path))[0]}_mirrored_{flipcode}_{os.path.splitext(image_path)[1]}"

        os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
        # 保存镜像后的图像
        cv2.imwrite(mirror_path, mirrored_img)
        # print(f"函数为保存模式，镜像后的图像已经保存在：{mirror_path}")
        return f"函数为保存模式，镜像后的图像已经保存在：{mirror_path}"
    if save_or_not is False:
        return mirrored_img


# 图像旋转处理函数
def image_rotate(image_path='', angle=180, save_or_not=True, rotate_path=None):
    """
    旋转图像函数

    :param image_path: 要旋转的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param angle: 旋转角度（逆时针），默认为 180°。
    :type angle: int
    :param save_or_not: 保存模式：
                        - True: 保存图片并返回保存信息。
                        - False: 返回旋转后的 OpenCV 数组。
    :type save_or_not: bool
    :param rotate_path: 旋转后图像的保存地址，默认为 None，表示保存到原始目录。
    :type rotate_path: str, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回旋转后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # 判断图像是否为灰度图像
    # if image_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # 假设图像是常见格式
    if image_path.lower().endswith('.bmp'):
        if get_bmp_bit_depth_by_file(image_path) == 8:
            img = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_GRAYSCALE)  # 如果是灰度图像或需要灰度处理
        else:
            img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)  # 如果是彩色图像
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 获取图像尺寸
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 进行仿射变换
    rotated_img = cv2.warpAffine(img, M, (w, h))

    # 保存控制器
    if save_or_not:
        if rotate_path is None:
            rotate_path = f"{os.path.splitext(image_path)[0]}_rotated_{angle}_{os.path.splitext(image_path)[1]}"
        else:
            rotate_path = f"{rotate_path}/{os.path.splitext(os.path.basename(image_path))[0]}_rotated_{angle}{os.path.splitext(image_path)[1]}"
        # print(rotate_path)
        os.makedirs(os.path.dirname(rotate_path), exist_ok=True)
        # print(os.path.dirname(rotate_path))
        # 保存镜像后的图像
        cv2.imwrite(rotate_path, rotated_img)
        # print(f"函数为保存模式，翻转后的图像已经保存在：{rotate_path}")
        return f"函数为保存模式，翻转后的图像已经保存在：{rotate_path}"
    else:
        return rotated_img


# 图像高斯模糊函数
def image_blur_gaussian(image_path='', ksize=(5, 5), save_or_not=True, blur_path=None):
    """
    高斯模糊图像函数

    :param image_path: 要进行高斯模糊的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param ksize: 高斯核的大小，默认为 (5, 5)。必须是正奇数。
    :type ksize: tuple
    :param save_or_not: 保存模式：
                        - True: 保存图片并返回保存信息。
                        - False: 返回模糊后的 OpenCV 数组。
    :type save_or_not: bool
    :param blur_path: 模糊后图像的保存地址，默认为 None，表示保存到原始目录。
    :type blur_path: str, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回模糊后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # 判断图像是否为灰度图像
    # if image_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # 假设图像是常见格式
    if image_path.lower().endswith('.bmp'):
        if get_bmp_bit_depth_by_file(image_path) == 8:
            img = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_GRAYSCALE)  # 如果是灰度图像或需要灰度处理
        else:
            img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)  # 如果是彩色图像
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 进行高斯模糊
    blurred_img = cv2.GaussianBlur(img, ksize, 0)

    # 保存控制器
    if save_or_not:
        if blur_path is None:
            blur_path = f"{os.path.splitext(image_path)[0]}_blurred_{ksize[0]}_{ksize[1]}{os.path.splitext(image_path)[1]}"
        else:
            blur_path = f"{blur_path}/{os.path.splitext(os.path.basename(image_path))[0]}_blurred_{ksize[0]}_{ksize[1]}{os.path.splitext(image_path)[1]}"

        os.makedirs(os.path.dirname(blur_path), exist_ok=True)

        # 保存模糊后的图像
        cv2.imwrite(blur_path, blurred_img)
        # print(f"函数为保存模式，模糊后的图像已经保存在：{blur_path}")
        return f"函数为保存模式，模糊后的图像已经保存在：{blur_path}"
    else:
        return blurred_img


# 图像高斯噪声函数
def image_noise_gaussian(image_path='', mean=-1, sigma=5, save_or_not=True, noisy_path=None):
    """
    添加高斯噪声函数

    :param image_path: 要添加高斯噪声的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param mean: 高斯噪声的均值，默认为 -1。
    :type mean: int
    :param sigma: 高斯噪声的标准差，默认为 5。推荐不要超过10
    :type sigma: int
    :param save_or_not: 保存模式：
                        - True: 保存噪声处理后的图片并返回保存信息。
                        - False: 返回添加噪声后的 OpenCV 数组。
    :type save_or_not: bool
    :param noisy_path: 添加噪声后图像的保存地址，默认为 None，表示保存到原始目录。
    :type noisy_path: str, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回添加噪声后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    # 判断图像是否为灰度图像
    # if image_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):  # 假设图像是常见格式
    if image_path.lower().endswith('.bmp'):
        if get_bmp_bit_depth_by_file(image_path) == 8:
            img = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_GRAYSCALE)  # 如果是灰度图像或需要灰度处理
        else:
            img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path)  # 如果是彩色图像
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    # 生成高斯噪声
    if len(img.shape) == 2:
        noise_shape = img.shape
    # print(img.shape)
    # row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, img.shape)  # 生成高斯噪声
    noisy_img = np.array(img + gauss, dtype=np.uint8)  # 添加噪声并转化为无符号整型

    # 保存控制器
    if save_or_not:
        if noisy_path is None:
            noisy_path = f"{os.path.splitext(image_path)[0]}_noisy_{mean}_{sigma}{os.path.splitext(image_path)[1]}"
        else:
            noisy_path = f"{noisy_path}/{os.path.splitext(os.path.basename(image_path))[0]}_gaussian-noisy_{mean}_{sigma}{os.path.splitext(image_path)[1]}"

        os.makedirs(os.path.dirname(noisy_path), exist_ok=True)

        # 保存添加噪声后的图像
        cv2.imwrite(noisy_path, noisy_img)
        print(f"函数为保存模式，添加噪声后的图像已经保存在：{noisy_path}")
        return f"函数为保存模式，添加噪声后的图像已经保存在：{noisy_path}"
    else:
        return noisy_img


# 图像椒盐噪声函数
def image_noise_salt_pepper(image_path='', salt_prob=0.01, pepper_prob=0.01, save_or_not=True, noisy_path=None):
    """
    添加椒盐噪声函数（Salt and Pepper Noise）

    :param image_path: 要添加椒盐噪声的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param salt_prob: 椒盐噪声中 "盐"（白色点）的比例，默认为 0.01（即 1% 像素变为 255）。
    :type salt_prob: float
    :param pepper_prob: 椒盐噪声中 "椒"（黑色点）的比例，默认为 0.01（即 1% 像素变为 0）。
    :type pepper_prob: float
    :param save_or_not: 保存模式：
                        - True: 保存噪声处理后的图片并返回保存信息。
                        - False: 返回添加噪声后的 OpenCV 数组。
    :type save_or_not: bool
    :param noisy_path: 添加噪声后图像的保存地址，默认为 None，表示保存到原始目录。
    :type noisy_path: str, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回添加噪声后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 读取图像（如果是 BMP 且是 8 位，则读取为灰度图）
    if image_path.lower().endswith('.bmp'):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 复制原始图像
    noisy_img = np.copy(img)

    # 获取图像的大小
    height, width = noisy_img.shape[:2]

    # 添加 "盐" 噪声（白色点，255）
    num_salt = int(height * width * salt_prob)  # 计算需要变成 255 的像素个数
    salt_coords = [np.random.randint(0, i, num_salt) for i in (height, width)]
    noisy_img[salt_coords[0], salt_coords[1]] = 255  # 将这些像素设为 255（白色）

    # 添加 "椒" 噪声（黑色点，0）
    num_pepper = int(height * width * pepper_prob)  # 计算需要变成 0 的像素个数
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in (height, width)]
    noisy_img[pepper_coords[0], pepper_coords[1]] = 0  # 将这些像素设为 0（黑色）

    # 保存控制器
    if save_or_not:
        if noisy_path is None:
            noisy_path = f"{os.path.splitext(image_path)[0]}_saltpepper_{salt_prob}_{pepper_prob}{os.path.splitext(image_path)[1]}"
        else:
            noisy_path = f"{noisy_path}/{os.path.splitext(os.path.basename(image_path))[0]}_saltpepper_{salt_prob}_{pepper_prob}{os.path.splitext(image_path)[1]}"

        os.makedirs(os.path.dirname(noisy_path), exist_ok=True)

        # 保存添加噪声后的图像
        cv2.imwrite(noisy_path, noisy_img)
        print(f"函数为保存模式，添加噪声后的图像已经保存在：{noisy_path}")
        return f"函数为保存模式，添加噪声后的图像已经保存在：{noisy_path}"
    else:
        return noisy_img


# 图像锐化函数（拉普拉斯算子）
def image_sharpen_laplacian(image_path='', save_or_not=True, sharpen_path=None):
    """
    实现拉普拉斯锐化方法。

    :param image_path: 要进行高斯锐化的图像的地址，可以是绝对地址，也可以是相对地址。
    :type image_path: str
    :param save_or_not: 保存模式：
                    - True: 保存噪声处理后的图片并返回保存信息。
                    - False: 返回添加噪声后的 OpenCV 数组。
    :type save_or_not: bool
    :param sharpen_path: 添加噪声后图像的保存地址，默认为 None，表示保存到原始目录。
    :type sharpen_path: str, optional
    :return:
        - str: 如果 `save_or_not` 为 True，返回保存信息。
        - numpy.ndarray: 如果 `save_or_not` 为 False，返回锐化后的 OpenCV 数组。
    :rtype: str | numpy.ndarray
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 读取图像（如果是 BMP 且是 8 位，则读取为灰度图）
    if image_path.lower().endswith('.bmp'):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 使用拉普拉斯算子检测边缘
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # 将拉普拉斯结果加回到原始图像
    sharpened_img = img - 0.5 * laplacian
    # 将像素值裁剪到有效范围
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    # 保存控制器
    if save_or_not:
        if sharpen_path is None:
            sharpen_path = f"{os.path.splitext(image_path)[0]}_laplacian{os.path.splitext(image_path)[1]}"
        else:
            sharpen_path = os.path.join(
                sharpen_path,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_laplacian{os.path.splitext(image_path)[1]}"
            )

        os.makedirs(os.path.dirname(sharpen_path), exist_ok=True)
        cv2.imwrite(sharpen_path, sharpened_img)
        print(f"函数为保存模式，锐化后的图像已经保存在：{sharpen_path}")
        return f"函数为保存模式，锐化后的图像已经保存在：{sharpen_path}"
    else:
        return sharpened_img



# 轨面自动裁剪函数
def railhead_crop_highlight_center_area(image_path, threshold=5, kernel_size=5, crop_wide=CROP_WIDE, save_or_not=True,
                                        output_path=None):
    """
    自动裁剪图像中最显著高亮区域，并居中裁剪固定宽度区域（横向）

    :param image_path: 输入图像路径
    :param threshold: 灰度阈值，用于提取高亮区域
    :param kernel_size: 闭运算结构核大小（用于平滑高亮块）
    :param crop_wide: 裁剪宽度
    :param save_or_not: 保存模式：
                        - True: 保存剪切处理后的图片并返回保存信息。
                        - False: 返回剪切处理后的 OpenCV 数组。
    :param output_path: 裁剪后保存路径（可选，仅在 save_or_not=True 时有效）
    :return: 裁剪后的图像（OpenCV 图像数组）
    """
    # 读取图像
    img = F.imread_unicode(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 转灰度（如果不是）
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # 二值化高亮区域
    _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

    # 闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找最大轮廓区域
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到高亮区域")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 计算中心点横坐标
    crod_m = int(x + w / 2)

    # 计算裁切区域（横向居中 CROP_WIDE 宽度）
    x1 = max(0, crod_m - int(crop_wide / 2))
    x2 = min(img.shape[1], crod_m + int(crop_wide / 2))
    y1 = 0
    y2 = img.shape[0]

    # 安全校验
    if x2 <= x1 or y2 <= y1:
        raise ValueError("裁切区域非法，无法裁剪")

    cropped = img[y1:y2, x1:x2]

    # 保存结果
    if save_or_not:
        if output_path is None:
            output_path = f"{os.path.splitext(image_path)[0]}{os.path.splitext(image_path)[1]}"
        else:
            output_path = os.path.join(output_path,
                                       f"{os.path.splitext(os.path.basename(image_path))[0]}{os.path.splitext(image_path)[1]}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        F.imwrite_unicode(output_path, cropped)
        return f"函数为保存模式，裁剪后的轨面图像已经保存在：{output_path}"

    return cropped


# 批量处理文件夹中的所有图像
def process_images_in_folder(folder_path, process_function, *args, **kwargs):
    """
    批量处理文件夹中的所有图像，遇到错误自动跳过，并记录失败的图像

    :param folder_path: 要处理的文件夹路径
    :type folder_path: str
    :param process_function: 处理图像的函数（如 image_noise_salt_pepper）
    :type process_function: function
    :param args: 传递给 process_function 的位置参数
    :param kwargs: 传递给 process_function 的关键字参数
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    # 支持的图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 获取所有文件
    files = os.listdir(folder_path)

    # 存储失败的图像
    failed_images = []

    # 遍历文件夹中的所有图片
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 跳过非图像文件
        if not file_name.lower().endswith(valid_extensions):
            continue

        try:
            print(f"正在处理: {file_path}")

            # 调用指定的处理函数（例如 image_noise_salt_pepper）
            result = process_function(file_path, *args, **kwargs)

            if isinstance(result, str):  # 说明是保存模式，打印成功信息
                print(f"✅ 处理完成: {result}")
            else:
                print(f"✅ 处理完成: {file_name} (未保存模式)")

        except Exception as e:
            print(f"❌ 处理失败: {file_name}, 错误信息: {e}")
            failed_images.append(file_name)  # 记录失败的图像

    print("\n🎉 所有图片处理完成！")

    # 如果有失败的图像，打印它们
    if failed_images:
        print("\n⚠️ 以下图片处理失败：")
        for failed_image in failed_images:
            print(f" - {failed_image}")
    else:
        print("\n✅ 所有图片都成功处理！")


# 单张图像处理包装器
def process_single_image(file_path, process_function, *args, **kwargs):
    """
    单张图像处理包装器，用于线程池调用
    """
    try:
        result = process_function(file_path, *args, **kwargs)
        return (file_path, True, result)
    except Exception as e:
        return (file_path, False, str(e))


# 安全打印函数，避免多线程输出混乱
def safe_print(*args, **kwargs):
    # 设置输出流为行缓冲模式
    sys.stdout.reconfigure(line_buffering=True)
    # 定义全局锁
    print_lock = Lock()
    with print_lock:
        tqdm.write(*args, **kwargs)


# 轨面2D图像纵向拉伸和切分函数
def Rail2D_image_stretch_and_split(image_path, stretch_factor=2, save_or_not=True, output_path=None):
    """
    将轨面2D图像纵向拉伸到指定倍率并切分重命名为“原图名称_n-stretch_factor.原图格式”

    :param image_path: 输入图像路径
    :param stretch_factor: 拉伸倍率
    :param save_or_not: 是否保存处理后的图像
    :param output_path: 输出路径（可选）
    :return: 处理后的图像列表或保存信息
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = F.imread_unicode(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 获取原始尺寸
    original_height, original_width = img.shape[:2]

    # 计算新高度
    new_height = int(original_height * stretch_factor)

    # 拉伸图像
    stretched_img = cv2.resize(img, (original_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 切分图像
    split_images = []
    for i in range(stretched_img.shape[0] // CROP_WIDE):
        split_img = stretched_img[i * CROP_WIDE:(i + 1) * CROP_WIDE, :]
        split_images.append(split_img)

        if save_or_not:
            if output_path is None:
                output_path = f"{os.path.splitext(image_path)[0]}_{i}_stretch_{stretch_factor}{os.path.splitext(image_path)[1]}"
            else:
                output_path = os.path.join(output_path,
                                           f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_stretch_{stretch_factor}{os.path.splitext(image_path)[1]}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            F.imwrite_unicode(output_path, split_img)

            print(f"函数为保存模式，裁剪后的轨面图像已经保存在：{output_path}")

    return split_images if not save_or_not else f"函数为保存模式，裁剪后的轨面图像已经保存在：{output_path}"


# 使用线程池并发处理图像
def process_images_in_folder_Thread(folder_path, process_function, max_workers=8, *args, **kwargs):
    """
    并发处理图像，使用线程池加速，并输出处理进度和成功/失败统计

    :param folder_path: 图像文件夹路径
    :param process_function: 图像处理函数
    :param max_workers: 最大线程数（默认 8）
    :param args: 处理函数的位置参数
    :param kwargs: 处理函数的关键字参数
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("⚠️ 未找到图像文件")
        return

    success_count = 0
    failed_images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for file_name in image_files:
            file_path = os.path.join(folder_path, file_name)
            future = executor.submit(process_single_image, file_path, process_function, *args, **kwargs)
            futures[future] = file_name  # 绑定 future 和文件名

        # 初始化 tqdm
        pbar = tqdm(total=len(futures),
                    desc=f"图像<{process_function.__name__}> 正在处理...",
                    unit="img",
                    dynamic_ncols=True)

        for future in as_completed(futures):
            file_name = futures[future]
            file_path, success, info = future.result()

            # ✅ 实时更新 tqdm 描述为当前正在处理的文件名
            pbar.set_description(f"图像<{process_function.__name__}> 正在处理: {file_name}")

            if success:
                if isinstance(info, str):
                    pbar.set_description(f"图像正在处理: {file_name} ✅ 处理完成: {info}")
                else:
                    pbar.set_description(
                        f"图像<{process_function.__name__}> 正在处理: {file_name} ✅ 处理完成: {file_name} (未保存模式)")
                success_count += 1
            else:
                safe_print(f"❌ 处理失败: {file_name}, 错误信息: {info}")
                failed_images.append(file_name)

            pbar.update(1)  # 手动更新进度

        pbar.close()

    total_images = len(image_files)
    failed_count = len(failed_images)

    print("\n🎉 所有图片处理完成！")
    print(f"\n📊 总计: {total_images} 张")
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {failed_count} 张")

    if failed_images:
        print("\n⚠️ 以下图片处理失败：")
        for failed_image in failed_images:
            print(f" - {failed_image}")
# process_images_in_folder("output_test", image_mirror, flipcode=-1, save_or_not=True)
# process_images_in_folder("output_test", image_rotate, angle=180, save_or_not=True)
# process_images_in_folder("output_test", image_blur_gaussian, ksize=(5, 5), save_or_not=True)
# process_images_in_folder("Test_IMG", image_noise_gaussian, mean=-1, sigma=5, save_or_not=True)
# process_images_in_folder("output_test", railhead_crop_highlight_center_area, save_or_not=True)
