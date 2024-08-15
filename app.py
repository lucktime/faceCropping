#-*-coding:utf8-*-
import os
import cv2
import time
import logging
from flask import Flask, request, jsonify
import configparser  # 用于读取配置文件

app = Flask(__name__)

# 初始化日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

def scale_image(img, target_width=720, target_height=960):
    """
    对图像进行等比例缩放
    :param img: 输入图像
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 缩放后的图像
    """
    original_height, original_width = img.shape[:2]
    if original_width / original_height > target_width / target_height:
        new_height = target_height
        new_width = int(original_width * target_height / original_height)
    else:
        new_width = target_width
        new_height = int(original_height * target_width / original_width)
    return cv2.resize(img, (new_width, new_height))

def detect_and_crop_faces(resized_img, width_crop, height_crop):
    """
    进行人脸识别和剪裁
    :param resized_img: 缩放后的图像
    :param width_crop: 剪裁宽度
    :param height_crop: 剪裁高度
    :return: 剪裁后的图像列表
    """
    face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(resized_img, 1.1, 5)
    cropped_images = []
    for (x, y, w, h) in faces:
        if w >= 64 and h >= 64:
            x_center = x + w // 2
            y_center = y + h // 2
            x_start = max(0, x_center - width_crop // 2)
            y_start = max(0, y_center - height_crop // 2)
            x_end = min(x_start + width_crop, resized_img.shape[1])
            y_end = min(y_start + height_crop, resized_img.shape[0])
            cropped_image = resized_img[y_start:y_end, x_start:x_end]
            cropped_images.append(cropped_image)
    return cropped_images

def save_cropped_image(cropped_image,save_path):
    """
    保存剪裁后的图像
    :param cropped_image: 剪裁后的图像
    :param save_path: 保存路径
    """
    cv2.imwrite(save_path, cropped_image)

def process_image(image_path, save_path,width_crop, height_crop):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to read image at {image_path}: File not found or invalid format")
            return {'status': 'failed','message': 'Image not found or invalid format'}
        resized_img = scale_image(img)
        cropped_images = detect_and_crop_faces(resized_img, width_crop, height_crop)
        if cropped_images:
            save_cropped_image(cropped_images[0],save_path)  # 只保存第一张剪裁后的图像替换原文件
        logging.info(f"success {save_path}")
        return {'status': 'success'}
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return {'status': 'failed','message': f'Error processing image: {str(e)}'}

@app.route('/resize_image', methods=['POST'])
def resize_image():
    image_path = request.form.get('image_path')
    save_path = request.form.get('save_path')
    target_width = int(request.form.get('target_width', 480))  # 提供默认值
    target_height = int(request.form.get('target_height', 640))  # 提供默认值
    logging.error(f"image_path: {image_path} save_path: {save_path} target_width: {target_width} target_height: {target_height}")
    if not os.path.exists(image_path):
        return jsonify({'status': 'failed','message': 'Image path does not exist'})
    if target_width <= 0 or target_height <= 0:
        return jsonify({'status': 'failed','message': 'Invalid target width or height'})
    result = process_image(image_path,save_path, target_width, target_height)
    return jsonify(result)

if __name__ == '__main__':
    # 读取配置文件获取目标路径
    config = configparser.ConfigParser()
    config.read('config.ini')
    target_path = config.get('Settings', 'target_path')
    port = int(config.get('Settings', 'port', fallback=5000))  # 从配置中获取端口，默认为 5000
    app.run(debug=True, port=port)