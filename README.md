
# 人脸图像剪裁工具

## 简介
这是一个基于 Flask 框架开发的人脸图像剪裁工具，能够对输入的图像进行等比例缩放、人脸识别和剪裁，并将结果保存到指定路径。

## 功能
- 读取指定路径的图像文件。
- 对图像进行等比例缩放。
- 检测并剪裁人脸区域。
- 将剪裁后的图像保存到指定路径。

## 代码结构
- `getAllPath` 函数：遍历目录获取指定后缀的文件路径。
- `scale_image` 函数：对图像进行等比例缩放。
- `detect_and_crop_faces` 函数：进行人脸识别和剪裁操作。
- `save_cropped_image` 函数：保存剪裁后的图像。
- `process_image` 函数：处理图像的整个流程，包括读取、缩放、剪裁和保存。
- `/resize_image` 路由：接收 POST 请求，处理图像剪裁请求。

## 环境配置
- Python 版本：[具体版本]
- 依赖库：
    - Flask
    - OpenCV
    - ConfigParser

## 启动方法
1. 确保已经安装了所需的依赖库。
2. 准备 `config.ini` 文件，配置 `target_path` （目标路径）和 `port` （端口，默认为 5000）。
3. 运行代码，服务将在指定端口启动。

## API 使用
通过发送 POST 请求到 `/resize_image` 路由，在请求体中以表单形式提供以下参数：
- `image_path` ：要处理的图像文件路径。
- `save_path` ：剪裁后图像的保存路径。
- `target_width` （可选，默认值 480）：剪裁的目标宽度。
- `target_height` （可选，默认值 640）：剪裁的目标高度。

## 注意事项
- 确保提供的图像路径和保存路径存在且可访问。
- 目标宽度和高度应大于 0。

---

