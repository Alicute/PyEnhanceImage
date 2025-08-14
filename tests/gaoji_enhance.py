import pydicom
import numpy as np
import cv2

def enhance_dicom_balanced(input_path, output_path):
    ds = pydicom.dcmread(input_path)
    img = ds.pixel_array.astype(np.float32)

    # 归一化到 0~1
    img_norm = (img - img.min()) / (img.max() - img.min())

    # ---------- 1. 多尺度高频增强 ----------
    blur_small = cv2.GaussianBlur(img_norm, (0, 0), 1)
    high_freq_small = img_norm - blur_small
    blur_large = cv2.GaussianBlur(img_norm, (0, 0), 5)
    high_freq_large = img_norm - blur_large

    img_detail = img_norm + 1.5 * high_freq_small + 0.8 * high_freq_large
    img_detail = np.clip(img_detail, 0, 1)

    # ---------- 2. 光照归一化（弱化效果） ----------
    illum = cv2.GaussianBlur(img_detail, (0, 0), 50)
    img_light_norm = img_detail / (illum + 1e-6)

    # 限制归一化强度，防止中心过亮
    img_light_norm = 0.5 * img_light_norm + 0.5 * img_detail
    img_light_norm = np.clip(img_light_norm / img_light_norm.max(), 0, 1)

    # ---------- 3. Gamma 曲线调整（压亮提暗） ----------
    gamma = 0.1  # <1 提亮暗部, >1 压亮高光
    img_gamma = np.power(img_light_norm, gamma)

    # ---------- 4. CLAHE 增强 ----------
    img_16bit = (img_gamma * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_16bit)

    # ---------- 5. 混合原图（保留部分原始光照质感） ----------
    alpha = 0.8  # 增强结果权重
    img_final = cv2.addWeighted(img_clahe.astype(np.float32), alpha,
                                img_16bit.astype(np.float32), 1 - alpha, 0)
    img_final = np.clip(img_final, 0, 65535).astype(np.uint16)

    # ---------- 6. 保存 ----------
    ds.PixelData = img_final.tobytes()
    ds.save_as(output_path)
    print(f"增强完成：{output_path}")

# 示例
enhance_dicom_balanced(r"D:\Projects\PyEnhanceImage\a.dcm", r"D:\Projects\PyEnhanceImage\钢板-高级增强.dcm")
