import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


CLASSES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter']

PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
            (255, 255, 0), (147, 116, 116), (0, 0, 255), (0, 0, 0)]


def draw(img, cx, cy, w, h, a, color=(0, 0, 255)):
    x = cx  # 矩形框的中心点x
    y = cy  # 矩形框的中心点y
    angle = a  # 矩形框的倾斜角度（长边相对于水平）
    width, height = w, h  # 矩形框的宽和高
    # if result[1][0] > result[1][1]:
    #     height, width = int(result[1][0]), int(result[1][1])
    # else:
    #     height, width = int(result[1][1]), int(result[1][0])
    anglePi = angle * math.pi / 180.0  # 计算角度
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    # 根据得到的点，画出矩形框
    cv2.line(img, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 0, 255), 1, 4)
    cv2.line(img, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (0, 0, 255), 1, 4)
    cv2.line(img, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 0, 255), 1, 4)
    cv2.line(img, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (0, 0, 255), 1, 4)


def read_dota_annotation(txt_path):
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[:8]))
            class_name = parts[8]
            boxes.append((coords, class_name))
    return boxes


def draw_dota_boxes(img_path, ann_path, save_path=None):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    boxes = read_dota_annotation(ann_path)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=30)

    for coords, cls in boxes:
        color = PALETTE[CLASSES.index(cls)] if cls in CLASSES else (255, 0, 0)
        polygon = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        draw.line(polygon + [polygon[0]], fill=color, width=5)

        text_pos = polygon[0]
        text = cls
        text_bbox = draw.textbbox(text_pos, text, font=font)
        # 扩大一下背景框大小
        bg_margin = 2
        bg_box = [text_bbox[0] - bg_margin, text_bbox[1] - bg_margin,
                  text_bbox[2] + bg_margin, text_bbox[3] + bg_margin]
        draw.rectangle(bg_box, fill=(0, 0, 0))
        draw.text(text_pos, text, fill=color, font=font)
    
    if save_path:
        img.save(save_path)
        print(f"Image saved to {save_path}")
    else:
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    img_path = '../data/CODrone/train/images/chenhuachengpark_day_30m_30c_frame_750.jpg'
    ann_path = '../data/CODrone/train/annfile/chenhuachengpark_day_30m_30c_frame_750.txt'
    save_path = './image.png'
    draw_dota_boxes(img_path, ann_path, save_path)

    # img_dir = '../data/dota/val_part/images/'
    # ann_dir = '../data/dota/val_part/annfiles/'
    # save_dir = '../work_dirs/gt/'
    # os.makedirs(save_dir, exist_ok=True)

    # for img_file in sorted(os.listdir(img_dir)):
    #     if not img_file.endswith('.png'):
    #         continue
    #     img_path = os.path.join(img_dir, img_file)
    #     ann_file = img_file.replace('.png', '.txt')
    #     ann_path = os.path.join(ann_dir, ann_file)
    #     save_path = os.path.join(save_dir, img_file)
    #     draw_dota_boxes(img_path, ann_path, save_path)
