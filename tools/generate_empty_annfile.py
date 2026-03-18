import os
from tqdm import tqdm

def create_empty_annotations(ann_dir, empty_dir):
    """
    为每个标注文件创建同名的空文件
    
    参数:
        ann_dir: 原始标注目录路径
        empty_dir: 空标注文件输出目录
    """
    # 创建输出目录
    os.makedirs(empty_dir, exist_ok=True)
    
    # 获取所有标注文件
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.txt')]
    
    print(f"正在为{len(ann_files)}个标注文件创建空版本...")
    for ann_file in tqdm(ann_files):
        empty_path = os.path.join(empty_dir, ann_file)
        
        # 创建空文件（如果已存在则覆盖）
        with open(empty_path, 'w') as f:
            pass  # 只创建空文件

ann_dir = "./data/CODrone/train_part/annfiles"
empty_ann_dir = "./data/CODrone/train_part/empty_annfiles"
create_empty_annotations(ann_dir, empty_ann_dir)
