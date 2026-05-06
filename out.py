import os
from pathlib import Path

def diagnose_directory_structure():
    base_path = "/data5/zhangjiening/Data_Generation/airport_placement_tiled_multi_results_labels_full"
    
    print("=== 目录结构诊断 ===")
    print(f"检查路径: {base_path}")
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"❌ 错误: 基础路径不存在!")
        return False
    
    print(f"✅ 基础路径存在")
    
    # 列出基础路径下的所有内容
    try:
        items = os.listdir(base_path)
        print(f"\n基础路径下的内容 ({len(items)} 个):")
        for item in items[:20]:  # 只显示前20个
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}")
            else:
                print(f"  📄 {item}")
        
        if len(items) > 20:
            print(f"  ... 还有 {len(items)-20} 个")
        
        # 查找outputs_batch_1030文件夹
        print(f"\n=== 查找 outputs_batch_1030 文件夹 ===")
        batch_folders = []
        for root, dirs, files in os.walk(base_path):
            if "outputs_batch_1030" in dirs:
                batch_path = os.path.join(root, "outputs_batch_1030")
                batch_folders.append(batch_path)
                print(f"✅ 找到: {batch_path}")
        
        if not batch_folders:
            print("❌ 没有找到任何 outputs_batch_1030 文件夹")
            print("\n当前存在的文件夹结构:")
            for root, dirs, files in os.walk(base_path, maxdepth=2):
                level = root.replace(base_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for dirname in dirs[:5]:  # 只显示前5个子文件夹
                    print(f"{subindent}{dirname}/")
                if len(dirs) > 5:
                    print(f"{subindent}...")
                break
        else:
            print(f"\n🎉 总共找到 {len(batch_folders)} 个 outputs_batch_1030 文件夹")
            
            # 检查其中一个文件夹的内容
            if batch_folders:
                sample_folder = batch_folders[0]
                print(f"\n=== 检查示例文件夹内容: {sample_folder} ===")
                try:
                    sample_files = os.listdir(sample_folder)
                    image_files = [f for f in sample_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                    print(f"文件夹中的文件: {len(sample_files)} 个")
                    print(f"图片文件: {len(image_files)} 个")
                    if image_files:
                        print("前5个图片文件:")
                        for img in image_files[:5]:
                            print(f"  📷 {img}")
                except Exception as e:
                    print(f"❌ 无法读取文件夹内容: {e}")
        
    except Exception as e:
        print(f"❌ 诊断过程中出错: {e}")
        return False
    
    return True

if __name__ == "__main__":
    diagnose_directory_structure()