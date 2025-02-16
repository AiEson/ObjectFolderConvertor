import os
import re
from pathlib import Path

def get_object_folder_path(obj_id):
    """Get the path to the object folder based on its ID"""
    if obj_id <= 100:
        base_dir = "ObjectFolder1-100"
    else:
        start = ((obj_id - 1) // 100) * 100 + 1
        end = start + 99
        base_dir = f"ObjectFolder{start}-{end}"
    return base_dir

def fix_mtl_file(mtl_path):
    """Fix texture paths in MTL file by adding 'textures/' prefix if needed"""
    modified = False
    texture_keys = ['map_Kd', 'map_Bump', 'map_Ks', 'map_Ka', 'map_Ns', 'map_d', 'bump', 'disp']
    
    # Read the original content
    with open(mtl_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    new_lines = []
    for line in lines:
        line = line.strip()
        if any(line.startswith(key + ' ') for key in texture_keys):
            # Split line into key and value
            parts = line.split(' ', 1)
            if len(parts) == 2:
                key, path = parts
                # If path doesn't start with 'textures/' and the texture exists in textures folder
                if not path.startswith('textures/'):
                    texture_path = os.path.join(os.path.dirname(mtl_path), 'textures', path)
                    if os.path.exists(texture_path):
                        line = f"{key} textures/{path}"
                        modified = True
        new_lines.append(line + '\n')
    
    # Write back if modified
    if modified:
        with open(mtl_path, 'w') as f:
            f.writelines(new_lines)
    
    return modified

def main():
    base_path = "/home/data3t/aieson/WCS_DATAS/Datasets"
    modified_objects = []
    
    # Process all objects from 1 to 1000
    for obj_id in range(1, 1001):
        folder_name = get_object_folder_path(obj_id)
        obj_path = os.path.join(base_path, folder_name, str(obj_id))
        mtl_path = os.path.join(obj_path, "model.mtl")
        
        if os.path.exists(mtl_path):
            try:
                if fix_mtl_file(mtl_path):
                    modified_objects.append(obj_id)
                    print(f"Fixed MTL file for object {obj_id}")
            except Exception as e:
                print(f"Error processing object {obj_id}: {str(e)}")
    
    # Save list of modified objects
    with open('modified_objects.txt', 'w') as f:
        f.write('\n'.join(map(str, modified_objects)))
    
    print(f"\nTotal modified objects: {len(modified_objects)}")
    print("Modified object IDs have been saved to 'modified_objects.txt'")
    return modified_objects

if __name__ == '__main__':
    main()
