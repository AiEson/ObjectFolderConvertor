import os
import shutil
from render import ObjectRenderer, get_object_folder_path

def main():
    base_input_path = "/home/data3t/aieson/WCS_DATAS/Datasets"
    base_output_path = "/home/data3t/aieson/WCS_DATAS/Datasets/ObjectFolderResults"
    
    # Read list of modified objects
    try:
        with open('modified_objects.txt', 'r') as f:
            modified_objects = [int(line.strip()) for line in f]
    except FileNotFoundError:
        print("No modified_objects.txt file found!")
        return
    
    print(f"Found {len(modified_objects)} objects to re-render")
    
    # Process modified objects
    for obj_id in modified_objects:
        folder_name = get_object_folder_path(obj_id)
        input_path = os.path.join(base_input_path, folder_name, str(obj_id))
        output_path = os.path.join(base_output_path, str(obj_id))
        
        # Remove old renders if they exist
        if os.path.exists(output_path):
            print(f"Removing old renders for object {obj_id}")
            shutil.rmtree(output_path)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        try:
            print(f"Re-rendering object {obj_id} from {input_path}...")
            renderer = ObjectRenderer(base_path=input_path)
            renderer.render_orbit_sequence(output_dir=output_path, step_degrees=15)
            print(f"Successfully re-rendered object {obj_id}")
        except Exception as e:
            print(f"Error processing object {obj_id}: {str(e)}")
            continue

if __name__ == '__main__':
    main()
