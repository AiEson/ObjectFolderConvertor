import os
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from PIL import Image

# Add Mesa software renderer configuration
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'

class ObjectRenderer:
    def __init__(self, base_path='./3'):
        """
        Initialize the renderer with the base path containing obj/mtl files
        """
        self.base_path = Path(base_path)
        self.obj_path = self.base_path / 'model.obj'
        self.mtl_path = self.base_path / 'model.mtl'
        self.texture_path = self.base_path / 'textures'
        
        # Initialize renderer settings
        self.width = 1024
        self.height = 1024
        self.bg_color = [1.0, 1.0, 1.0]  # White background
        self.fov = np.pi / 3.0  # 60 degrees field of view
        self.margin_factor = 1.2  # Add 20% margin around the object
        
    def load_mesh(self):
        """
        Load the OBJ file with materials using trimesh, ensuring proper texture loading
        """
        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {self.obj_path}")
            
        try:
            # Load the mesh with materials, disable processing to preserve UVs
            mesh = trimesh.load(str(self.obj_path), process=False, force='mesh')
            
            # Verify texture loading
            if hasattr(mesh.visual, 'material') and mesh.visual.material is None:
                print(f"Warning: No material loaded for {self.obj_path}")
                
            if hasattr(mesh.visual, 'uv') and (mesh.visual.uv is None or len(mesh.visual.uv) == 0):
                print(f"Warning: No UV coordinates found for {self.obj_path}")
                
            # If mesh is empty or invalid, try loading with processing
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                print(f"Warning: Empty mesh detected, trying with processing enabled...")
                mesh = trimesh.load(str(self.obj_path), process=True, force='mesh')
            
            # Ensure the mesh has valid geometry
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                raise ValueError(f"Invalid mesh: no vertices or faces found in {self.obj_path}")
                
            # Print mesh information for debugging
            print(f"Loaded mesh: {self.obj_path}")
            print(f"Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")
            if hasattr(mesh.visual, 'uv'):
                print(f"UV coordinates: {mesh.visual.uv.shape if mesh.visual.uv is not None else 'None'}")
            
            return mesh
            
        except Exception as e:
            print(f"Error loading mesh {self.obj_path}: {str(e)}")
            raise
        
    def calculate_camera_distance(self, mesh):
        """
        Calculate optimal camera distance based on mesh bounds
        """
        bounds = mesh.bounds
        mesh_size = np.linalg.norm(bounds[1] - bounds[0])
        distance = (mesh_size * self.margin_factor) / (2 * np.tan(self.fov / 2))
        min_distance = mesh_size * 0.5
        return max(distance, min_distance)
        
    def get_camera_pose(self, azimuth, elevation, distance):
        """
        Calculate camera pose matrix for given angles and distance
        
        Args:
            azimuth (float): Horizontal angle in degrees
            elevation (float): Vertical angle in degrees
            distance (float): Distance from center
            
        Returns:
            np.ndarray: 4x4 camera pose matrix
        """
        # Convert angles to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        # Calculate camera position
        x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        
        # Create look-at matrix
        camera_pos = np.array([x, y, z])
        target = np.zeros(3)
        up = np.array([0, 1, 0])
        
        # Calculate camera orientation
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Construct pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        return pose
        
    def setup_scene(self, mesh, camera_pose):
        """
        Set up the pyrender scene with mesh and lighting
        """
        # Convert trimesh to pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        # Create scene
        scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=np.array([0.7, 0.7, 0.7, 1.0]))
        
        # Add mesh to scene at the center
        scene.add(mesh)
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=self.fov)
        scene.add(camera, pose=camera_pose)
        
        # Calculate light positions based on camera position
        camera_pos = camera_pose[:3, 3]
        light_distance = np.linalg.norm(camera_pos) * 0.8
        light_intensity = 3.0  # 增加主光源强度
        
        # Key light
        key_light_pos = camera_pos + np.array([light_distance, light_distance, 0])
        key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
        key_light_pose = np.eye(4)
        key_light_pose[:3, 3] = key_light_pos
        scene.add(key_light, pose=key_light_pose)
        
        # Fill light
        fill_light_pos = camera_pos + np.array([-light_distance, light_distance * 0.5, 0])
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity * 0.7)  # 增加填充光强度
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = fill_light_pos
        scene.add(fill_light, pose=fill_light_pose)
        
        return scene
        
    def render_view(self, azimuth=0, elevation=30):
        """
        Render the object from specified viewpoint
        """
        # Load and center mesh
        mesh = self.load_mesh()
        mesh.vertices -= mesh.centroid
        
        # Calculate camera distance
        camera_distance = self.calculate_camera_distance(mesh)
        
        # Get camera pose
        camera_pose = self.get_camera_pose(azimuth, elevation, camera_distance)
        
        # Setup scene
        scene = self.setup_scene(mesh, camera_pose)
        
        # Create renderer
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        
        # Render
        color, depth = renderer.render(scene)
        
        # Convert to PIL Image
        image = Image.fromarray(color)
        
        return image
        
    def save_render(self, output_path, azimuth=0, elevation=30):
        """
        Render and save the image
        """
        image = self.render_view(azimuth, elevation)
        image.save(output_path)
        print(f"Rendered image saved to: {output_path}")
        
    def render_orbit_sequence(self, output_dir='renders', step_degrees=15):
        """
        Render a sequence of images from three different orbits
        
        Args:
            output_dir (str): Directory to save renders
            step_degrees (float): Angular step size in degrees
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Calculate number of steps
        steps = int(360 / step_degrees)
        
        # Define the three orbits
        orbits = [
            ('horizontal', 0),    # Horizontal orbit (elevation = 0)
            ('top', 45),         # Top-down orbit (elevation = 45)
            ('bottom', -45),     # Bottom-up orbit (elevation = -45)
        ]
        
        # Render each orbit
        for orbit_name, elevation in orbits:
            for i in range(steps):
                azimuth = i * step_degrees
                output_path = output_dir / f'{orbit_name}_{azimuth:03d}.png'
                self.save_render(output_path, azimuth, elevation)

def get_object_folder_path(obj_id):
    """
    Get the path to the object folder based on its ID
    
    Args:
        obj_id (int): Object ID from 1 to 1000
        
    Returns:
        str: Path to the object folder
    """
    if obj_id <= 100:
        base_dir = "ObjectFolder1-100"
    else:
        start = ((obj_id - 1) // 100) * 100 + 1
        end = start + 99
        base_dir = f"ObjectFolder{start}-{end}"
    
    return base_dir

def main():
    base_input_path = "/path/to/Datasets"
    base_output_path = "/path/to/Datasets/ObjectFolderResults"
    
    # Create output directory if it doesn't exist
    os.makedirs(base_output_path, exist_ok=True)
    
    # Initialize renderer once for all objects
    renderer = None
    
    # Process all objects from 1 to 1000
    for obj_id in range(1, 1001):
        folder_name = get_object_folder_path(obj_id)
        input_path = os.path.join(base_input_path, folder_name, str(obj_id))
        output_path = os.path.join(base_output_path, str(obj_id))
        
        # Skip if input path doesn't exist
        if not os.path.exists(input_path):
            print(f"Skipping object {obj_id}: Input path {input_path} does not exist")
            continue
            
        # Create output directory for this object
        os.makedirs(output_path, exist_ok=True)
        
        try:
            print(f"Processing object {obj_id} from {input_path}...")
            renderer = ObjectRenderer(base_path=input_path)
            renderer.render_orbit_sequence(output_dir=output_path, step_degrees=15)
            print(f"Successfully rendered object {obj_id}")
        except Exception as e:
            print(f"Error processing object {obj_id}: {str(e)}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
