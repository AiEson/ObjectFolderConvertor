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
        self.bg_color = [0.0, 0.0, 0.0]  # Black background
        self.fov = np.pi / 3.0  # 60 degrees field of view
        self.margin_factor = 1.3  # Add 30% margin around the object
        
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
                # print(f"Warning: Empty mesh detected, trying with processing enabled...")
                mesh = trimesh.load(str(self.obj_path), process=True, force='mesh')
            
            # Ensure the mesh has valid geometry
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                raise ValueError(f"Invalid mesh: no vertices or faces found in {self.obj_path}")
                
            # # print mesh information for debugging
            # print(f"Loaded mesh: {self.obj_path}")
            # print(f"Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")
            if hasattr(mesh.visual, 'uv'):
                # print(f"UV coordinates: {mesh.visual.uv.shape if mesh.visual.uv is not None else 'None'}")
                ...
            
            return mesh
            
        except Exception as e:
            # print(f"Error loading mesh {self.obj_path}: {str(e)}")
            raise
        
    def calculate_camera_distance (self, mesh):
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
        # print(f"Rendered image saved to: {output_path}")
        
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

    def mesh_to_point_cloud(self, mesh, n_points=10000, target_distance=5.0):
        """
        Convert mesh to point cloud with consistent scaling based on target camera distance.
        
        Args:
            mesh (trimesh.Trimesh): Input mesh
            n_points (int): Number of points to sample
            target_distance (float): Target camera distance (should match generate_images_txt.py)
            
        Returns:
            tuple: (points, colors) where:
                - points: np.ndarray of shape (n_points, 3) containing point coordinates
                - colors: np.ndarray of shape (n_points, 3) containing RGB colors
        """
        # Calculate current optimal camera distance for this mesh
        current_distance = self.calculate_camera_distance(mesh)
        
        # Calculate scaling factor to match target distance
        scale_factor = target_distance / current_distance
        
        # Create a copy of the mesh and scale it
        scaled_mesh = mesh.copy()
        
        # Center the mesh at origin before scaling
        centroid = scaled_mesh.centroid
        scaled_mesh.vertices -= centroid
        
        # Apply scaling
        scaled_mesh.apply_scale(scale_factor)
        
        # Sample points and get their barycentric coordinates
        points, face_indices = trimesh.sample.sample_surface(scaled_mesh, n_points)
        
        # Initialize colors array
        colors = np.zeros((n_points, 3), dtype=np.uint8)
        
        # Get colors based on the mesh's visual properties
        if mesh.visual.kind == 'texture':
            try:
                # Get UV coordinates for each face
                uv = mesh.visual.uv
                
                # Get texture image
                texture = mesh.visual.material.image
                
                # Convert texture to numpy array if needed
                if hasattr(texture, 'convert'):
                    texture = np.array(texture.convert('RGB'))
                
                # Get texture dimensions
                tex_height, tex_width = texture.shape[:2]
                
                # For each sampled point
                for i, face_idx in enumerate(face_indices):
                    # Get vertices of the face
                    face_vertices = mesh.faces[face_idx]
                    
                    # Get UV coordinates of the face vertices
                    face_uvs = uv[face_vertices]
                    
                    # Simple average of UV coordinates (this is an approximation)
                    point_uv = np.mean(face_uvs, axis=0)
                    
                    # Ensure UV coordinates are within [0, 1]
                    point_uv = np.clip(point_uv, 0, 1)
                    
                    # Convert UV to pixel coordinates
                    # Subtract small epsilon to ensure we don't get index out of bounds
                    px = min(int(point_uv[0] * (tex_width - 1)), tex_width - 1)
                    py = min(int((1 - point_uv[1]) * (tex_height - 1)), tex_height - 1)
                    
                    # Sample texture color
                    colors[i] = texture[py, px]
            except Exception as e:
                # If any error occurs during texture sampling, fall back to material color
                if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'diffuse'):
                    diffuse = np.array(mesh.visual.material.diffuse[:3]) * 255
                    colors[i] = diffuse.astype(np.uint8)
                else:
                    # Default color for metal barrel
                    colors[i] = [218, 165, 32]  # Golden yellow
                
        elif mesh.visual.kind == 'vertex':
            # For vertex colors, interpolate based on barycentric coordinates
            vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB only
            for i, face_idx in enumerate(face_indices):
                face_vertices = mesh.faces[face_idx]
                vert_colors = vertex_colors[face_vertices]
                colors[i] = np.mean(vert_colors, axis=0)
                
        elif mesh.visual.kind == 'face':
            # For face colors, directly use the face color
            face_colors = mesh.visual.face_colors[:, :3]  # RGB only
            colors = face_colors[face_indices]
            
        elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'diffuse'):
            # Try to get color from material if available
            diffuse = np.array(mesh.visual.material.diffuse[:3]) * 255
            colors[:] = diffuse.astype(np.uint8)
            
        else:
            # Default color for metal barrel
            colors[:] = [218, 165, 32]  # Golden yellow
            
        return points, colors

    def generate_points3D_txt(self, mesh, output_path, n_points=10000):
        """
        Generate points3D.txt in COLMAP format from the mesh.
        
        Args:
            mesh (trimesh.Trimesh): Input mesh
            output_path (str or Path): Path to save points3D.txt
            n_points (int): Number of points to sample
        """
        # Generate scaled point cloud with colors
        points, colors = self.mesh_to_point_cloud(mesh, n_points=n_points)
        
        # Prepare output path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write points3D.txt in COLMAP format
        with open(output_path, 'w') as f:
            # Write header
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {n_points}, mean track length: 0\n")
            
            # Write each point
            for i, (point, color) in enumerate(zip(points, colors)):
                f.write(f"{i} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{color[0]} {color[1]} {color[2]} "
                       f"0.0\n")

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
    
    return os.path.join(base_dir, str(obj_id))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render objects and generate point clouds')
    parser.add_argument('--obj_id', type=int, help='Object ID to process')
    parser.add_argument('--output_dir', type=str, default='../ObjectFolderResults', help='Directory to save renders')
    args = parser.parse_args()
    
    if args.obj_id is not None:
        # Process single object
        obj_path = get_object_folder_path(args.obj_id)
        renderer = ObjectRenderer(obj_path)
        
        # Load mesh
        mesh = renderer.load_mesh()
        
        # Generate renders
        renderer.render_orbit_sequence(output_dir=f'{args.output_dir}/{args.obj_id}', step_degrees=15)
        
        # Generate points3D.txt
        output_dir = Path(f'{args.output_dir}/{args.obj_id}')
        output_dir.mkdir(parents=True, exist_ok=True)
        renderer.generate_points3D_txt(mesh, output_dir / 'points3D.txt')
    else:
        print("Please specify an object ID with --obj_id")

if __name__ == '__main__':
    main()
