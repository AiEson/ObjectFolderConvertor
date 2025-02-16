import numpy as np
from scipy.spatial.transform import Rotation
import os

def euler_to_quaternion(azimuth, elevation):
    """Convert azimuth and elevation to quaternion in Hamilton convention"""
    # Convert to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    # Create rotation matrix
    # First rotate around Y (elevation)
    # Then rotate around Z (azimuth)
    R_y = Rotation.from_euler('y', elevation_rad)
    R_z = Rotation.from_euler('z', azimuth_rad)
    R = R_z * R_y
    
    # Convert to quaternion (already in Hamilton convention)
    quat = R.as_quat()  # Returns in (x, y, z, w)
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Reorder to (w, x, y, z)

def get_camera_position(azimuth, elevation, distance):
    """Calculate camera position in world coordinates"""
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    
    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.sin(elevation_rad)
    z = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    
    return np.array([x, y, z])

def generate_images_txt(output_path, distance=5.0):
    """Generate images.txt with camera poses from orbit trajectories"""
    # Define the three orbits from render.py
    orbits = [
        ('horizontal', 0),    # Horizontal orbit (elevation = 0)
        ('top', 45),         # Top-down orbit (elevation = 45)
        ('bottom', -45),     # Bottom-up orbit (elevation = -45)
    ]
    
    # Step size in degrees
    step_degrees = 15
    steps = int(360 / step_degrees)
    
    # Open file for writing
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        total_images = len(orbits) * steps
        f.write(f"# Number of images: {total_images}, mean observations per image: 1\n")
        
        # Generate poses for each orbit
        image_id = 1
        for orbit_name, elevation in orbits:
            for i in range(steps):
                azimuth = i * step_degrees
                
                # Get quaternion (world to camera)
                # Add 180 to azimuth because we want camera to look at center
                quat = euler_to_quaternion(azimuth + 180, -elevation)  # Negate elevation for camera convention
                
                # Get camera position
                pos = get_camera_position(azimuth, elevation, distance)
                
                # Convert position to translation (world to camera)
                # The translation in world-to-camera transform is -R * c
                # where c is the camera center in world coordinates
                R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                T = -R @ pos
                
                # Write camera pose
                image_name = f"{orbit_name}_{azimuth:03d}.png"
                f.write(f"{image_id} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                       f"{T[0]:.6f} {T[1]:.6f} {T[2]:.6f} 1 {image_name}\n")
                # Write points3D info (empty in this case)
                # f.write("")
                
                image_id += 1

if __name__ == '__main__':
    generate_images_txt('images.txt')
    print("Generated images.txt with camera poses")
