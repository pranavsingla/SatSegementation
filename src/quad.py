import numpy as np
import matplotlib.pyplot as plt

# Optional: For true multi-spectral images (e.g., GeoTIFF)
# import rasterio

class QuadTreeNode:
    """Represents a node in the Quadtree."""
    def __init__(self, x, y, width, height, depth):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth
        self.is_leaf = False
        self.children = [None, None, None, None]  # TL, TR, BL, BR
        # Store stats per channel for color images
        self.average_value = None # Now potentially multi-channel (e.g., avg BGR)
        self.stats = {} # Dictionary to hold 'std_dev_channels', 'mean_channels', 'std_dev_gradient'

    def is_splittable(self, max_depth, intensity_threshold, use_texture=False, texture_threshold=0, min_size=1):
        """Check if the node should be split based on criteria."""
        if self.depth >= max_depth:
            return False
        if self.width <= min_size or self.height <= min_size:
            return False

        # 1. Check Intensity Homogeneity (any channel)
        is_inhomogeneous_intensity = False
        if 'std_dev_channels' in self.stats:
             # Split if *any* channel's std dev exceeds the threshold
            if np.any(self.stats['std_dev_channels'] > intensity_threshold):
                 is_inhomogeneous_intensity = True
        else:
             print(f"Warning: Intensity stats not calculated for node at ({self.x},{self.y})")
             return False # Cannot decide without stats

        # 2. (Optional) Check Texture Homogeneity (gradient std dev)
        is_inhomogeneous_texture = False
        if use_texture:
             if 'std_dev_gradient' in self.stats:
                  if self.stats['std_dev_gradient'] > texture_threshold:
                       is_inhomogeneous_texture = True
             else:
                  # Gradient might not be calculated if region is too small
                  pass # Assume homogeneous texture if gradient wasn't calculated

        # Decision to split:
        # Split if intensity is inhomogeneous.
        # If using texture, also split if texture is inhomogeneous, *even if* intensity is homogeneous (or adjust logic as needed)
        # Common approach: split if intensity OR (if enabled) texture is inhomogeneous
        if use_texture:
            return is_inhomogeneous_intensity or is_inhomogeneous_texture
        else:
            return is_inhomogeneous_intensity