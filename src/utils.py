from src.quad import QuadTreeNode
import cv2
import numpy as np


def calculate_region_stats_color(image_color, x, y, width, height, calculate_gradient=False, grad_mag=None):
    """Calculates stats for a color region."""
    stats = {}
    if width <= 0 or height <= 0:
        stats['std_dev_channels'] = np.array([0.0] * image_color.shape[2])
        stats['mean_channels'] = np.array([0.0] * image_color.shape[2])
        if calculate_gradient:
            stats['std_dev_gradient'] = 0.0
        return stats

    region = image_color[y : y + height, x : x + width]
    if region.size == 0:
        stats['std_dev_channels'] = np.array([0.0] * image_color.shape[2])
        stats['mean_channels'] = np.array([0.0] * image_color.shape[2])
        if calculate_gradient:
            stats['std_dev_gradient'] = 0.0
        return stats

    # Calculate stats per channel (axis=0 means collapse rows, axis=1 means collapse columns)
    stats['mean_channels'] = np.mean(region, axis=(0, 1))
    stats['std_dev_channels'] = np.std(region, axis=(0, 1))

    # Calculate gradient standard deviation if needed
    if calculate_gradient and grad_mag is not None:
        grad_region = grad_mag[y : y + height, x : x + width]
        if grad_region.size > 0:
            stats['std_dev_gradient'] = np.std(grad_region)
        else:
            stats['std_dev_gradient'] = 0.0

    return stats

def calculate_gradient_magnitude(image_gray):
    """Calculates the magnitude of the image gradient using Sobel operators."""
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    # Normalize to 0-255 range for potential visualization or consistent thresholding
    if np.max(grad_mag) > 0:
        grad_mag = (grad_mag / np.max(grad_mag) * 255).astype(np.uint8)
    else:
         grad_mag = np.zeros_like(grad_mag, dtype=np.uint8)
    return grad_mag


def build_quadtree_color(node, image_color, max_depth, intensity_threshold,
                         use_texture=False, texture_threshold=0, grad_mag=None):
    """Recursively builds the quadtree for color images."""
    # Calculate stats for the current node's region
    node.stats = calculate_region_stats_color(
        image_color, node.x, node.y, node.width, node.height,
        calculate_gradient=use_texture, grad_mag=grad_mag
    )
    # Store the average color for potential leaf node rendering
    node.average_value = node.stats['mean_channels']


    # Check if the node should be split
    if not node.is_splittable(max_depth, intensity_threshold, use_texture, texture_threshold):
        node.is_leaf = True
        return node # Return leaf node

    # --- Split the node into four children ---
    half_width1 = node.width // 2
    half_height1 = node.height // 2
    half_width2 = node.width - half_width1
    half_height2 = node.height - half_height1

    child_coords = [
        (node.x, node.y, half_width1, half_height1),                           # TL
        (node.x + half_width1, node.y, half_width2, half_height1),             # TR
        (node.x, node.y + half_height1, half_width1, half_height2),            # BL
        (node.x + half_width1, node.y + half_height1, half_width2, half_height2) # BR
    ]

    for i, (cx, cy, cw, ch) in enumerate(child_coords):
        if cw > 0 and ch > 0:
            child_node = QuadTreeNode(cx, cy, cw, ch, node.depth + 1)
            # Pass necessary info down
            node.children[i] = build_quadtree_color(
                child_node, image_color, max_depth, intensity_threshold,
                use_texture, texture_threshold, grad_mag
            )

    return node

def draw_segmentation_color(node, output_image):
    """Traverses the tree and draws the color segmentation."""
    if node is None:
        return

    if node.is_leaf:
        x, y, w, h = int(node.x), int(node.y), int(node.width), int(node.height)
        # Get average BGR color, ensure it's valid uint8
        avg_color = np.clip(node.average_value, 0, 255).astype(np.uint8) if node.average_value is not None else (0,0,0)
        # Fill rectangle with the average color
        output_image[y : y + h, x : x + w] = avg_color
    else:
        for child in node.children:
            draw_segmentation_color(child, output_image)

def draw_boundaries(node, output_image, color=(0, 255, 0), thickness=1):
    """Traverses the tree and draws the boundaries on the output image."""
    if node is None:
        return

    if not node.is_leaf:
        # Draw boundaries for internal nodes (which represent splits)
         # Ensure coordinates and dimensions are integers
        x1, y1 = int(node.x), int(node.y)
        x2, y2 = int(node.x + node.width -1), int(node.y + node.height -1) # Use -1 for inclusive bounds

        # Calculate midpoints for drawing split lines
        mid_x = x1 + node.width // 2
        mid_y = y1 + node.height // 2

        # Draw vertical line (if width > 0)
        if node.width > 0:
             cv2.line(output_image, (mid_x, y1), (mid_x, y2), color, thickness)
        # Draw horizontal line (if height > 0)
        if node.height > 0:
             cv2.line(output_image, (x1, mid_y), (x2, mid_y), color, thickness)

        # Recursively draw for children
        for child in node.children:
            draw_boundaries(child, output_image, color, thickness)
    # Optional: you could also draw the outer box of leaf nodes if desired
    # else: # node.is_leaf
    #    x1, y1 = int(node.x), int(node.y)
    #    x2, y2 = int(node.x + node.width), int(node.y + node.height)
    #    cv2.rectangle(output_image, (x1, y1), (x2, y2), (255,0,0), 1) # Example: Red boundary for leaves
