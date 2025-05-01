import cv2
from src.quad import QuadTreeNode
from src.utils import  calculate_gradient_magnitude, build_quadtree_color, draw_segmentation_color, draw_boundaries, collect_quadtree_stats
import numpy as np
import time
import matplotlib.pyplot as plt
import sys



def main(image_path, intensity_threshold=25, max_depths_to_test=[4, 6, 8], use_texture_criterion=True, texture_threshold=30):

    image_color = cv2.imread(image_path)
    if image_color is None:
        print(f"Error: Could not load image at {image_path}")
        sys.exit()

    height, width, channels = image_color.shape
    print(f"Image loaded: {image_path} ({width}x{height}, {channels} channels)")

    # --- Pre-calculate Gradient Magnitude (if using texture) ---
    gradient_magnitude = None
    if use_texture_criterion:
        print("Calculating gradient magnitude...")
        # Convert to grayscale for gradient calculation
        image_gray_for_grad = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        gradient_magnitude = calculate_gradient_magnitude(image_gray_for_grad)
        print("Gradient magnitude calculated.")


    results = {} # Store results {depth: (segmented_img, boundary_img, time)}

    # --- Process for each depth ---
    for max_depth in max_depths_to_test:
        print(f"\nProcessing for max_depth = {max_depth}...")
        start_time = time.time()

        root_node = QuadTreeNode(0, 0, width, height, depth=0)

        quadtree_root = build_quadtree_color(
            root_node, image_color, max_depth, intensity_threshold,
            use_texture=use_texture_criterion,
            texture_threshold=texture_threshold,
            grad_mag=gradient_magnitude # Pass pre-calculated gradient
        )

        # --- compute exact stats ---
        leaf_count, total_area = collect_quadtree_stats(quadtree_root)
        mean_region_size = total_area / leaf_count if leaf_count else 0

        # store or print them
        print(f"Depth={max_depth:2d} →  leaves={leaf_count:4d}, "
            f"mean region size={mean_region_size:.1f} px², "
            f"time={processing_time:.2f}s")

        # you can also save into results:
        results[max_depth] = {
            'segmented': segmented_image_color,
            'boundary': boundary_image_color,
            'time': processing_time,
            'leaves': leaf_count,
            'mean_area': mean_region_size
        }

        # Create output image for color segmentation visualization
        segmented_image_color = np.zeros_like(image_color)
        draw_segmentation_color(quadtree_root, segmented_image_color)

        # Boundary visualization (use the same draw_boundaries function)
        boundary_image_color = image_color.copy()
        draw_boundaries(quadtree_root, boundary_image_color, color=(0, 255, 0), thickness=1)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Finished max_depth = {max_depth} in {processing_time:.2f} seconds.")

        results[max_depth] = (segmented_image_color, boundary_image_color, processing_time)
    # --- Display Results ---
    num_depths = len(max_depths_to_test)
    plt.figure(figsize=(6 * num_depths, 12)) # Adjust figure size

    plt.subplot(3, num_depths, 1)
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)) # Display original in RGB
    plt.title('Original Image')
    plt.axis('off')

    for i, max_depth in enumerate(max_depths_to_test):
        segmented_img, boundary_img, proc_time = results[max_depth]

        # Plot Segmented Image (Color)
        plt.subplot(3, num_depths, num_depths + i + 1)
        plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)) # Show segmented in RGB
        plt.title(f'Segmented (Depth={max_depth})\nTime: {proc_time:.2f}s')
        plt.axis('off')

        # Plot Boundary Image (Color)
        plt.subplot(3, num_depths, 2 * num_depths + i + 1)
        plt.imshow(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)) # Show boundaries in RGB
        plt.title(f'Boundaries (Depth={max_depth})')
        plt.axis('off')

    plt.tight_layout(pad=2.0)
    title_str = f'Quadtree Segmentation (Intensity Thr={intensity_threshold}'
    if use_texture_criterion:
         title_str += f', Texture Thr={texture_threshold})'
    else:
         title_str += ')'
    plt.suptitle(title_str, fontsize=16, y=1.03)
    plt.show()




