import streamlit as st
import io
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from src.quad import QuadTreeNode
from src.utils import calculate_gradient_magnitude, build_quadtree_color, draw_segmentation_color, draw_boundaries

def process_image_and_plot(image_color, intensity_threshold=25, max_depths_to_test=[4, 6, 8], use_texture_criterion=True, texture_threshold=30):
    """
    Processes the image using quadtree for multiple depths and returns a matplotlib figure.
    Args:
        image_color (np.array): Input color image (BGR format from OpenCV).
        ... other params ...
    Returns:
        matplotlib.figure.Figure: The figure containing the plots, or None if error.
    """
    if image_color is None:
        st.error("Image data is invalid.")
        return None

    height, width, channels = image_color.shape
    st.info(f"Processing image ({width}x{height}, {channels} channels)...")

    # --- Pre-calculate Gradient Magnitude ---
    gradient_magnitude = None
    if use_texture_criterion:
        with st.spinner("Calculating gradient magnitude..."):
            try:
                image_gray_for_grad = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
                gradient_magnitude = calculate_gradient_magnitude(image_gray_for_grad)
                st.info("Gradient magnitude calculated.")
            except Exception as e:
                 st.error(f"Error calculating gradient: {e}")
                 use_texture_criterion = False # Disable texture if gradient fails


    results = {} # Store results {depth: (segmented_img, boundary_img, time)}
    processing_successful = True

    # --- Process for each depth ---
    for max_depth in max_depths_to_test:
        st.info(f"Running segmentation for max_depth = {max_depth}...")
        start_time = time.time()

        try:
            root_node = QuadTreeNode(0, 0, width, height, depth=0)
            # Use a context manager for spinner during intensive computation
            with st.spinner(f'Building quadtree (Depth {max_depth})...'):
                quadtree_root = build_quadtree_color(
                    root_node, image_color, max_depth, intensity_threshold,
                    use_texture=use_texture_criterion,
                    texture_threshold=texture_threshold,
                    grad_mag=gradient_magnitude
                )

            with st.spinner(f'Generating output images (Depth {max_depth})...'):
                segmented_image_color = np.zeros_like(image_color)
                draw_segmentation_color(quadtree_root, segmented_image_color)

                boundary_image_color = image_color.copy()
                draw_boundaries(quadtree_root, boundary_image_color, color=(0, 255, 0), thickness=1)

        except Exception as e:
            st.error(f"Error during processing for depth {max_depth}: {e}")
            results[max_depth] = (None, None, 0) # Mark as failed
            processing_successful = False
            continue # Skip to next depth

        end_time = time.time()
        processing_time = end_time - start_time
        st.info(f"Finished max_depth = {max_depth} in {processing_time:.2f} seconds.")

        results[max_depth] = (segmented_image_color, boundary_image_color, processing_time)

    if not processing_successful and not results:
         st.error("All processing depths failed.")
         return None

    # --- Create Matplotlib Figure ---
    num_depths = len(max_depths_to_test)
    if num_depths == 0:
        st.warning("No depths selected for processing.")
        return None

    fig, axes = plt.subplots(3, num_depths, figsize=(6 * num_depths, 10)) # Adjust figure size
    # Ensure axes is always a 2D array for consistent indexing, even if num_depths=1
    if num_depths == 1:
        axes = axes.reshape(3, 1)

    # Plot Original Image only once
    # Make space for original image (plot it in the first slot of the first row)
    axes[0, 0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    # Turn off other axes in the first row if num_depths > 1
    for j in range(1, num_depths):
         axes[0, j].axis('off')


    plot_col_index = 0
    for i, max_depth in enumerate(max_depths_to_test):
        segmented_img, boundary_img, proc_time = results.get(max_depth, (None, None, 0))

        # Plot Segmented Image (Color) - Middle Row
        if segmented_img is not None:
            axes[1, plot_col_index].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
            axes[1, plot_col_index].set_title(f'Segmented (Depth={max_depth})\nTime: {proc_time:.2f}s')
        else:
             axes[1, plot_col_index].set_title(f'Segmented (Depth={max_depth})\nFailed')
        axes[1, plot_col_index].axis('off')

        # Plot Boundary Image (Color) - Bottom Row
        if boundary_img is not None:
            axes[2, plot_col_index].imshow(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB))
            axes[2, plot_col_index].set_title(f'Boundaries (Depth={max_depth})')
        else:
            axes[2, plot_col_index].set_title(f'Boundaries (Depth={max_depth})\nFailed')
        axes[2, plot_col_index].axis('off')

        plot_col_index += 1


    fig.tight_layout(pad=1.5) # Adjust padding
    title_str = f'Quadtree Segmentation (Intensity Thr={intensity_threshold}'
    if use_texture_criterion:
         title_str += f', Texture Thr={texture_threshold})'
    else:
         title_str += ')'
    fig.suptitle(title_str, fontsize=16, y=1.0) # Adjust title position slightly

    return fig



# --- Streamlit App UI ---
st.set_page_config(layout="wide") # Use wide layout for better plot display
st.title("🌳 Quadtree Image Segmentation App 🌳")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"])

intensity_thresh = st.sidebar.slider("Intensity Threshold (Std Dev)", 1, 100, 25)
use_texture = st.sidebar.checkbox("Use Texture Criterion (Gradient Std Dev)", value=True)
texture_thresh = st.sidebar.slider("Texture Threshold (Std Dev)", 1, 100, 30, disabled=(not use_texture))

# Allow selecting multiple depths
default_depths = [4, 6, 8]
available_depths = list(range(1, 13)) # Limit max depth reasonably
selected_depths = st.sidebar.multiselect("Select Max Depths to Test", options=available_depths, default=default_depths)
selected_depths.sort() # Keep depths ordered

# --- Main Area ---
if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode image using OpenCV
    cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_image is not None:
        st.subheader("Original Image")
        # Display original image (converting BGR to RGB for st.image)
        st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption=f"Original: {uploaded_file.name}", use_container_width=True)

        st.sidebar.markdown("---") # Separator
        run_button = st.sidebar.button("▶️ Run Segmentation", use_container_width=True)

        if run_button:
            if not selected_depths:
                st.warning("Please select at least one depth level to test.")
            else:
                st.subheader("Segmentation Results")
                # Call the processing function
                fig = process_image_and_plot(
                    cv_image, # Pass the decoded image
                    intensity_threshold=intensity_thresh,
                    max_depths_to_test=selected_depths,
                    use_texture_criterion=use_texture,
                    texture_threshold=texture_thresh
                )

                # Display the Matplotlib figure in Streamlit
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.error("Failed to generate segmentation plot.")

    else:
        st.error("Could not decode the uploaded image. Please try a different file (jpg, png, tif).")

else:
    st.info("Upload an image using the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.markdown("App developed using Streamlit.")
