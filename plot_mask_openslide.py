import argparse
import openslide
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_and_save_mask_openslide(mask_path, output_path, read_level=4):
    """
    Loads a TIFF mask file using OpenSlide, plots it, and saves the plot.

    Args:
        mask_path (str or Path): The path to the input TIFF mask file.
        output_path (str or Path): The path to save the output plot image.
        read_level (int): The downsample level to read.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    if not mask_path.exists():
        print(f"Error: Input mask file not found at {mask_path}")
        return

    print(f"Opening mask with OpenSlide: {mask_path}")
    
    try:
        # Open the slide
        slide = openslide.OpenSlide(str(mask_path))
        
        # Check if the requested level exists
        if read_level >= slide.level_count:
            print(f"Warning: Level {read_level} not found. Slide only has {slide.level_count} levels.")
            read_level = slide.level_count - 1
            print(f"Falling back to maximum available level: {read_level}")

        # Get dimensions for the specific level
        level_dims = slide.level_dimensions[read_level]
        print(f"Loading level {read_level} with dimensions: {level_dims}")

        # read_region(location_at_level_0, level, size_at_requested_level)
        # Note: location is always (0,0) for the top-left of the whole mask
        mask_rgba = slide.read_region((0, 0), read_level, level_dims)

        # OpenSlide returns a PIL RGBA image. 
        # For masks, we usually want a single channel (Grayscale/L)
        mask_pil = mask_rgba.convert("L")
        mask_np = np.array(mask_pil)

        print(f"Successfully loaded mask. Shape: {mask_np.shape}, Dtype: {mask_np.dtype}")

    except Exception as e:
        print(f"Error loading mask with OpenSlide: {e}")
        return
    finally:
        if 'slide' in locals():
            slide.close()

    # Plotting
    plt.figure(figsize=(10, 10))
    # Use 'viridis' or 'gray'—if it's a binary mask, gray is usually clearer
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"OpenSlide Mask: {mask_path.name}\n(Level {read_level})")
    plt.colorbar(label="Pixel Intensity")
    plt.axis('off')
    plt.tight_layout()

    # Save the plot
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot a TIFF mask using OpenSlide.")
    
    parser.add_argument("-o", "--output_plot", type=str, default="/data/ccardona/sclam_debug/mask_plot_openslide.png",
                        help="Path to save the output plot image.")
    parser.add_argument("-l", "--level", type=int, default=4,
                        help="The resolution level to read. (default: 4)")

    args = parser.parse_args()
    
    # Path configuration
    folder_mask = "/data/wsi_data/CAMELYON16/masks_all/"
    mask_file = "test_042.tif_1.25_mask.png"
    full_mask_path = Path(folder_mask) / mask_file
    
    plot_and_save_mask_openslide(full_mask_path, args.output_plot, args.level)