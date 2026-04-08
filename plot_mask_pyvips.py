import argparse
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _detect_openslide(img_fname):
    """
    Detects whether an image is better handled by the openslide loader (`level`)
    or the standard tiff loader (`page`).
    """
    try:
        # Try to open with 'page', which is for standard pyramidal TIFFs.
        # If it works, it's not an openslide-specific file.
        pyvips.Image.new_from_file(img_fname, page=0)
        return False
    except pyvips.Error:
        # If opening with 'page' fails, it's likely an openslide-compatible
        # format (like .svs) that requires the 'level' argument.
        return True

def plot_and_save_mask(mask_path, output_path, read_level=4):
    """
    Loads a TIFF mask file, plots it, and saves the plot to a file.

    Args:
        mask_path (str or Path): The path to the input TIFF mask file.
        output_path (str or Path): The path to save the output plot image.
        read_level (int): The resolution level (page) to read from the TIFF file.
                          A higher number means lower resolution.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    if not mask_path.exists():
        print(f"Error: Input mask file not found at {mask_path}")
        return

    print(f"Loading mask: {mask_path} at level/page {read_level}...")
    try:
        # Auto-detect whether to use 'level' (for openslide formats) or 'page' (for standard TIFFs)
        use_openslide = _detect_openslide(str(mask_path))
        if use_openslide:
            print("-> Detected OpenSlide compatible format, using 'level' argument.")
            mask_image = pyvips.Image.new_from_file(str(mask_path), level=read_level)
        else:
            print("-> Detected standard TIFF format, using 'page' argument.")
            mask_image = pyvips.Image.new_from_file(str(mask_path), page=read_level)

        # If the mask has an alpha channel, take the first band
        if mask_image.bands > 1:
            print(f"Mask has {mask_image.bands} bands. Visualizing the first band.")
            mask_image = mask_image[0]

        # Convert to a NumPy array for plotting
        mask_np = mask_image.numpy()
        print(f"Successfully loaded mask. Shape: {mask_np.shape}, Dtype: {mask_np.dtype}")

    except pyvips.Error as e:
        print(f"Error loading mask with pyvips: {e}")
        print("This could indicate a corrupted file or an issue with the read level.")
        return

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Mask: {mask_path.name}\n(Level {read_level})")
    plt.colorbar(label="Pixel Intensity")
    plt.axis('off')
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Close the plot to free memory
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot a TIFF mask file.")
    
    parser.add_argument("-o", "--output_plot", type=str, default="/data/ccardona/sclam_debug/mask_plot.png",
                        help="Path to save the output plot image. (default: mask_plot.png)")
    parser.add_argument("-l", "--level", type=int, default=4,
                        help="The resolution level (page) to read from the TIFF file. (default: 4)")

    args = parser.parse_args()
    folder_mask= f"/data/wsi_data/CAMELYON16/background_tissue/"
    mask_path = f"{folder_mask}/tumor_012_tissue.tif"
    plot_and_save_mask(mask_path, args.output_plot, args.level)
