import argparse
import pyvips
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_vips_loaders():
    """Prints all foreign loaders pyvips can use and checks for key ones."""
    print("--- Available pyvips Loaders ---")
    has_openslide = False
    try:
        # This function is available in recent pyvips versions
        loaders = pyvips.base.get_loaders()
        if loaders:
            has_tiff = any(".tif" in s for s in loaders)
            has_openslide = any("openslide" in s for s in loaders)
            print(f"Found {len(loaders)} loaders.")
            print(f"  - Standard TIFF loader available: {'Yes' if has_tiff else 'No'}")
            print(f"  - OpenSlide loader available:    {'Yes' if has_openslide else 'No'}")
        else:
            print("Could not retrieve loader list.")
    except AttributeError:
        print("Could not query loaders: `pyvips.base.get_loaders()` not found (likely older pyvips version).")
    except Exception as e:
        print(f"An error occurred while querying loaders: {e}")
    print("---------------------------------")
    return has_openslide

def plot_and_save_mask(mask_path, output_path, read_level=4, delete_corrupt=False):
    """
    Loads a TIFF mask file, plots it, and saves the plot to a file.

    Args:
        mask_path (str or Path): The path to the input TIFF mask file.
        output_path (str or Path): The path to save the output plot image.
        read_level (int): The resolution level (page) to read from the TIFF file.
                          A higher number means lower resolution.
        delete_corrupt (bool): If True, deletes the source file if it's corrupt.
    """
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    if not mask_path.exists():
        print(f"Error: Input mask file not found at {mask_path}")
        return

    has_openslide_loader = get_vips_loaders()
    if not has_openslide_loader:
        print("\nWARNING: pyvips does not have the OpenSlide loader. It will only use the standard TIFF loader.")

    mask_image = None

    # --- Attempt 1: Load with `level` (OpenSlide method) ---
    print(f"\n--- Attempt 1: Loading '{mask_path.name}' with 'level={read_level}' (for OpenSlide-compatible files) ---")
    try:
        mask_image = pyvips.Image.new_from_file(str(mask_path), level=read_level)
        print("✅ Success: File loaded successfully using 'level'. This is an OpenSlide-compatible WSI format.")
    except pyvips.Error as e:
        print(f"❌ Failed: Could not load with 'level'. Error: {e}")
        print("   This is expected for standard pyramidal TIFFs.")

    # --- Attempt 2: Load with `page` (Standard TIFF method) ---
    if mask_image is None:
        print(f"\n--- Attempt 2: Loading '{mask_path.name}' with 'page={read_level}' (for standard pyramidal TIFFs) ---")
        try:
            mask_image = pyvips.Image.new_from_file(str(mask_path), page=read_level)
            print("✅ Success: File loaded successfully using 'page'. This is a standard pyramidal TIFF.")
        except pyvips.Error as e:
            print(f"❌ Failed: Could not load with 'page'. Error: {e}")

    if mask_image is None:
        print("\nFATAL: Could not load the image with either 'level' or 'page'. The file may be corrupt or unsupported.")
        return

    # If the mask has an alpha channel, take the first band
    if mask_image.bands > 1:
        print(f"\nMask has {mask_image.bands} bands. Visualizing the first band.")
        mask_image = mask_image[0]

    # Convert to a NumPy array for plotting
    is_corrupt = False
    try:
        mask_np = mask_image.numpy()
        print(f"Successfully processed mask. Shape: {mask_np.shape}, Dtype: {mask_np.dtype}")
    except pyvips.Error as e:
        is_corrupt = True
        print(f"\n❌ FATAL: Failed to convert pyvips image to numpy array. The file is likely corrupt.")
        print(f"   Error details: {e}")
        # Create a black placeholder image to indicate failure
        mask_np = np.zeros((256, 256), dtype=np.uint8)
        print("   A black placeholder image will be saved instead.")

        if delete_corrupt:
            try:
                mask_path.unlink()
                print(f"🗑️  Deleted corrupt file: {mask_path}")
            except OSError as del_e:
                print(f"🔥 Error deleting file {mask_path}: {del_e}")

    plotting = False if is_corrupt else True
    plotting = False
    if plotting:
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_np, cmap='gray')
        title = f"Mask: {mask_path.name}\n(Level {read_level})"
        if is_corrupt:
            title = f"CORRUPT FILE: {mask_path.name}\n(Placeholder Image)"
            plt.title(title, color='red')
        else:
            plt.title(title)
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
    parser = argparse.ArgumentParser(description="Load and plot all TIFF masks in a directory.")

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="/data/wsi_data/CAMELYON16/background_tissue/",
        help="Directory containing the input TIFF mask files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/data/ccardona/sclam_debug/",
        help="Directory to save the output plot images.",
    )
    parser.add_argument(
        "-l", "--level", type=int, default=4, help="The resolution level to read from the TIFF file. (default: 4)"
    )
    parser.add_argument(
        "--delete_corrupt",
        action="store_true",
        help="If set, delete TIFF files that are identified as corrupt.",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_dir)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Processing TIFF files from: {input_folder}")
    print(f"Saving plots to: {output_folder}")

    # Loop through all .tif files in the input folder
    for mask_path in sorted(input_folder.glob("*.tif")):
        print(f"\n{'='*20} Processing: {mask_path.name} {'='*20}")
        # The output filename will be the same as the input, but with a .png extension
        output_path = output_folder / mask_path.with_suffix(".png").name
        # Call the plotting function for each file
        plot_and_save_mask(mask_path, output_path, args.level, args.delete_corrupt)

    print(f"\n{'='*20} All files processed. {'='*20}")
