import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pyvips
from pathlib import Path

def generate_pyramidal_mask(xml_path, wsi_path, output_path, level=2):
    """
    Converts CAMELYON16 XML to a Tiled Pyramidal TIFF mask using pyvips.
    
    Args:
        xml_path: Path to .xml annotation
        wsi_path: Path to original .tif WSI
        output_path: Where to save the new _mask.tif
        level: The resolution level to render at (default 2 is a good balance).
               Note: Level 0 might be too large for RAM.
    """
    # 1. Load WSI with pyvips to get dimensions and downsample info
    # We use access='sequential' to be memory efficient
    wsi = pyvips.Image.new_from_file(str(wsi_path), access='sequential')
    
    # Get the downsample factor for the requested level
    # CAMELYON16 levels are usually powers of 2 (1, 2, 4, 8, 16, 32...)
    # If level=2, scale is 1/4. If level=4, scale is 1/16.
    scale_factor = 1.0 / (2**level) 
    
    target_width = int(wsi.width * scale_factor)
    target_height = int(wsi.height * scale_factor)
    
    print(f"Rendering mask at Level {level} ({target_width}x{target_height})")

    # 2. Create a blank NumPy array for drawing
    # This is where the actual rasterization happens.
    mask_np = np.zeros((target_height, target_width), dtype=np.uint8)

    # 3. Parse XML and scale coordinates
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for annotation in root.findall(".//Annotation"):
            points = []
            for coord in annotation.findall(".//Coordinate"):
                # Scale coordinates from Level 0 to our target level
                x = float(coord.get("X")) * scale_factor
                y = float(coord.get("Y")) * scale_factor
                points.append([x, y])
            
            if points:
                pts = np.array(points, dtype=np.int32)
                # color=1 because your dataset logic uses 'mask >= 1'
                cv2.fillPoly(mask_np, [pts], color=1)

    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return

    # 4. Convert NumPy array back to pyvips Image
    # We set the interpretation to 'grey12' (standard 8-bit mono)
    vips_mask = pyvips.Image.new_from_array(mask_np)

    # 5. Save as a Tiled, Pyramidal TIFF (Compatibility Mode)
    # These parameters match the CAMELYON16 standard:
    # - tile: enables tiling (prevents the corruption error)
    # - pyramid: creates the multi-res levels
    # - compression: LZW is lossless and standard for masks
    # - bigtiff: useful if the file ends up being > 4GB
    print(f"Saving pyramidal TIFF to {output_path}...")
    vips_mask.write_to_file(
        str(output_path),
        tile=True,
        pyramid=True,
        compression="lzw",
        bigtiff=True,
        tile_width=256,
        tile_height=256
    )
    print("Done.")

# --- Automation Loop ---
wsi_folder = Path("/data/wsi_data/CAMELYON16/images")
xml_folder = Path("/data/wsi_data/CAMELYON16/annotations")
mask_folder = Path("/data/wsi_data/CAMELYON16/masks")
mask_folder.mkdir(parents=True, exist_ok=True)

for xml_file in xml_folder.glob("*.xml"):
    # CAMELYON convention: 'tumor_001.xml' maps to 'tumor_001.tif'
    # but generates 'tumor_001_mask.tif'
    wsi_file = wsi_folder / (xml_file.stem + ".tif")
    out_file = mask_folder / (xml_file.stem + "_mask.tif")
    
    if wsi_file.exists():
        generate_pyramidal_mask(xml_file, wsi_file, out_file, level=2)
    else:
        print(f"Skipping {xml_file.name}: WSI not found at {wsi_file}")