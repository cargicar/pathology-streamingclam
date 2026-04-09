import pyvips
import torch
import math

import pandas as pd

from pathlib import Path
import numpy as np # Add this import
from torch.utils.data import Dataset
import albumentationsxl as A

# A.OneOrOther(A.OneOf([A.Blur(), A.GaussianBlur(sigma_limit=7)]), A.Sharpen()),
# A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
# A.RandomGamma(gamma_limit=(90, 110)),

augmentations = A.Compose(
    [
        A.Flip(),
        A.HueSaturationValue(p=0.5),
        A.Rotate(),
    ], 
    is_check_shapes=False,
)


class StreamingClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir: Path | str,
        csv_file: str | pd.DataFrame,
        tile_size: int,
        img_size: int,
        read_level: int,
        transform: A.BaseCompose | None = None,
        mask_dir: Path | str | None = None,
        mask_suffix: str = "_tissue",
        variable_input_shapes: bool = False,
        tile_stride: int | None = None,
        network_output_stride: int = 1,
        filetype=".tif",
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.network_output_stride = network_output_stride
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform

        if not isinstance(csv_file, pd.DataFrame):
            self.classification_frame = pd.read_csv(csv_file)
        else:
            self.classification_frame = csv_file

        # Will be populated in check_csv function
        self.data_paths = {"images": [], "masks": [], "labels": []}

        self.random_crop = A.RandomCrop(self.img_size, self.img_size, p=1.0)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory {self.img_dir} not found or doesn't exist")
        self.check_csv()

        self.labels = self.data_paths["labels"]

    def check_csv(self):
        """Check if entries in csv file exist"""

        included = {"images": [], "masks": [], "labels": []} if self.mask_dir else {"images": [], "labels": []}
        #for i in range(len(self)):
        for i in range(len(self.classification_frame)): # Iterate over the original DataFrame length
            images, label = self.get_img_path(i)  #

            # Files can be just images, but also image, mask
            all_exist = True
            for file in images:
                if not file.exists():
                    print(f"WARNING {file} not found, excluded both image and mask (if present)!")
                    all_exist = False
                    break

            if not all_exist:
                continue

            included["images"].append(images[0])
            included["labels"].append(label)

            if self.mask_dir:
                included["masks"].append(images[1])

        self.data_paths = included

    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        label = self.classification_frame.iloc[idx, 1]

        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            #mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(self.filetype)
            imag_path = Path(img_fname)
            mask_path = self.mask_dir / f"{imag_path.stem}{self.mask_suffix}{self.filetype}"
            return [img_path, mask_path], label

        return [img_path], label

    def get_img_pairs(self, idx):
        images = {"image": None}

        img_fname = str(self.data_paths["images"][idx])
        label = int(self.data_paths["labels"][idx])
        try:
            # For openslide-compatible images (e.g. .svs, some .tif)
            image = pyvips.Image.new_from_file(img_fname, level=self.read_level)
        except pyvips.error.Error:
            # Fallback for standard pyramidal TIFFs
            image = pyvips.Image.new_from_file(img_fname, page=self.read_level)

        # openslide can produce RGBA, and some TIFFs might have an alpha channel
        # flatten to RGB so downstream expects 3 bands
        if image.bands == 4:
            image = image.flatten()
        images["image"] = image

        if self.mask_dir:
            mask_fname = str(self.data_paths["masks"][idx])
            # The masks are standard TIFF files, which are loaded using the `page`
            # argument to select a resolution from the pyramid. The `level` argument
            # is specific to formats that pyvips opens via the openslide library (e.g., .svs),
            # which is not the case for these mask files.
            mask = pyvips.Image.new_from_file(mask_fname, page=self.read_level)
            if mask.bands == 4:
                mask = mask.flatten()

            #J: Suggestion 
            # mask_meta = pyvips.Image.new_from_file(mask_fname)
            # n_mask_pages = mask_meta.get('n-pages') if mask_meta.get_typeof("n-pages") else 1
            # mask = pyvips.Image.new_from_file(mask_fname, page=n_mask_pages-1)

            # With the mask loaded at the same level, its dimensions should match the image.
            # This resize is now mostly a safeguard for minor dimension mismatches.
            ratio = image.width / mask.width
            images["mask"] = mask.resize(ratio, kernel="nearest")

        return images, label, img_fname

    def __getitem__(self, idx):
        sample, label, img_fname = self.get_img_pairs(idx)

        if self.transform:
            sample = self.transform(**sample)

        pad_to_tile_size = sample["image"].width < self.tile_size or sample["image"].height < self.tile_size
        # Get the resize op depending on image size
        resize_op = self.get_resize_op(pad_to_tile_size=pad_to_tile_size)
        sample = resize_op(**sample)

        # if after padding to the tile stride the image is bigger than img_size, crop it (upper bound on memory)
        if sample["image"].width * sample["image"].height > self.img_size**2:
            sample = self.random_crop(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        if "mask" in sample.keys():
            # Resize to streamingclam output stride, with max pool kernel
            # Rescale between model max pool and pyvips might not exactly align, so calculate new scale values
            new_height = math.ceil(sample["mask"].height / self.network_output_stride)
            vscale = new_height / sample["mask"].height

            new_width = math.ceil(sample["mask"].width / self.network_output_stride)
            hscale = new_width / sample["mask"].width

            sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")

            # Old breakpoint() was here. Moving it after potential error handling
            # breakpoint()

            # Handle potential pyvips errors during mask conversion
            # The `albumentationsxl.ToTensor` transform expects a pyvips.Image (or PIL Image)
            # that it can call `.numpy()` on.
            # We explicitly convert to numpy, handle errors, then convert back to pyvips.Image
            # to satisfy the transform.
            try:
                mask_np = sample["mask"].numpy()
            except pyvips.Error as e:
                mask_file_path = str(self.data_paths["masks"][idx])
                print(f"WARNING: pyvips error converting mask file '{mask_file_path}' to numpy. Replacing with all-zero mask: {e}")
                # Create an all-zero mask of the expected shape (H, W)
                mask_np = np.zeros((sample["mask"].height, sample["mask"].width), dtype=np.uint8)
            
            # Convert the numpy array back to a pyvips.Image object for the ToTensor transform
            sample["mask"] = pyvips.Image.new_from_array(mask_np)

        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False) # Ensure this line is outside the if 'mask' block if it was previously inside
        sample = to_tensor(**sample)

        # To ToTensor does not support cast to bool arrays yet, so do here
        if "mask" in sample.keys():
            sample["mask"] = sample["mask"] >= 1

        sample["label"] = torch.tensor(label)
        sample["image_name"] = Path(img_fname).stem
        return sample

    def __len__(self):
        #return len(self.classification_frame)
        return len(self.data_paths["images"])

    def get_resize_op(self, pad_to_tile_size=False, check_shapes=False):
        if not self.variable_input_shapes:
            # Crop everything to specific image size if variable_input_shapes is off
            return A.Compose([A.CropOrPad(self.img_size, self.img_size, p=1.0)], is_check_shapes=check_shapes)

        # Pad images that are smaller than the tile size to the tile size
        # Also, if one dimensions is larger than the tile size, make sure it is a multiple of at least the network output
        # stride
        # todo: Figure out if second padding must be tile_stride
        if pad_to_tile_size:
            return A.Compose(
                [
                    A.PadIfNeeded(
                        min_width=self.tile_size,
                        min_height=self.tile_size,
                        value=[255, 255, 255],
                        mask_value=[0, 0, 0],
                    ),
                    A.PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_width_divisor=self.network_output_stride,
                        pad_height_divisor=self.network_output_stride,
                        value=[255, 255, 255],
                        mask_value=[0, 0, 0],
                    ),
                ],
                is_check_shapes=check_shapes,
            )

        # Images that are already larger than tile size should be padded to a multiple of tile_stride
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_width_divisor=self.tile_stride,
                    pad_height_divisor=self.tile_stride,
                    value=[255, 255, 255],
                    mask_value=[0, 0, 0],
                ),
            ],
            is_check_shapes=check_shapes,
        )



if __name__ == "__main__":
    root = Path("/data/pathology/projects/pathology-bigpicture-streamingclam")
    data_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images")
    mask_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks")
    csv_file = root / Path("streaming_experiments/camelyon/data_splits/train_0.csv")

    dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(csv_file),
        tile_size=1600,
        img_size=1600,
        transform=augmentations,
        mask_dir=mask_path,
        mask_suffix="_tissue",
        variable_input_shapes=False,
        tile_stride=680,
        filetype=".tif",
        read_level=5,
    )

    import matplotlib.pyplot as plt

    ds = iter(dataset)
    print("Creating dataset")
    sample, label, name = next(ds)
    print("retrieving image")
    image = sample["image"]
    mask = sample["mask"]
    print("image shape", image.shape)
    print("image dtype", image.dtype)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.show()
    print("mask shape", mask.shape)
    plt.imshow(mask.cpu().numpy())
    plt.show()
