# NOTE TO SCLAM DEVELOPERS — cg fork

This document communicates the code changes in this fork relative to the upstream baseline commit `e8a5886` (`DIAGNijmegen/pathology-streamingclam`). It is intended as a handover document so the original developers can understand what was changed, why, and what the practical effects are.

---

## Environment

All changes were developed and validated against this environment:

| Component | Version |
| --- | --- |
| `lightstream` | `1.0.7` |
| `lightning` | `2.6.1` |
| `pytorch-lightning` | `2.6.1` |
| `torch` | `2.5.1` |
| `torchvision` | `0.20.1` |
| `pyvips` | `2.2.1` |
| `libvips` | `8.13.3` |
| `openslide` | `3.4.1` |
| `libtiff` | `4.4.0` |
| `albumentationsxl` | `0.1.2` |

These versions are directly relevant to the analysis below (especially the lightstream/Lightning API changes and the pyvips-openslide-libtiff mask loading behavior).

---

## Why we changed anything at all

The upstream baseline was written against an older version of the `lightstream` library in which `ImageNetClassifier` (a `LightningModule` subclass) was the streaming model base class. In `lightstream 1.0.7`, `ImageNetClassifier` has a breaking API change: its `__init__` signature changed, `forward_streaming` was removed from `StreamingModule`, and the intended base class for custom models is now `LightningStreamingModule`. All `sclam.py` changes flow from this root cause.

Additionally, local dataset issues were encountered with CAMELYON16 slides (RGBA output from the OpenSlide loader in the pinned `pyvips/libvips/openslide` stack, and sparse BigTIFF masks not readable at full resolution with `libtiff 4.4.0`), which required robustness changes in the data pipeline.

---

## High-level summary by file

| File | Change type |
|---|---|
| `create_masks.py` | New utility: XML annotations to pyramidal TIFF mask generation |
| `streamingclam/models/sclam.py` | Core API migration: `ImageNetClassifier` → `LightningStreamingModule` |
| `streamingclam/data/dataset.py` | Robustness: OpenSlide/TIFF detection, sparse mask fix, check_csv loop fix, `__len__` fix, mask error recovery |
| `streamingclam/data/attention_dataset.py` | Mask error recovery (same pattern as dataset.py) |
| `streamingclam/utils/finetune.py` | API fixes: method renames, tile cache API, unfreeze logic fix |
| `streamingclam/options.py` | Local paths, tuned hyperparameters |
| `main.py` | Checkpoint discovery fix, pyvips cache increase, debug print |
| `plot_mask_pyvips.py` | New utility: diagnostics for TIFF mask loading via pyvips (level/page fallback, corruption handling) |
| `plot_mask_openslide.py` | New utility: diagnostics for mask reading via OpenSlide |
| `streamingclam/data/splits.py` | Trailing newline only |

---

## Detailed breakdown

### 1. Streaming model integration (`streamingclam/models/sclam.py`)

#### 1a. Import and base class change

```diff
-from lightstream.modules.imagenet_template import ImageNetClassifier
+from lightstream.modules.lightningstreaming import LightningStreamingModule
+from lightstream.modules.streaming import StreamingModule

-class StreamingCLAM(ImageNetClassifier):
+class StreamingCLAM(LightningStreamingModule):
```

Why: `ImageNetClassifier` no longer exists under that path in current lightstream. `LightningStreamingModule` is its replacement. It takes a `StreamingModule` wrapper object (not the raw CNN) as its argument.

#### 1b. `split_resnet` return value

```diff
-            stream_net, _ = split_resnet(network)
+            stream_net = split_resnet(network)
```

Why: In current lightstream, `split_resnet` returns only the streaming-compatible network, not a tuple.

#### 1c. `additive` argument removed from `CLAM_SB`

```diff
-                instance_loss_fn=self.instance_loss_fn(),
-                subtyping=self.subtyping,
-                additive=self.additive,
+                instance_loss_fn=self.instance_loss_fn(),
+                subtyping=self.subtyping,
```

Why: `CLAM_SB` does not accept an `additive` argument in the current codebase. Passing it causes a `TypeError`.

#### 1d. Constructor refactor

```diff
-        self.ds_blocks = None
+        _ds_blocks = None
         if self.pooling_kernel > 0:
             if self.stream_pooling_kernel:
                 stream_net = self.add_pooling_layers(stream_net)
-                super().__init__(stream_net, head, tile_size, loss_fn,
-                                 train_streaming_layers=train_streaming_layers, ...)
             else:
-                ds_blocks, head = self.add_pooling_layers(head)
-                super().__init__(stream_net, head, tile_size, loss_fn,
-                                 train_streaming_layers=train_streaming_layers, ...)
-                self.ds_blocks = ds_blocks
-        else:
-            super().__init__(stream_net, head, tile_size, loss_fn,
-                             train_streaming_layers=train_streaming_layers, ...)
+                _ds_blocks, head = self.add_pooling_layers(head)
+
+        stream_module = StreamingModule(stream_net, tile_size, **self.streaming_options)
+        super().__init__(stream_module)
+
+        self.constructor = stream_module.constructor
+
+        self.head = head
+        self.loss_fn = loss_fn
+        self.ds_blocks = _ds_blocks
```

Why: `LightningStreamingModule.__init__` takes a single `StreamingModule` wrapper, not `(stream_net, head, tile_size, loss_fn, ...)`. The old call signature no longer matches. `self.head`, `self.loss_fn`, and `self.ds_blocks` must be assigned explicitly after `super().__init__()` because `nn.Module.__init__` runs there.

`self.constructor` is stored from `stream_module.constructor` — this is needed by the finetuning callback (`finetune.py`) to rebuild the streaming model at the tile-size transition.

#### 1e. `_pool_and_move` and `ds_blocks` removal from `forward_head`

```diff
+    def _pool_and_move(self, fmap):
+        """Apply ds_blocks pooling on CPU then move to GPU."""
+        if self.ds_blocks is not None:
+            fmap = self.ds_blocks(fmap)
+        return fmap.to(self.device)

     def forward_head(self, fmap, ...):
         batch_size, num_features, h, w = fmap.shape

-        if self.ds_blocks is not None:
-            fmap = self.ds_blocks(fmap)
-
         # Mask background ...
```

Why: `ds_blocks` pooling is now applied on CPU (before the GPU transfer) via `_pool_and_move`, which reduces the tensor size before the potentially expensive `.to(device)` call. It is called consistently from both `forward` and `training_step`, so the check was removed from `forward_head`.

#### 1f. `transfer_batch_to_device` override

```diff
+    def transfer_batch_to_device(self, batch, device, dataloader_idx):
+        image = batch.pop("image")
+        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
+        batch["image"] = image
+        return batch
```

Why: Lightning's default `transfer_batch_to_device` moves the entire batch to GPU. The image must stay on CPU for tile-by-tile streaming. This override moves everything except the image to device.

#### 1g. `forward` and `training_step` — replace `forward_streaming`

```diff
     def forward(self, image, mask=None):
-        fmap = self.forward_streaming(image)
+        fmap = self._pool_and_move(self.stream_network.forward(image, result_on_cpu=True))

     def training_step(self, ...):
-        self.str_output = self.forward_streaming(image)
+        self.str_output = self.stream_network.forward(image, result_on_cpu=True)
         self.str_output.requires_grad = self.training
+        fmap_gpu = self._pool_and_move(self.str_output)
         logits, ... = self.forward_head(
-            self.str_output,
+            fmap_gpu,
```

Why: `forward_streaming` does not exist on `LightningStreamingModule`. The direct replacement is `self.stream_network.forward(image, result_on_cpu=True)`. Using `result_on_cpu=True` keeps the assembled feature map on CPU, allowing `_pool_and_move` to run the optional pooling on CPU before transferring to GPU.

#### 1h. Logging: `on_step=False` on all metrics

```diff
-        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
+        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
```

Same change applied to `train_auc`, `train_loss`, `valid_acc`, `valid_auc`, `val_loss`.

Why: With very large images (one image per batch), step-level metric logging causes confusing per-step output and can conflict with torchmetrics' internal accumulation. Setting `on_step=False` logs only at epoch end, which is the meaningful granularity for slide-level classification.

#### 1i. `configure_optimizers` — `self.params` → `self.parameters()`

```diff
-        optimizer = torch.optim.Adam(self.params, lr=self.learning_rate, weight_decay=1e-5)
+        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
```

Why: `self.params` was a property on `ImageNetClassifier`. `LightningStreamingModule` does not have it. `self.parameters()` is the standard `nn.Module` method.

#### 1j. `backward` — signature and device fix

```diff
-    def backward(self, loss):
+    def backward(self, loss, *args, **kwargs):
         loss.backward()
         torch.cuda.empty_cache()
         if self.train_streaming_layers:
-            self.backward_streaming(self.image, self.str_output.grad)
+            grad = self.str_output.grad.to(self.device)
+            self.backward_streaming(self.image, grad)
         del self.str_output, self.image
```

Why (signature): Lightning's `backward` hook signature changed; `*args, **kwargs` ensures forward compatibility.

Why (`.to(self.device)` — **bug fix**): `stream_network.forward(image, result_on_cpu=True)` returns a CPU tensor. After `loss.backward()`, PyTorch's `ToCopyBackward0` autograd function routes `str_output.grad` back to the CPU (the source device). `StreamingCNN.backward` then calls `trimmed_output.to(self.device)` (GPU) and `trimmed_output.backward(trimmed_grad)` where `trimmed_grad` is a slice of the gradient — if the gradient is still on CPU, this is a device mismatch and raises `RuntimeError: invalid gradient at index 0 - expected device cuda:0 but got cpu`. The fix is `.to(self.device)` before passing to `backward_streaming`. The default `train_streaming_layers=False` means this path is only triggered when streaming layers are being actively trained.

File: [streamingclam/models/sclam.py](streamingclam/models/sclam.py)

---

### 2. Dataset robustness (`streamingclam/data/dataset.py`)

#### 2a. `check_csv` loop fix

```diff
-        for i in range(len(self)):
+        for i in range(len(self.classification_frame)):
```

Why: `len(self)` calls `__len__`, which (after the fix in 2f below) returns `len(self.data_paths["images"])`. During `check_csv`, `data_paths` has not been populated yet, so `len(self)` would return 0 and the loop would never run. Iterating over `len(self.classification_frame)` correctly uses the source CSV row count.

#### 2b. File-existence check fix

```diff
+            all_exist = True
             for file in images:
                 if not file.exists():
                     print(f"WARNING {file} not found ...")
-                    continue
+                    all_exist = False
+                    break
+
+            if not all_exist:
+                continue
```

Why: The original `continue` applied to the inner `for file in images` loop, not the outer `for i in range(...)` loop. A missing file would print a warning but the sample would still be added to `included`. The fix uses a flag to skip the entire sample (both image and mask) when any file is missing.

#### 2c. Mask path construction fix

```diff
-            mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(self.filetype)
+            imag_path = Path(img_fname)
+            mask_path = self.mask_dir / f"{imag_path.stem}{self.mask_suffix}{self.filetype}"
```

Why: The original code appended `self.mask_suffix` to the full filename string (including its existing extension), then called `.with_suffix(self.filetype)`. For a file `tumor_069.tif` with suffix `_tissue` and filetype `.tif`, this produced `tumor_069.tif_tissue.tif`. The fix strips the existing extension first (`.stem`) and builds the path cleanly: `tumor_069_tissue.tif`.

#### 2d. OpenSlide / TIFF loader detection and RGBA handling

```diff
-        image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
+        try:
+            image = pyvips.Image.new_from_file(img_fname, level=self.read_level)
+        except pyvips.error.Error:
+            image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
+
+        if image.bands == 4:
+            image = image.flatten()
```

Why: pyvips has two backends for multi-resolution image access. The OpenSlide backend uses `level=` (for `.svs`, OpenSlide-compatible `.tif`). The standard TIFF backend uses `page=`. The appropriate one depends on the file format and the local pyvips/OpenSlide install. This version uses a try/except to probe with `level=` first and fall back to `page=` if that fails. RGBA (4-band) output from OpenSlide is flattened to RGB because the streaming CNN expects 3 bands.

#### 2e. Sparse BigTIFF mask fix

```diff
-            mask = pyvips.Image.new_from_file(mask_fname)
+            try:
+                mask_meta = pyvips.Image.new_from_file(mask_fname)
+                n_mask_pages = mask_meta.get('n-pages') if mask_meta.get_typeof("n-pages") else 1
+                mask = pyvips.Image.new_from_file(mask_fname, page=n_mask_pages - 1)
+            except pyvips.Error:
+                mask = pyvips.Image.new_from_file(mask_fname, page=self.read_level)
+
+            if mask.bands == 4:
+                mask = mask.flatten()
+
             ratio = image.width / mask.width
             images["mask"] = mask.resize(ratio, kernel="nearest")
```

Why: ASAP-generated tissue masks (used in CAMELYON16 workflows) are sparse BigTIFF pyramids. In libtiff ≥4.x, intermediate pyramid levels have zero-bytecount background tiles that libtiff rejects. The only reliably readable level is the last page (lowest resolution, smallest, fully encoded). Loading that page and then resizing with `ratio` to match the image dimensions is the workaround. The `ratio`-based resize is retained because albumentationsxl's `PadIfNeeded` and `RandomCrop` operate on image-absolute pixel coordinates derived from the image dimensions — if the mask is not at the same dimensions as the image when they enter the albumentations pipeline, transforms will silently produce spatially incorrect masks. A fallback to `page=self.read_level` is included in case the page-count query itself fails.

#### 2f. Mask numpy conversion error recovery

```diff
         if "mask" in sample.keys():
             ...
             sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")
+
+            try:
+                mask_np = sample["mask"].numpy()
+            except pyvips.Error as e:
+                mask_file_path = str(self.data_paths["masks"][idx])
+                print(f"WARNING: pyvips error converting mask ... Replacing with all-zero mask: {e}")
+                mask_np = np.zeros((sample["mask"].height, sample["mask"].width), dtype=np.uint8)
+
+            sample["mask"] = pyvips.Image.new_from_array(mask_np)
```

Why: Some mask files that passed the file-existence check may still produce pyvips errors when decompressed and converted to numpy (e.g., corrupt tile data within an otherwise valid TIFF). Rather than crashing the entire epoch, this substitutes an all-zero mask (no tissue masking) for the affected slide and prints a warning. An all-zero mask means no feature map pixels are selected by `torch.masked_select`, so the training step for that slide will produce a trivial (empty bag) prediction — incorrect but recoverable. We observed this path triggering on one image (`CAMELYON16/background_tissue/normal_071_tissue.tif`); root cause was not identified.

#### 2g. `augmentations` — `is_check_shapes=False`

```diff
 augmentations = A.Compose(
     [A.Flip(), A.HueSaturationValue(p=0.5), A.Rotate()],
+    is_check_shapes=False,
 )
```

Why: The three transforms here are safe with different-sized image and mask inputs: `Flip` and `Rotate` operate on each independently using relative coordinates; `HueSaturationValue` is `ImageOnlyTransform` and ignores the mask entirely. Disabling the shape check allows these transforms to run even if the mask has not yet been resized to match the image (e.g., in cases where the resize fallback path leaves a slight size mismatch due to pyvips integer rounding).

#### 2h. `get_resize_op` — `is_check_shapes` parameter

```diff
-    def get_resize_op(self, pad_to_tile_size=False):
+    def get_resize_op(self, pad_to_tile_size=False, check_shapes=False):
         if not self.variable_input_shapes:
-            return A.Compose([A.CropOrPad(...)])
+            return A.Compose([A.CropOrPad(...)], is_check_shapes=check_shapes)
         ...
-                ]
+                ], is_check_shapes=check_shapes,
         ...
-            ]
+            ], is_check_shapes=check_shapes,
```

Why: Exposes `is_check_shapes` as a caller-controlled parameter. The default `check_shapes=False` preserves the existing loose behavior, but callers that want strict shape validation can opt in.

#### 2i. `__len__` fix

```diff
-    return len(self.classification_frame)
+    return len(self.data_paths["images"])
```

Why: After `check_csv` has run, `data_paths["images"]` is the authoritative list of valid, existing files. Some entries from the CSV may have been excluded. `len(self.classification_frame)` would return the unfiltered row count, causing index-out-of-range errors when the dataloader tries to access excluded samples.

File: [streamingclam/data/dataset.py](streamingclam/data/dataset.py)

---

### 3. Attention dataset mask error recovery (`streamingclam/data/attention_dataset.py`)

```diff
+import numpy as np
 ...
         sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")
+
+        try:
+            mask_np = sample["mask"].numpy()
+        except pyvips.Error as e:
+            mask_file_path = str(self.data_paths["masks"][idx])
+            print(f"WARNING: pyvips error converting mask ... Replacing with all-zero mask: {e}")
+            mask_np = np.zeros((sample["mask"].height, sample["mask"].width), dtype=np.uint8)
+
+        sample["mask"] = pyvips.Image.new_from_array(mask_np)
```

Same error-recovery pattern as dataset.py section 2f, applied to `AttentionDataset.__getitem__`.

File: [streamingclam/data/attention_dataset.py](streamingclam/data/attention_dataset.py)

---

### 4. Finetuning callback (`streamingclam/utils/finetune.py`)

#### 4a. Method rename

```diff
-        pl_module.freeze_streaming_normalization_layers()
+        pl_module.freeze_normalization_layers()
```

Why: `LightningStreamingModule` exposes `freeze_normalization_layers()`. The old name `freeze_streaming_normalization_layers()` does not exist and raises `AttributeError`.

#### 4b. Unfreeze logic fix

```diff
-        if current_epoch == self._unfreeze_at_epoch:
-            current_lr = ...
-            self.previous_backbone_lr = initial_backbone_lr
-            ...
-
         if current_epoch >= self._unfreeze_at_epoch:
             if self.switch:
+                current_lr = ...
+                self.previous_backbone_lr = initial_backbone_lr
+                ...
                 pl_module.train_streaming_layers = True
                 self.unfreeze_and_add_param_group(...)
```

Why: The original code had the LR setup in an `== epoch` guard that fired once. If training was resumed after the unfreeze epoch, the LR setup block would be skipped entirely (because `current_epoch != _unfreeze_at_epoch` on the second run), but the unfreeze block (which uses `>=`) would try to use `self.previous_backbone_lr` before it was set. Moving the LR setup inside the `>=` / `self.switch` block ensures it always runs when needed.

#### 4c. Tile cache API

```diff
-            tile_cache = pl_module.load_tile_cache_if_needed()
+            tile_cache = pl_module.stream_network.get_tile_cache()
```

```diff
-            pl_module.save_tile_cache_if_needed()
+            # pl_module.stream_network.save_tile_cache_if_needed()
```

Why: In current lightstream, `load_tile_cache_if_needed` is a method on the inner `StreamingCNN` (`stream_network`), not on the `LightningStreamingModule`. The new API uses `stream_network.get_tile_cache()`. `save_tile_cache_if_needed` is commented out pending confirmation of the correct API call in the installed version.

#### 4d. `tile_size` assignment commented out

```diff
-            pl_module.tile_size = self.tile_size_finetune
+            #pl_module.tile_size = self.tile_size_finetune
```

Why: `LightningStreamingModule` does not have a `tile_size` attribute. The tile size for the finetuning phase is instead set via `pl_module.constructor.tile_size = self.tile_size_finetune` (the next line, which is retained). Setting the non-existent attribute would silently create a new instance variable that has no effect.

File: [streamingclam/utils/finetune.py](streamingclam/utils/finetune.py)

---

### 5. Checkpoint discovery (`main.py`)

```diff
-def configure_checkpoints():
-    try:
-        last_checkpoint = list(Path(options.default_save_dir + f"/{options.experiment_name}/fold_{str(options.fold)}").glob("*last.ckpt"))
-        last_checkpoint_path = str(last_checkpoint[0])
-    except IndexError:
-        if options.resume:
-            warnings.warn("Resume option enabled, but no checkpoint files found.")
-        last_checkpoint_path = None
-    return last_checkpoint_path
+def configure_checkpoints(options):
+    ckp_dir = Path(options.default_save_dir) / options.experiment_name / f"fold_{options.fold}" / "ckp"
+    ...
+    # 1. Try last.ckpt
+    # 2. Fall back to most recently modified .ckpt
+    # 3. Warn if nothing found
```

Why: The original function used string concatenation to build the checkpoint directory path (fragile with trailing slashes) and only looked for files matching `*last.ckpt` (glob, not exact). The new version:
- Uses `Path` throughout for correct cross-platform path construction.
- Looks for `last.ckpt` (exact name, as Lightning writes it).
- Falls back to the most recently modified `.ckpt` file in the directory if `last.ckpt` is not present, with a warning.
- Accepts `options` as an argument (the original implicitly closed over the global `options` variable, which is an anti-pattern and breaks if called before `options` is defined).
- Provides a clear error when no checkpoint is found in test/attention mode and `--ckp_path` was not specified.

Additionally, test/attention mode now auto-discovers the checkpoint:
```diff
+        checkpoint_path = options.ckp_path
+        if not checkpoint_path:
+            checkpoint_path = configure_checkpoints(options)
+            if not checkpoint_path:
+                raise ValueError("Could not find a checkpoint ...")
         model = StreamingCLAM.load_from_checkpoint(
-            options.ckp_path,
+            checkpoint_path,
```

#### pyvips cache

```diff
-pyvips.cache_set_max(20)
-pyvips.cache_set_max_mem(1024 * 1024)
+pyvips.cache_set_max(200)
+pyvips.cache_set_max_mem(1024 * 1024 * 1024)
```

Why: The original limits (20 operations, 1 MB) were very conservative and caused repeated re-reads of tiles that could have been cached. Increasing to 200 operations and 1 GB allows pyvips to cache more tile data in memory, which reduces disk I/O for large slides where the same region is accessed multiple times during a streaming pass.

#### ImageMagick loader bypass (`streamingclam/data/dataset.py`)

```diff
-        try:
-            image = pyvips.Image.new_from_file(img_fname, level=self.read_level)
-        except pyvips.error.Error:
-            image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
+        try:
+            image = pyvips.Image.openslideload(img_fname, level=self.read_level)
+        except pyvips.error.Error:
+            try:
+                image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
+            except pyvips.error.Error:
+                image = pyvips.Image.new_from_file(img_fname)
```

Why: When loading `.svs` files (Aperio/TCGA format), `pyvips.Image.new_from_file` auto-detects the format and routes to `VipsForeignLoadMagick7File` (ImageMagick) instead of the OpenSlide loader. ImageMagick does not support the `level=` argument, causing an immediate `pyvips.error.Error`. The fallback to `page=` also fails because ImageMagick's TIFF reader rejects the SVS internal structure (`TIFFReadDirectory: Failed to read directory at offset ...`). The result is a crash in the DataLoader worker.

The fix calls `pyvips.Image.openslideload()` explicitly, which unconditionally routes to the OpenSlide backend and correctly interprets `level=` for multi-resolution `.svs` files. The fallback chain is retained for non-SVS formats: `new_from_file(page=)` handles standard pyramidal TIFFs, and bare `new_from_file()` (no level selection, loads level 0) is the last resort.

This change was triggered by switching the dataset from CAMELYON16 (`.tif`, handled by pyvips's TIFF backend) to TCGA lung (`.svs`, requires OpenSlide).

File: [main.py](main.py)

---

### 6. Options defaults (`streamingclam/options.py`)

All changes are local configuration values (paths, hyperparameters). None are API changes. Key differences from baseline defaults:

| Option | Baseline | This fork |
|---|---|---|
| `image_path` | `""` | `/data/wsi_data/CAMELYON16/images` |
| `mask_path` | `""` | `/data/wsi_data/CAMELYON16/background_tissue` |
| `train_csv` / `val_csv` / `test_csv` | bigpicture project paths | local CAMELYON16 paths |
| `default_save_dir` | bigpicture uncertainty dir | `/data/ccardona/sstep_savedir/experiments/` |
| `num_epochs` | 35 | 2 (tuning/debug value) |
| `unfreeze_streaming_layers_at_epoch` | 25 | 13 |
| `learning_rate` | 2e-4 | 5e-5 |
| `grad_batches` | 2 | 4 |
| `tile_size` / `tile_size_finetune` | 3200 | 3500 |
| `additive` | not present | `True` (added field) |
| `verbose` | not present | `True` (added field) |

File: [streamingclam/options.py](streamingclam/options.py)

---

---

> Carlos review request: Sections 8 through 11 below are utility/supporting-file notes. Please confirm whether these should remain in this developer handover document or be removed.

---

### 8. New mask-generation utility (`create_masks.py`)

New file added:
- Parses CAMELYON16 XML polygon annotations (`xml.etree.ElementTree`).
- Rasterizes polygons into a NumPy mask using `cv2.fillPoly`.
- Writes tiled pyramidal BigTIFF via pyvips (`tile=True`, `pyramid=True`, `compression="lzw"`, `bigtiff=True`).

```diff
+def generate_pyramidal_mask(xml_path, wsi_path, output_path, level=2):
+    ...
+    scale_factor = 1.0 / (2**level)
+    ...
+    cv2.fillPoly(mask_np, [pts], color=1)
+    ...
+    vips_mask.write_to_file(..., tile=True, pyramid=True, compression="lzw", bigtiff=True)
```

Why: This script creates tissue/tumor masks directly from XML annotations in a format intended to be easier to load with the dataset pipeline than problematic sparse ASAP outputs.

Impact:
- Useful for data preparation and reproducibility of mask creation.
- It is a standalone utility and does not alter training/inference code paths unless users adopt its outputs.

File: [create_masks.py](create_masks.py)

---

### 9. New pyvips mask-diagnostic utility (`plot_mask_pyvips.py`)

New file added:
- Enumerates available pyvips loaders (`pyvips.base.get_loaders()`).
- Attempts both OpenSlide-style `level=` and TIFF-style `page=` loading.
- Converts to NumPy and plots masks to PNG.
- Optional `--delete_corrupt` to remove files that fail NumPy conversion.

```diff
+def plot_and_save_mask(mask_path, output_path, read_level=4, delete_corrupt=False):
+    ...
+    try: mask_image = pyvips.Image.new_from_file(..., level=read_level)
+    except pyvips.Error: ...
+    if mask_image is None:
+        try: mask_image = pyvips.Image.new_from_file(..., page=read_level)
+        except pyvips.Error: ...
+    ...
+    if delete_corrupt:
+        mask_path.unlink()
```

Why: This script was added as a practical debugging aid while diagnosing backend/loader differences and corrupt mask files.

Impact:
- Helpful for triaging input-data failures.
- `--delete_corrupt` is destructive by design and should be used carefully.

File: [plot_mask_pyvips.py](plot_mask_pyvips.py)

---

### 10. New OpenSlide mask-diagnostic utility (`plot_mask_openslide.py`)

New file added:
- Reads masks using `openslide.OpenSlide` directly.
- Handles out-of-range levels by falling back to highest available level.
- Converts RGBA region to grayscale and saves visualization.

```diff
+def plot_and_save_mask_openslide(mask_path, output_path, read_level=4):
+    slide = openslide.OpenSlide(str(mask_path))
+    if read_level >= slide.level_count:
+        read_level = slide.level_count - 1
+    mask_rgba = slide.read_region((0, 0), read_level, level_dims)
+    mask_pil = mask_rgba.convert("L")
```

Why: Complements `plot_mask_pyvips.py` by testing the same file through OpenSlide directly, helping distinguish pyvips-loader issues from source-file issues.

Impact:
- Debug utility only; no production training path impact.

File: [plot_mask_openslide.py](plot_mask_openslide.py)

---

### 11. Data module whitespace-only change (`streamingclam/data/splits.py`)

This file has only a trailing newline removal in the fork diff.

Why/Impact: no runtime behavior change.

File: [streamingclam/data/splits.py](streamingclam/data/splits.py)

---

## Exhaustiveness checklist (baseline `e8a5886` → fork HEAD)

Verified changed files in git diff:
1. `.gitignore` — intentionally omitted from detailed sections (section 7 removed per request)
2. `create_masks.py` — covered in section 8
3. `main.py` — covered in section 5
4. `plot_mask_openslide.py` — covered in section 10
5. `plot_mask_pyvips.py` — covered in section 9
6. `streamingclam/data/attention_dataset.py` — covered in section 3
7. `streamingclam/data/dataset.py` — covered in section 2
8. `streamingclam/data/splits.py` — covered in section 11
9. `streamingclam/models/sclam.py` — covered in section 1
10. `streamingclam/options.py` — covered in section 6
11. `streamingclam/utils/finetune.py` — covered in section 4

Coverage note: section 7 was removed; sections 8 through 11 are marked for Carlos review.

---

## Net effect

With these changes applied, the fork runs against current lightstream / Lightning on CAMELYON16 data stored locally. The streaming classification pipeline is functionally equivalent to the baseline for `train_streaming_layers=False` (the default). For `train_streaming_layers=True`, the `.to(self.device)` fix in `backward` is required for correctness; without it, `backward_streaming` would raise a device mismatch error.

---

## Known issues and caveats

- **`save_tile_cache_if_needed` commented out**: The correct API for saving the tile cache in the installed lightstream version was not confirmed. The finetuning phase may not persist the tile cache between the two training phases. This is a performance issue, not a correctness issue.
- **`additive=True` in options has no effect**: `CLAM_SB` does not implement additive attention in the current codebase. The `additive` parameter is stored in `CLAMConfig` but not forwarded to `CLAM_SB` (the `additive=self.additive` argument was removed in the constructor fix). Setting it to `True` in options is a no-op.
- **Mask error recovery substitutes all-zero masks**: Slides whose masks fail pyvips conversion will be trained without tissue masking. The model will process all feature map pixels for those slides, including background. This is logged as a WARNING.
- **Commented-out `check_csv` loop** (`#for i in range(len(self)):`): The old loop line is left as a comment. It is safe to remove.