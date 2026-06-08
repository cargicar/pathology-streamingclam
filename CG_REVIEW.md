# CG Review: Bug-Risk Analysis (Items 1-7)

Scope:
- Repository: `sclam_carlos`
- Comparison base: `e8a5886`
- Goal: highlight whether changes are only compatibility/runtime fixes, or can change results.

---

## 1) Defaults now materially change training/eval behavior (not just "get running")

Impact: **High** (can change metrics and outputs without code-path bugs)

```diff
diff --git a/streamingclam/options.py b/streamingclam/options.py
@@
-    num_epochs: int = 35
+    num_epochs: int = 2
@@
-    grad_batches: int = 2
+    grad_batches: int = 4
@@
-    learning_rate: float = 2e-4
+    learning_rate: float = 5e-5
@@
-    tile_size: int = 3200
-    tile_size_finetune: int = 3200
+    tile_size: int = 3500
+    tile_size_finetune: int = 3500
@@
-    image_path: str = ""
-    mask_path: str = ""
+    image_path: str = "/data/wsi_data/CAMELYON16/images"
+    mask_path: str = "/data/wsi_data/CAMELYON16/background_tissue"
```

Why this matters:
- Any run that relies on defaults is now behaviorally different.
- This is a policy/config change, not purely a compatibility patch.

---

## 2) Checkpoint auto-fallback may evaluate a different model than intended

Impact: **High** (can silently change test/attention results)

```diff
diff --git a/main.py b/main.py
@@
+def configure_checkpoints(options):
+    ...
+    all_checkpoints = list(ckp_dir.glob("*.ckpt"))
+    if all_checkpoints:
+        latest_checkpoint = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
+        latest_checkpoint_path = str(latest_checkpoint)
+        warnings.warn(f"WARNING: 'last.ckpt' not found. Using the most recently modified checkpoint: {latest_checkpoint_path}")
+        return latest_checkpoint_path
@@
-        model = StreamingCLAM.load_from_checkpoint(
-            options.ckp_path,
+        checkpoint_path = options.ckp_path
+        if not checkpoint_path:
+            checkpoint_path = configure_checkpoints(options)
+        model = StreamingCLAM.load_from_checkpoint(
+            checkpoint_path,
```

Why this matters:
- If `ckp_path` is omitted, the selected checkpoint depends on filesystem mtime, not explicit experiment intent.

---

## 3) Mask conversion fallback rewrites failed masks to zeros

Impact: **High** (changes effective inputs; can affect attention and predictions)

```diff
diff --git a/streamingclam/data/dataset.py b/streamingclam/data/dataset.py
@@
+            try:
+                mask_np = sample["mask"].numpy()
+            except pyvips.Error as e:
+                ...
+                mask_np = np.zeros((sample["mask"].height, sample["mask"].width), dtype=np.uint8)
+            sample["mask"] = pyvips.Image.new_from_array(mask_np)
```

```diff
diff --git a/streamingclam/data/attention_dataset.py b/streamingclam/data/attention_dataset.py
@@
+            try:
+                mask_np = sample["mask"].numpy()
+            except pyvips.Error as e:
+                ...
+                mask_np = np.zeros((sample["mask"].height, sample["mask"].width), dtype=np.uint8)
+            sample["mask"] = pyvips.Image.new_from_array(mask_np)
```

Why this matters:
- Corrupt-mask samples stop crashing, but become semantically different samples.
- This is robustness + behavior change, not pure runtime compatibility.

---

## 4) Shape checks disabled broadly in augmentation/tensor conversion paths

Impact: **Medium** (can allow silent mask/image misalignment)

```diff
diff --git a/streamingclam/data/dataset.py b/streamingclam/data/dataset.py
@@
 augmentations = A.Compose(
@@
-    ],
+    ],
+    is_check_shapes=False,
 )
@@
-    def get_resize_op(self, pad_to_tile_size=False):
+    def get_resize_op(self, pad_to_tile_size=False, check_shapes=False):
@@
-            return A.Compose([A.CropOrPad(self.img_size, self.img_size, p=1.0)])
+            return A.Compose([A.CropOrPad(self.img_size, self.img_size, p=1.0)], is_check_shapes=check_shapes)
@@
-        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
+        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
```

Why this matters:
- More tolerant pipelines can keep runs alive, but may suppress early detection of bad geometry.

---

## 5) Attention dataset recovery block references pyvips without importing it

Impact: **Medium** runtime risk (path-dependent)

```diff
diff --git a/streamingclam/data/attention_dataset.py b/streamingclam/data/attention_dataset.py
@@
+import numpy as np # Add this import
@@
+            except pyvips.Error as e:
+                ...
+            sample["mask"] = pyvips.Image.new_from_array(mask_np)
```

Observed issue:
- `pyvips` is used in new code but there is no `import pyvips` added in this file.

Why this matters:
- If that recovery branch is hit, it can raise `NameError`.

---

## 6) Output path join can ignore default_save_dir (pre-existing in this branch lineage)

Impact: **Medium** operational risk (wrong output location, harder reproducibility)

Current code:

```python
output_dir=Path(options.default_save_dir) / Path(f"/{options.experiment_name}/attentions")
```

Note on diff status:
- No explicit hunk in this fork diff changed this line relative to baseline.
- It is included here because it is a real behavior risk in current code and can affect where outputs are written.

Why this matters:
- Second path is absolute (`/...`), so it can discard `default_save_dir` when joined.

---

## 7) Logging semantics changed to epoch-only for train/val metrics

Impact: **Low** for model outputs, **Medium** for observability

```diff
diff --git a/streamingclam/models/sclam.py b/streamingclam/models/sclam.py
@@
-        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
+        self.log("train_loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
@@
-        self.log("valid_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
+        self.log("valid_acc", self.val_acc,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
```

Why this matters:
- Usually does not change learned parameters directly.
- Does change diagnostics granularity and may hide step-level anomalies.

---

## Bottom line

- Items **1, 2, 3** are the most likely to cause unintended result drift versus a strict "just get it running" objective.
- Items **4, 5, 6** are correctness/robustness risks that should be addressed to avoid silent failures or runtime surprises.
- Item **7** is mostly observability behavior.