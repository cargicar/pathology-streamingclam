import os

os.environ["WANDB_DIR"] = os.environ.get("WANDB_DIR", os.path.expanduser("~"))
os.environ["VIPS_CONCURRENCY"] = "30"
os.environ["OMP_NUM_THREADS"] = "4"
import pyvips

#pyvips.cache_set_max(20)
#pyvips.cache_set_max_mem(1024 * 1024)
pyvips.cache_set_max(200)
pyvips.cache_set_max_mem(1024 * 1024  * 1024)

import torch
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import roc_auc_score

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger, TensorBoardLogger

from streamingclam.options import TrainConfig
from streamingclam.utils.memory_format import MemoryFormat
from streamingclam.utils.printing import PrintingCallback
from streamingclam.utils.finetune import FeatureExtractorFreezeUnfreeze
from streamingclam.data.splits import StreamingCLAMDataModule
from streamingclam.data.dataset import augmentations
from streamingclam.models.sclam import StreamingCLAM
from streamingclam.utils.writers import AttentionWriter, TestPredictionWriter

torch.set_float32_matmul_precision("medium")


def configure_callbacks(options):
    callbacks = []
    if options.mode == "fit":
        checkpoint_callback = ModelCheckpoint(
            dirpath=options.default_save_dir + f"/{options.experiment_name}/fold_{options.fold}/ckp",
            monitor="val_loss",
            filename="streamingclam-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
            save_top_k=3,
            save_last=True,
            mode="min",
            verbose=True,
        )
        finetune_cb = FeatureExtractorFreezeUnfreeze(
            options.unfreeze_streaming_layers_at_epoch,
            tile_size_finetune=options.tile_size_finetune,
            lambda_func=lambda epoch: 5,
        )
        memory_format_cb = MemoryFormat()
        print_cb = PrintingCallback(options)

        callbacks = [checkpoint_callback, finetune_cb, memory_format_cb, print_cb]
    elif options.mode=="attention":
        writer_cb = AttentionWriter(Path(options.default_save_dir) / Path(f"{options.experiment_name}/attentions"),
                                    read_level=options.read_level,
                                    write_level=options.write_level,
                                    write_interval="batch" if options.mode=="attention" else "epoch")
        callbacks = [writer_cb]
    elif options.mode=="test":
        test_writer = TestPredictionWriter(Path(options.default_save_dir + f"/{options.experiment_name}/fold_{str(options.fold)}"))
        callbacks = [test_writer]
    return callbacks


def configure_logger(options):
    log_dir = str(options.default_save_dir)
    if options.logger_type == "wandb":
        return WandbLogger(
            name=options.experiment_name,
            project=options.wandb_project_name,
            save_dir=log_dir,
        )
    elif options.logger_type == "tensorboard":
        return TensorBoardLogger(save_dir=log_dir, name=options.experiment_name)
    elif options.logger_type == "csv":
        return CSVLogger(save_dir=log_dir, name=options.experiment_name)
    else:
        raise ValueError(f"Unknown logger_type '{options.logger_type}'. Choose from: wandb, tensorboard, csv")


def configure_trainer(options, logger=None):
    callbacks = configure_callbacks(options)
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu",
        max_epochs=options.num_epochs,
        devices=options.num_gpus,
        accumulate_grad_batches=options.grad_batches,
        precision=options.precision,
        callbacks=callbacks,
        strategy=options.strategy,
        benchmark=False,
        reload_dataloaders_every_n_epochs=options.unfreeze_streaming_layers_at_epoch,
        logger=logger,
    )
    return trainer


def get_model_statistics(model):
    """Prints model statistics for reference purposes

    Prints network output strides, and tile delta for streaming

    Parameters
    ----------
    model : pytorch lightning model object


    """

    tile_stride = model.configure_tile_stride()
    network_output_stride = model.stream_network.output_stride[1]
    return tile_stride, network_output_stride


def get_streaming_options(options):
    fields = [
        "statistics_on_cpu",
        "normalize_on_gpu",
        "copy_to_gpu",
        "verbose",
    ]
    opt_dict = options.to_dict()
    return {key: opt_dict[key] for key in fields}


def configure_checkpoints(options):
    ckp_dir = Path(options.default_save_dir) / options.experiment_name / f"fold_{options.fold}" / "ckp"
    print(f"INFO: Searching for checkpoints in: {ckp_dir}")
    if not ckp_dir.is_dir():
        if options.mode == 'fit' and options.resume:
            warnings.warn(f"Checkpoint directory {ckp_dir} not found. Training will start from scratch.")
        return None

    # 1. Try to find `last.ckpt`
    last_checkpoint_list = list(ckp_dir.glob("last.ckpt"))
    if last_checkpoint_list:
        last_checkpoint_path = str(last_checkpoint_list[0])
        print(f"Found last checkpoint file at {last_checkpoint_path}")
        return last_checkpoint_path

    # 2. If not found, find all .ckpt files and pick the latest one by modification time.
    all_checkpoints = list(ckp_dir.glob("*.ckpt"))
    if all_checkpoints:
        latest_checkpoint = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
        latest_checkpoint_path = str(latest_checkpoint)
        warnings.warn(f"WARNING: 'last.ckpt' not found. Using the most recently modified checkpoint: {latest_checkpoint_path}")
        return latest_checkpoint_path

    if options.mode == 'fit' and options.resume:
        warnings.warn(f"Resume option enabled, but no checkpoint files found in {ckp_dir}. Training will start from scratch.")
    return None


def configure_streamingclam(options, streaming_options):
    sclam_opts = {
        "encoder": options.encoder,
        "tile_size": options.tile_size,
        "loss_fn": options.loss_fn,
        "branch": options.branch,
        "n_classes": options.num_classes,
        "pooling_layer": options.pooling_layer,
        "pooling_kernel": options.pooling_kernel,
        "stream_pooling_kernel": options.stream_pooling_kernel,
        "train_streaming_layers": options.train_streaming_layers,
        "instance_eval": options.instance_eval,
        "return_features": options.return_features,
        "attention_only": options.attention_only,
        "unfreeze_at_epoch": options.unfreeze_streaming_layers_at_epoch,
        "learning_rate": options.learning_rate,
        "additive": options.additive,
        "write_attention": True
    }

    if options.mode == "fit":
        model = StreamingCLAM(
            **sclam_opts,
            **streaming_options,
        )
    else:
        checkpoint_path = options.ckp_path
        if not checkpoint_path:
            # If ckp_path is not provided, try to find the last checkpoint
            print("No ckp_path provided, trying to find last checkpoint...")
            checkpoint_path = configure_checkpoints(options)
            if not checkpoint_path:
                raise ValueError(
                    "Could not find a checkpoint to load for test/attention mode. "
                    "Please specify a --ckp_path or ensure a 'last.ckpt' file exists in the experiment directory."
                )
        model = StreamingCLAM.load_from_checkpoint(
            checkpoint_path,
            **sclam_opts,
            **streaming_options,
        )
    return model


def configure_datamodule(options):
    return StreamingCLAMDataModule(
        image_dir=options.image_path,
        level=options.read_level,
        tile_size=options.tile_size,
        tile_stride=options.tile_stride,
        network_output_stride=options.network_output_stride,
        train_csv_path=options.train_csv,
        val_csv_path=options.val_csv,
        test_csv_path=options.test_csv,
        attention_csv_path=options.attention_csv,
        tissue_mask_dir=options.mask_path,
        mask_suffix=options.mask_suffix,
        image_size=options.image_size,
        variable_input_shapes=options.variable_input_shapes,
        copy_to_gpu=options.copy_to_gpu,
        num_workers=options.num_workers,
        filetype=options.filetype,
        transform=augmentations if (options.use_augmentations and options.mode == "fit") else None,
        output_dir=Path(options.default_save_dir) / Path(f"/{options.experiment_name}/attentions")
    )


def compute_bootstrap_auc(y_true, y_score, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for AUC via slide-level resampling with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate samples that have only one class
        auc_scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    auc_point = roc_auc_score(y_true, y_score)
    auc_scores = np.array(auc_scores)
    ci_lower = float(np.percentile(auc_scores, 2.5))
    ci_upper = float(np.percentile(auc_scores, 97.5))
    return auc_point, ci_lower, ci_upper


def run_bootstrap_analysis(options):
    test_csv = (
        Path(options.default_save_dir)
        / options.experiment_name
        / f"fold_{options.fold}"
        / "test.csv"
    )
    if not test_csv.exists():
        print(f"Bootstrap skipped: {test_csv} not found.")
        return

    df = pd.read_csv(test_csv)
    y_true = df["label"].values
    # probs is stored as numpy repr e.g. "[0.3 0.7]" — take the positive-class column
    y_score = np.array(
        [np.fromstring(p.strip("[]"), sep=" ")[-1] for p in df["probs"].astype(str)]
    )

    auc, ci_low, ci_high = compute_bootstrap_auc(y_true, y_score)
    print("\n=== Bootstrap AUC (n=10,000, 95% CI) ===")
    print(f"  AUC:     {auc:.4f}")
    print(f"  95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Slides:  {len(df)}")


def get_options():
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))

    return options


if __name__ == "__main__":
    pl.seed_everything(1)

    options = get_options()
    print(f"[Rank {os.environ.get('RANK', 'N/A')}] DEBUG: After parsing, options.num_workers is: {options.num_workers}")
    streaming_options = get_streaming_options(options)

    model = configure_streamingclam(options, streaming_options)
    tile_stride, network_output_stride = get_model_statistics(model)
    options.tile_stride = tile_stride

    if options.stream_pooling_kernel:
        options.network_output_stride = network_output_stride
    else:
        options.network_output_stride = max(network_output_stride * options.pooling_kernel, network_output_stride)
    dm = configure_datamodule(options)
    dm.setup(stage=options.mode)

    if options.mode == "fit":
        logger = configure_logger(options)
        trainer = configure_trainer(options, logger)

        if options.logger_type == "wandb" and trainer.global_rank == 0:
            logger.experiment.config.update(options.to_dict())

        last_checkpoint_path = configure_checkpoints(options)
        # model.head = torch.compile(model.head)
        # model.stream_network.stream_module = torch.compile(model.stream_network.stream_module)
        # print(model.stream_network)

        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=last_checkpoint_path if (options.resume and last_checkpoint_path) else None,
        )

    elif options.mode=="attention" or options.mode=="test":
        trainer = configure_trainer(options)
        if options.mode=="attention":
            trainer.predict(model=model, datamodule=dm,)
        elif options.mode=="test":
            trainer.test(model=model, datamodule=dm,)
            if trainer.is_global_zero:
                run_bootstrap_analysis(options)



    else:
        raise ValueError("mode must be one of fit, test, attention, or predict, found {}".format(options.mode))

# DO:
