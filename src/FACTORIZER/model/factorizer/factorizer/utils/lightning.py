import torch
from torch import nn, optim
from pytorch_lightning.core import LightningModule

from .helpers import (
    wrap_class,
    all_gather,
    collate,
    decollate_channels,
    reduce_channels,
    remove_dublicates,
    reduce_batch,
)


class SemanticSegmentation(LightningModule):
    def __init__(
        self,
        network,
        loss=None,
        metrics=None,
        optimizer=optim.SGD,
        scheduler=None,
        scheduler_config=None,
        inferer=nn.Identity,
        **kwargs,
    ):
        super().__init__()

        # init network
        network = wrap_class(network)
        self.net = network()

        # loss
        self.loss = wrap_class(loss)

        # metric
        self.metrics = {k: wrap_class(v) for k, v in metrics.items()}

        # optimizar
        self.optimizer = wrap_class(optimizer)

        # learning rate scheduler
        if scheduler is not None:
            self.scheduler = wrap_class(scheduler)
            self.scheduler_config = (
                {} if scheduler_config is None else scheduler_config
            )

        # inference
        self.inferer = wrap_class(inferer)

        # Validation results (scores)
        self.val_results = None

        # save hyperparameters
        self.save_hyperparameters()

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.net(x)
        if not isinstance(out, torch.Tensor):
            out = out[0]

        return out

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters())
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler(optimizer)
            config = (
                [optimizer],
                [{"scheduler": scheduler, **self.scheduler_config}],
            )
        else:
            config = [optimizer]

        return config

    def predict_step(self, batch, batch_idx):
        y_hat = self.inferer(batch, self)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch["input"], batch["target"]

        # forward
        y_hat = self.net(x)

        # calculate loss
        loss = self.loss(y_hat, y)

        # add logging and calculate metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, id_ = batch["target"], batch["id"]

        # inference
        y_hat = self.predict_step(batch, batch_idx)

        # calculate metrics
        out = {"id": id_}
        for metric_name, metric in self.metrics.items():
            out[f"val_{metric_name}"] = metric(y_hat, y)

        return out

    def validation_epoch_end(self, val_outputs):
        metrics = [f"val_{m}" for m in self.metrics]
        # collate batches
        val_outputs = collate(val_outputs)

        # add average scores over channels
        reduced_channels = reduce_channels(val_outputs, metrics)
        val_outputs = {**val_outputs, **reduced_channels}

        # decollate channels
        val_outputs = decollate_channels(val_outputs, metrics)

        # gather tensors from all distributed processes
        self.val_results = all_gather(val_outputs)

        # drop duplicate ids
        self.val_results = remove_dublicates(self.val_results, "id")

        # average over batch
        avg_scores = reduce_batch(self.val_results)
        for key, value in avg_scores.items():
            self.log(key, value, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # inference
        y_hat = self.predict_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return None

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

