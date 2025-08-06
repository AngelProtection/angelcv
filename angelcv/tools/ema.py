from copy import deepcopy

from ema_pytorch import EMA
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from angelcv.utils.logging_manager import get_logger

logger = get_logger(__name__)


class EMACallback(Callback):
    """
    Applies Exponential Moving Average to a model's weights.

    This callback integrates with PyTorch Lightning to maintain an EMA of the model's
    parameters. The EMA model is used for evaluation (validation, testing) and
    inference, which can lead to improved performance and more stable training.

    The swapping of weights between the original model and the EMA model is handled
    automatically at the beginning and end of evaluation and inference phases.

    Args:
        decay (float): The decay factor for the EMA. A higher value means the EMA
                       is slower to change. Defaults to 0.9999.
        update_every (int): Update the EMA model every `update_every` steps. This
                            can save computational resources. Defaults to 1.
        update_after_step (int): Start updating the EMA model only after this
                                 many steps. Defaults to 0.
    """

    def __init__(self, decay: float = 0.9999, update_every: int = 1, update_after_step: int = 0):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.ema_model = None
        self._ema_copy = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Initializes the EMA model when training starts.
        """
        self.ema_model = EMA(
            pl_module,
            beta=self.decay,
            update_every=self.update_every,
            update_after_step=self.update_after_step,
        )
        self.ema_model.to(pl_module.device)
        logger.info("EMA callback initialized.")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """
        Updates the EMA model after each training batch.
        """
        if self.ema_model is not None:
            # EMA regular update - helps model to generalize encouraing flat minima,
            # but it may hurt the final performance and stagnate optimization
            self.ema_model.update()

    # NOTE: Used only for SEMA
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Swaps the model weights with the EMA weights after each epoch.
        """
        if self.ema_model is not None:
            # SEMA update: https://arxiv.org/pdf/2402.09240
            # After each epoch switch the original model weights to the EMA weights
            self.ema_model.update_model_with_ema()

    # NOTE: the code below was for EMA
    # def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the model weights with the EMA weights before validation.
    #     """
    #     self._swap_model_weights(pl_module)

    # def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the EMA weights back to the original model weights after validation.
    #     """
    #     self._swap_model_weights(pl_module, restore=True)

    # def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the model weights with the EMA weights before testing.
    #     """
    #     self._swap_model_weights(pl_module)

    # def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the EMA weights back to the original model weights after testing.
    #     """
    #     self._swap_model_weights(pl_module, restore=True)

    # def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the model weights with the EMA weights before prediction.
    #     """
    #     self._swap_model_weights(pl_module)

    # def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     """
    #     Swaps the EMA weights back to the original model weights after prediction.
    #     """
    #     self._swap_model_weights(pl_module, restore=True)

    def _swap_model_weights(self, pl_module: pl.LightningModule, restore: bool = False):
        """
        Helper function to swap model weights with EMA weights.

        Args:
            pl_module (pl.LightningModule): The PyTorch Lightning module.
            restore (bool): If True, restores the original weights. Otherwise,
                            it swaps the EMA weights into the model.
        """
        if self.ema_model is None:
            return

        if not restore:
            logger.debug("Swapping model weights with EMA weights.")
            self._ema_copy = deepcopy(pl_module.state_dict())
            pl_module.load_state_dict(self.ema_model.ema_model.state_dict())
        else:
            logger.debug("Restoring original model weights.")
            if self._ema_copy is not None:
                pl_module.load_state_dict(self._ema_copy)
                self._ema_copy = None
            else:
                logger.warning("No original model weights to restore.")
