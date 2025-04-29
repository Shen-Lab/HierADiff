import wandb
import os

from tensorboardX import SummaryWriter


class Logger:
    """Abstract class for logger."""

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        """Logs scalar value."""
        raise Exception("Not implemented for this logger.")

# [nm]
class WandbLogger(Logger):
    """Weight and Biases logger."""

    def __init__(self, project: str, entity: str, name: str = None, group: str = None, config=None, notes: str = None, idfolder: str = None):
        """
        notes: A longer description of the run. helps you remember what you were doing when you
            ran this run.
        """

        
        try:

            # [ONLINE]
            # store this id to use it later when resuming
            idpath = idfolder + "/id.txt"
            if os.path.exists(idpath):
                with open(idpath, "r") as f:
                    id = f.read()
                resume = "allow" # no overriding of the run
            else:
                id = wandb.util.generate_id()

                # create folder if it doesn't exist & compatible with all other os.makedirs(, exist_ok=True) functions
                if not os.path.exists(idfolder):
                    os.makedirs(idfolder, exist_ok=True)

                with open(idpath, "w") as f:
                    f.write(id)
                resume = "allow" # no overriding of the run

            wandb.init(id=id, project=project, entity=entity, name=name, group=group, config=config, notes=notes, resume=resume)
        except:
            # # [OFFLINE]
            wandb.init(project=project, entity=entity, name=name, group=group, config=config, notes=notes, mode="offline")

    def add_scalar(self, tag: str, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step)


class TensorboardLogger(Logger):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)