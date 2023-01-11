from torch.utils.tensorboard import SummaryWriter

from configs import experiment_name, log_path

neptune_project_name = "mherkhachatryan/intelli-recon"
neptune_config = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MzM0ZThlNS1hZGUxLTRkOTQtYmQyYy1hYzEzM2U5MWUzODAifQ=="

tb_writer = SummaryWriter(log_path / f'runs/{experiment_name}')
