import fire
import pathlib
import random
import numpy as np
import torch

from epsilon_transformers.training.configs import TrainConfig

# TODO: Generalize train_model so that it doesn't depend on the HookedTransformer internal loss function
# TODO: move _check_if_action_batch asserts to a config validator

# TODO: Add eval part of the code
# TODO: Couple eval, logging & saving frequencies (??)
# TODO: Test _check_if_action_batch()
# TODO: Review best practices regarding seed setting
# TODO: Add Wandb Logging
# TODO: Test on GPUs
# TODO: Implement to_hooked_transformer()
# TODO: Implement save_model()
# TODO: Add DP

def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _check_if_action_batch(perform_action_every_n_tokens: int, total_tokens: int, batch_size: int, sequence_len: int, batch_idx: int) -> bool:

    tokens_per_batch = (batch_size * sequence_len)
    assert perform_action_every_n_tokens > tokens_per_batch, "perform_action_every_n_tokens must be greater than tokens_per_batch"
    total_batches = total_tokens // tokens_per_batch
    perform_action_every_n_batches = perform_action_every_n_tokens // tokens_per_batch
    action_interval = total_batches // perform_action_every_n_batches
    return (batch_idx + 1) % action_interval == 0

def _main(config_path: pathlib.Path):
    config: TrainConfig = TrainConfig.from_yaml(config_path)
    train_model(config)

def train_model(config: TrainConfig):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    _set_random_seed(config.seed)

    model = config.model.to_hooked_transformer(device=device, seed=config.seed)
    optimizer = config.optimizer.from_model(model=model, device=device)
    dataloader = config.dataset.to_dataloader(sequence_length=model.cfg.n_ctx)
    
    model.train()
    for batch_idx, (input_data, target_data) in enumerate(dataloader):
        input_data, target_data = input_data.to(device), target_data.to(device)

        loss = model(input_data, return_type="loss")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: Logging
        
        if _check_if_action_batch(perform_action_every_n_tokens=config.persistance.checkpoint_every_n_tokens, total_tokens=config.dataset.num_tokens, batch_size=config.dataset.batch_size, batch_idx=batch_idx, sequence_len=config.model.n_ctx):
            config.persistance.save_model()


if __name__ == "__main__":
    fire.Fire(_main)