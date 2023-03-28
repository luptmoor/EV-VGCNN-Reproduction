import numpy as np
import torch

# I stole this from another course and need to fix it

def load_dataloader(_filepath, _rng: random.KeyArray, _val_ratio=0.3, batch_size=64,):
    assert 0.0 <= _val_ratio <= 1.0, "Validation ratio needs to be in interval [0, 1]."

    _dataset = np.load(_filepath)
    num_samples = _dataset["th_curr_ss"].shape[0]

    indices = np.arange(num_samples)
    shuffled_indices = random.permutation(_rng, indices)
    num_train_samples = int((1 - _val_ratio) * num_samples)
    split_config = np.array(
        [
            num_train_samples,
        ]
    )
    train_indices, val_indices = np.split(shuffled_indices, split_config)

    _train_ds, _val_ds = {}, {}
    for key, val in _dataset.items():
        _train_ds[key] = val[train_indices]
        _val_ds[key] = val[val_indices]

    train_data = ThetaDataset(_train_ds)
    val_data = ThetaDataset(_val_ds)

    _train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    _val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=True
    )

    return _train_dataloader, _val_dataloader

