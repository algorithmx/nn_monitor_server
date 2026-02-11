import os
import struct
import numpy as np


def _read_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def load_numpy(raw_dir=None):
    """Return (x_train, y_train, x_test, y_test) as numpy arrays.

    If IDX files exist under `raw_dir` (default: `data/FashionMNIST/raw`) they
    will be used. Otherwise the function will attempt to fall back to
    `tensorflow.keras.datasets.fashion_mnist.load_data()`.
    """
    if raw_dir is None:
        raw_dir = os.path.join('data', 'FashionMNIST', 'raw')

    train_img_path = os.path.join(raw_dir, 'train-images-idx3-ubyte')
    train_lbl_path = os.path.join(raw_dir, 'train-labels-idx1-ubyte')
    test_img_path = os.path.join(raw_dir, 't10k-images-idx3-ubyte')
    test_lbl_path = os.path.join(raw_dir, 't10k-labels-idx1-ubyte')

    # If IDX files exist, read and normalize them
    if all(os.path.exists(p) for p in (train_img_path, train_lbl_path, test_img_path, test_lbl_path)):
        x_train = _read_idx_images(train_img_path)
        y_train = _read_idx_labels(train_lbl_path)
        x_test = _read_idx_images(test_img_path)
        y_test = _read_idx_labels(test_lbl_path)
        # Normalize to float32 in range [-1, 1]
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        x_train = (x_train - 0.5) / 0.5
        x_test = (x_test - 0.5) / 0.5
        return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)

    # If IDX files are missing, require torchvision to download them (PyTorch-only download)
    try:
        from torchvision import datasets
    except Exception:
        raise RuntimeError('Raw IDX files not found and torchvision is not available to download them.\n'
                           'Install torchvision or place IDX files under data/FashionMNIST/raw')

    # Request torchvision to download the raw IDX files into `root` (default 'data')
    root = os.path.join('data')
    datasets.FashionMNIST(root=root, train=True, download=True)
    datasets.FashionMNIST(root=root, train=False, download=True)

    # After download, read the IDX files (expect them to exist now)
    if all(os.path.exists(p) for p in (train_img_path, train_lbl_path, test_img_path, test_lbl_path)):
        x_train = _read_idx_images(train_img_path)
        y_train = _read_idx_labels(train_lbl_path)
        x_test = _read_idx_images(test_img_path)
        y_test = _read_idx_labels(test_lbl_path)
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        x_train = (x_train - 0.5) / 0.5
        x_test = (x_test - 0.5) / 0.5
        return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)

    raise FileNotFoundError('Unable to obtain Fashion-MNIST IDX files after attempting download via torchvision.')


def get_torch_datasets(root='data', transform=None, download=True, raw_dir=None):
    """Return (train_dataset, test_dataset) suitable for PyTorch training.

    If `torchvision` is available and `download=True`, this will prefer
    returning `torchvision.datasets.FashionMNIST` instances so the caller can
    pass transforms. If IDX raw files exist (or are downloaded), the function
    will return `torch.utils.data.TensorDataset` built from the raw IDX files.

    Arguments:
    - root: base data directory (default: 'data')
    - transform: torchvision transform applied when returning torchvision datasets
    - download: whether to attempt download via `torchvision.datasets.FashionMNIST`
    - raw_dir: explicit raw directory (overrides default `data/FashionMNIST/raw`)
    """
    if raw_dir is None:
        raw_dir = os.path.join(root, 'FashionMNIST', 'raw')

    train_img_path = os.path.join(raw_dir, 'train-images-idx3-ubyte')
    train_lbl_path = os.path.join(raw_dir, 'train-labels-idx1-ubyte')
    test_img_path = os.path.join(raw_dir, 't10k-images-idx3-ubyte')
    test_lbl_path = os.path.join(raw_dir, 't10k-labels-idx1-ubyte')

    # If raw IDX files exist already, return TensorDataset built from normalized numpy output
    if all(os.path.exists(p) for p in (train_img_path, train_lbl_path, test_img_path, test_lbl_path)):
        try:
            import torch
            from torch.utils.data import TensorDataset
        except Exception as e:
            raise RuntimeError('PyTorch is required to build TensorDataset from IDX files') from e

        x_train, y_train, x_test, y_test = load_numpy(raw_dir)
        # load_numpy returns normalized float arrays in [-1,1]
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)

        tX_train = torch.from_numpy(x_train)
        tY_train = torch.from_numpy(y_train.astype(np.int64))
        tX_test = torch.from_numpy(x_test)
        tY_test = torch.from_numpy(y_test.astype(np.int64))

        train_ds = TensorDataset(tX_train, tY_train)
        test_ds = TensorDataset(tX_test, tY_test)
        return train_ds, test_ds

    # Raw files missing. If download requested, require torchvision to perform download.
    if download:
        try:
            from torchvision import datasets
        except Exception:
            raise RuntimeError('Raw IDX files not found; torchvision is required to download them.')

        # Ask torchvision to download the raw files, then return torchvision dataset objects
        datasets.FashionMNIST(root=root, train=True, download=True)
        datasets.FashionMNIST(root=root, train=False, download=True)

        # Now return torchvision dataset objects which will apply `transform` as requested
        train_ds = datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
        return train_ds, test_ds

    # If we reach here, raw files missing and download was False: error out
    raise FileNotFoundError('Raw IDX files not found under {} and download=False.'.format(raw_dir or os.path.join(root, 'FashionMNIST', 'raw')))


def get_torch_loaders(batch_size=64, root='data', transform=None, download=True, raw_dir=None, num_workers=0, pin_memory=False):
    """Return (train_loader, test_loader) for PyTorch training.

    Parameters are passed to DataLoader; datasets are created via
    `get_torch_datasets` which prefers torchvision datasets (so transforms apply)
    when available, otherwise builds TensorDataset from raw IDX files.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError('PyTorch is required for get_torch_loaders') from e

    train_ds, test_ds = get_torch_datasets(root=root, transform=transform, download=download, raw_dir=raw_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader
