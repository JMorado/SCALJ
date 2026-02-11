import pickle as pkl
from pathlib import Path
from typing import Any


def load_pickle(file_path: Path) -> Any:
    """
    Load a pickle file.

    Parameters
    ----------
    file_path : Path
        Path to the pickle file.

    Returns
    -------
    Any
        The object loaded from the pickle file.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"Pickle file not found: {file}")

    with open(file_path, "rb") as f:
        return pkl.load(f)


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Save an object to a pickle file.

    Parameters
    ----------
    obj : Any
        The object to save.
    file_path : Path
        Path to the output pickle file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pkl.dump(obj, f)
