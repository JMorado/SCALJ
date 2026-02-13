"""Workflow utility functions.

This module re-exports I/O utilities from the main scalej.io module
for backward compatibility with existing workflow nodes.
"""

from scalej.io import load_pickle, save_pickle

__all__ = ["load_pickle", "save_pickle"]
