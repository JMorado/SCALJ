"""SCALeJ: Lennard-Jones Parameter Fitting via Condensed-Phase Volume-Scaling."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("scalej")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"
