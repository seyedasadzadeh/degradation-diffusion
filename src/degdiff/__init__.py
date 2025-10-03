"""degdiff package - lightweight extraction from learner.ipynb for tests and demo."""

from .generators import ParisLawDegradation

# Try to export model classes if torch is available. Use try/except so importing
# the package doesn't fail on systems without torch during lightweight tasks.
try:
	from .model_def import TimeSeriesDiffusionModel, DegDiffusion  # type: ignore
	__all__ = ["ParisLawDegradation", "TimeSeriesDiffusionModel", "DegDiffusion"]
except Exception:
	__all__ = ["ParisLawDegradation"]
