"""Pytest root configuration.

Its presence at the project root makes pytest add the root to sys.path, so
'from app.main import app' resolves no matter which directory pytest is
invoked from. Without it, running bare 'pytest' can raise ModuleNotFoundError.
"""