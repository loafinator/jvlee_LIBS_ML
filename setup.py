"""
Python > setup.py
"""

from pathlib import Path 
import sys 

def add_project_root_to_path(parent_generation: int=1):
    """Adds the project root to the sys.path so that 
    submodules can be imported more cleanly."""
    try:
        project_root = Path(__file__).resolve().parents[parent_generation]
    except NameError:
        project_root = Path.cwd().parents[parent_generation - 1]
    sys.path.insert(0, str(project_root))
    print(f'Added to the sys.path: {project_root}')