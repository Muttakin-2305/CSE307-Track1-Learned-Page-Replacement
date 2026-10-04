# conftest.py - makes pytest find src/ automatically without PYTHONPATH
import sys
from pathlib import Path

# Add src/ to the path so test files can import from it directly
sys.path.insert(0, str(Path(__file__).parent / "src"))
