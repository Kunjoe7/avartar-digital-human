import os
import sys

# Make `modules.*` importable when pytest runs from anywhere. The deterministic
# clinical modules under test (instruments/runtime/crisis/...) are pure Python — they
# must never require the GPU/audio stack, so no heavy imports happen here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
