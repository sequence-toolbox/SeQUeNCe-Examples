import sys
from pathlib import Path


QLAN_ROOT = Path(__file__).resolve().parents[1]
if str(QLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(QLAN_ROOT))
