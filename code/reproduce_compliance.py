"""Recalculate manuscript Figure 6."""
from pathlib import Path
import runpy,sys
if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    sys.argv=[str(root/'reproduce.py'),'--figures','6','--replay',*sys.argv[1:]]
    runpy.run_path(str(root/'reproduce.py'),run_name='__main__')
