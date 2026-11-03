from pathlib import Path
import argparse
from cav_dl.data.illustration import generate

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,default=Path('data'))
    args=parser.parse_args()
    generate(args.project,args.output)
