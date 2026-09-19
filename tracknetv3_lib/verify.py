import argparse

import pandas as pd


def compare(baseline_csv, candidate_csv):
    """ Compare two prediction csv files frame by frame.

        Args:
            baseline_csv (str): File path of the reference prediction csv
            candidate_csv (str): File path of the candidate prediction csv

        Returns:
            match (bool): Whether the two csv files are identical
    """
    base = pd.read_csv(baseline_csv)
    cand = pd.read_csv(candidate_csv)

    if len(base) != len(cand):
        print(f'Mismatch: baseline has {len(base)} rows, candidate has {len(cand)} rows.')
        return False
    if list(base.columns) != list(cand.columns):
        print(f'Mismatch: columns differ. baseline: {list(base.columns)}, candidate: {list(cand.columns)}')
        return False

    diff = base != cand
    if diff.to_numpy().any():
        n = int(diff.to_numpy().sum())
        print(f'Mismatch: {n} cells differ.')
        bad = diff.any(axis=1)
        print(pd.concat([base[bad].head(), cand[bad].head()], axis=1, keys=['baseline', 'candidate']))
        return False

    print(f'PASS: {len(base)} rows identical across columns {list(base.columns)}.')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help='csv file from the original predict.py')
    parser.add_argument('--candidate', type=str, required=True, help='csv file from tracknetv3_lib')
    args = parser.parse_args()

    ok = compare(args.baseline, args.candidate)
    raise SystemExit(0 if ok else 1)
