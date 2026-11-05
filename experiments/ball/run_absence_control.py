"""Compare both trained detectors on the different-venue ball-absence control."""
from . import run_continuous_001 as bench


def main():
    root = bench.pilot.WORK
    source = root / 'cone_holdout_001'
    configs = []
    for name, folder in [('multigame_baseline', 'pilot_002_multigame'),
                         ('hard_negative_candidate', 'pilot_003_hard_negatives')]:
        run = root / folder
        checkpoint = bench.read_json(run / 'trained_checkpoint.json')
        weights = run / checkpoint['checkpoint']
        if bench.digest(weights) != checkpoint['sha256']:
            raise ValueError('Checkpoint changed')
        configs.append((name, weights, 0, 960))
    bench.run(source, source / 'comparison', configs)


if __name__ == '__main__':
    main()
