"""Replay the same frozen cone and out-of-view cases with both trained detectors."""
from . import run_continuous_001 as bench
from . import run_hard_negatives_001 as trial


def main():
    configs = []
    for name, folder in [('multigame_baseline',trial.PRIOR),('hard_negative_candidate',trial.RUN)]:
        ck=trial.pilot.read_json(folder/'trained_checkpoint.json')
        path=folder/ck['checkpoint']
        if trial.pilot.digest(path)!=ck['sha256']:
            raise ValueError('Model changed')
        configs.append((name,path,0,960))
    paths = []
    for name in ('continuous_001','continuous_002'):
        output=trial.RUN/f'replay_{name}'
        bench.run(trial.pilot.WORK/name,output,configs)
        paths.append(output/'report.json')
    baseline_data = [bench.read_json(trial.pilot.WORK/'continuous_001/comparison/report.json'),
                     bench.read_json(trial.pilot.WORK/'grass_filter_001/prospective/report.json')]
    totals={}
    for path,original,original_name in zip(paths,baseline_data,('multi_match_n','strict')):
        result=bench.read_json(path)
        for key,clip in result['clips'].items():
            prior=original['clips'][key]['details'][original_name]
            now=clip['details']['multigame_baseline']
            if [(r['state'],r['x'],r['y']) for r in now] != [(r['state'],r['x'],r['y']) for r in prior]:
                raise ValueError('Baseline replay differs from historical track positions')
        totals[path.parent.name]=result['models']
    bench.write_json(trial.RUN/'tracking_comparison.json',totals)


if __name__=='__main__':
    main()
