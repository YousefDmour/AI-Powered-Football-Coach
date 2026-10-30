"""Score fixed baseline/tiled ablations without changing annotations or predictions."""

import argparse
import json
from pathlib import Path

import pandas as pd

from vision.evaluate_ball import evaluate


def score(output,annotations):
    output,annotations=Path(output),Path(annotations)
    labels=json.loads(annotations.read_text(encoding='utf-8'))
    report={'annotations':str(annotations),'modes':{},
            'limits':'Development results are not held out. Prospective test labels are assistant-reviewed, not independent human ground truth. No performance claim applies outside sampled frames.'}
    for mode in ('baseline','tiled_strict','tiled_feet'):
        table=pd.read_csv(output/f'{mode}.csv')
        result=evaluate(table,labels['frames'])
        result['limits']='Annotation provenance is inherited from the supplied label file; evaluate only within this sampled window.'
        report['modes'][mode]=result
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{key:value for key,value in result.items() if key not in ('details','limits')}
                      for name,result in report['modes'].items()},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    args=parser.parse_args();score(args.output,args.annotations)
