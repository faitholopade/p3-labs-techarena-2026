"""Collect all disclosed V23 experiments and retain only both-date improvements."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from screen_v23_temporal import corrected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);grid=[]
    for name in ['control','decay3','decay1','corr05','corr1','history','event_rate']:
        folder=name+'_screen' if name!='history' else 'history_screen_reproduced'
        report=json.loads((a.experiments/folder/'report.json').read_text())
        for row in report['results']:
            grid.append(dict(row,family=name,folder=folder,both_dates_improve=all(row['gain_vs_v20'][k]>0 for k in ['earlier','later'])))
    (a.out/'grid.json').write_text(json.dumps(grid,indent=2)+'\n');eligible=[r for r in grid if r['both_dates_improve']]
    summary={'version':'v23_research','grid_entries':len(grid),'both_date_candidates':len(eligible),'official_incumbent':'v20','official_auc':.832548,
             'portal_uploads':0,'last_reported_attempts':8,'latest_result':'V22 regressed to 0.832538; incumbent unchanged.',
             'limitations':'Adaptive research on reused weekend cohorts; no weekday confirmation; long-gap gains are calibration-only.'}
    if not eligible:
        summary.update(status='no_candidate',selected=None,best_combined_only=max(grid,key=lambda r:r['gain_vs_v20']['combined']))
    else:
        maximum=max(r['gain_vs_v20']['combined'] for r in eligible)
        selected=min([r for r in eligible if r['gain_vs_v20']['combined']>=maximum-.000002],key=lambda r:(r['weight'],not r.get('conditional',True),r['family']))
        ev=dict(np.load(a.incumbent));folder=a.experiments/selected['folder']
        if selected['family'] in ['history','event_rate']:
            new=np.load(folder/'predictions.npz')['standalone'];pred=(1-selected['weight'])*ev['candidate']+selected['weight']*new
        else:
            delta=np.load(folder/'correction.npz')['matched_delta'];pred=corrected(ev['candidate'],delta,selected['weight'],selected['conditional'])
        np.savez_compressed(a.out/'evidence.npz',baseline=ev['candidate'],candidate=pred,labels=ev['labels'],users=ev['users'],period=ev['period'],indices=ev['indices'])
        summary.update(status='candidate_requires_uncertainty_assessment',selected=selected,selection_rule='Both original dates improve; within 0.000002 of best combined gain prefer smaller correction weight, then preservation of event click mass. Selection rule is adaptive after earlier families were inspected.')
    (a.out/'selection.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
