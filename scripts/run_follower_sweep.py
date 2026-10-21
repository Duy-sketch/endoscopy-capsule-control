"""Reproducible noise/speed sweep of test_passive_follower.py; no gain changes.

Place beside test_passive_follower.py in scripts/. Batch runs are headless;
the existing single-run script retains its default viewer.
Default: three circle durations, three all-noise seeds and one no-noise
baseline per duration (12 runs). Completion is not a stability proof.
Optional Cartesian grid: --current-taus-ms 0 10 --localization-delays-ms 0 20.
Values are hypothetical sensitivity tests until measured in the lab.
Outputs: summary.csv, grouped_summary.csv, metadata.json, and raw CSV/JSON/log
per case. Group error ranges use completed runs only; stopped runs are counted.
Single-run viewer remains enabled by default; this batch script is headless.
Validated 2026-10-06 with MuJoCo 3.15.0 and the common-current follower:
12/12 runs completed inside ROI. With all noise, seeds 42/43/44, circle
radius 5 mm, during-motion XY RMS ranges (mm):
40 s/revolution: 0.040..0.054; 20 s: 0.128..0.134; 10 s: 0.507..0.509.
No gain changes or observed current clipping in logged samples.
These are finite simulation tests, not a proof of stability.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def summarize(path, duration, move_time):
    data=np.genfromtxt(path,delimiter=',',names=True)
    data=np.atleast_1d(data)
    if data.size<2:
        raise ValueError('Fewer than two recorded samples')
    def vector(prefix):
        return np.column_stack([data[prefix+axis+'_m'] for axis in 'xyz'])
    times=data['time_s']
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid or duplicate timestamps')
    pos=vector('capsule_world_');reference=vector('capsule_ref_world_')
    robot=vector('dema_world_');robot_ref=vector('leader_ref_world_')
    error=pos-reference
    # All test trajectories translate in the initial DEMA XY plane.
    # Its normal is available from capsule_ref - leader_ref (100 mm).
    normal=reference[0]-robot_ref[0]
    normal/=np.linalg.norm(normal)
    vertical=error@normal
    lateral_squared=np.maximum(0,np.sum(error**2,axis=1)-vertical**2)
    moving=(times>0)&(times<=move_time+1e-9)
    if not moving.any():raise ValueError('No during-motion samples')
    command=np.column_stack([data['i1_command_A'],data['i2_command_A']])
    actual=np.column_stack([data['i1_actual_A'],data['i2_actual_A']])
    finite=np.isfinite(np.column_stack([pos,reference,robot,robot_ref,command,actual,
                                       data['alignment_deg'],data['z_lcs_m']])).all()
    rms=lambda x:float(np.sqrt(np.mean(x**2)))
    radial=np.hypot(data['x_lcs_m'],data['y_lcs_m'])
    roi=(radial<=.08)&(data['z_lcs_m']>.04)&(data['z_lcs_m']<.18)
    return {
        'completed':bool(finite and times[-1]>=duration-.0011 and roi.all()),
        'final_time_s':float(times[-1]),
        'motion_path_xy_rms_mm':float(np.sqrt(np.mean(lateral_squared[moving]))*1000),
        'motion_path_3d_rms_mm':float(np.sqrt(np.mean(np.sum(error[moving]**2,axis=1)))*1000),
        'motion_leader_3d_rms_mm':float(np.sqrt(np.mean(np.sum((robot-robot_ref)[moving]**2,axis=1)))*1000),
        'whole_run_z_rms_mm':rms(data['z_lcs_m']-.1)*1000,
        'max_z_error_mm':float(np.max(abs(data['z_lcs_m']-.1))*1000),
        'max_radial_error_mm':float(np.max(radial)*1000),
        'max_alignment_deg':float(np.max(data['alignment_deg'])),
        'max_command_A':float(np.max(abs(command))),
        'max_actual_A':float(np.max(abs(actual))),
        'equal_command_max_error_A':float(np.max(abs(command[:,0]-command[:,1]))),
        # Supply clipping can happen between log samples; not a per-step count.
        'logged_actual_at_20A_fraction':float(np.mean(np.any(abs(actual)>=20.-1e-8,axis=1))),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner',type=Path,default=Path(__file__).with_name('test_passive_follower.py'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--move-times',type=float,nargs='+',default=[40.,20.,10.])
    parser.add_argument('--current-taus-ms',type=float,nargs='+',default=[0.])
    parser.add_argument('--localization-delays-ms',type=float,nargs='+',default=[0.])
    parser.add_argument('--radius-mm',type=float,default=5.)
    parser.add_argument('--tail-time',type=float,default=5.)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=300.,help='Wall-clock timeout per simulation')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if (not np.isfinite([*args.move_times,args.radius_mm,args.tail_time,args.timeout]).all()
            or min(args.move_times)<=0 or args.radius_mm<=0 or args.tail_time<0
            or min(args.seeds)<0 or args.jobs<1 or args.timeout<=0):
        parser.error('Invalid duration, radius, seed, job count or timeout')
    from endoscopy_capsule_control.config import DEFAULT_CONFIG
    values=[*args.current_taus_ms,*args.localization_delays_ms]
    period_ms=DEFAULT_CONFIG.simulation.control_dt*1000
    if not np.isfinite(values).all() or min(values)<0:
        parser.error('Delay and source tau must be finite and nonnegative')
    if any(not np.isclose(d/period_ms,round(d/period_ms),rtol=0,atol=1e-8) for d in args.localization_delays_ms):
        parser.error(f'Localization delays must be multiples of {period_ms:g} ms')
    runner=args.runner.resolve()
    if not runner.is_file():parser.error(f'Runner not found: {runner}')
    output=(args.output or Path('results')/datetime.now(timezone.utc).strftime('follower_sweep_%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):parser.error('Output directory must be empty; choose a new output directory')
    cases=[]
    from itertools import product
    for duration,tau,delay in product(dict.fromkeys(args.move_times),
                                     dict.fromkeys(args.current_taus_ms),
                                     dict.fromkeys(args.localization_delays_ms)):
        for noise,seed in [('none',args.seeds[0]),*[('all',s) for s in dict.fromkeys(args.seeds)]]:
            cases.append({'case_id':len(cases),'move_time_s':duration,'noise':noise,'seed':seed,
                          'current_tau_ms':tau,'localization_delay_ms':delay,
                          'peak_requested_speed_mm_s':3*np.pi*args.radius_mm/duration})
    from dataclasses import asdict
    from endoscopy_capsule_control.config import DEFAULT_CONFIG
    from endoscopy_capsule_control.simulation.mujoco_system import default_model_path
    metadata={'arguments':vars(args)|{'runner':str(runner),'output':str(output)},
              'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),
              'config':asdict(DEFAULT_CONFIG),
              'model_xml_sha256':hashlib.sha256(default_model_path().read_bytes()).hexdigest(),
              'python':sys.version,'mujoco':importlib.metadata.version('mujoco'),
              'numpy':np.__version__,'utc_start':datetime.now(timezone.utc).isoformat(),
              'cases':cases,'notes':'Tau and delay values are assumed, not identified hardware. Per-case JSON stores effective configuration. No gain changes; baseline synthetic noise models. Completion means duration reached inside diagnostic ROI, not proven stability. Actual clipping fraction is based on logged samples and the baseline 20 A limit.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    def run(case):
        label=f"case_{case['case_id']:02d}_{case['noise']}_seed{case['seed']}"
        raw=output/(label+'.csv');log=output/(label+'.txt')
        total=case['move_time_s']+args.tail_time
        command=[sys.executable,str(runner),'--mode','follow','--trajectory','circle',
                 '--radius-mm',str(args.radius_mm),'--move-time',str(case['move_time_s']),
                 '--time',str(total),'--x0-mm','0','--noise',case['noise'],'--seed',str(case['seed']),
                 '--current-tau-ms',str(case['current_tau_ms']),
                 '--localization-delay-ms',str(case['localization_delay_ms']),
                 '--headless','--csv',str(raw)]
        result=case|{'completed':False,'stop_reason':'not started','error':'','csv':str(raw),'log':str(log)}
        try:
            with log.open('w',encoding='utf-8') as stream:
                stream.write(' '.join(command)+'\n');stream.flush()
                p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout)
            if p.returncode!=0:raise RuntimeError(f'Runner exit code {p.returncode}; see log')
            run_metadata=json.loads(raw.with_suffix('.json').read_text(encoding='utf-8'))
            result['stop_reason']=run_metadata['status']
            result['final_time_s']=run_metadata['actual_time_s']
            if run_metadata['actual_time_s']>0:
                result.update(summarize(raw,total,case['move_time_s']))
            result['completed']=bool(result['completed'] and result['stop_reason']=='duration completed')
            metrics=run_metadata.get('metrics',{})
            result['minimum_sampled_clearance_mm']=metrics.get('minimum_sampled_clearance_mm')
            result['final_alignment_deg']=metrics.get('final_alignment_deg')
            for channel,value in enumerate(metrics.get('clean_current_tracking_rms_A',[]),1):
                result[f'i{channel}_clean_tracking_rms_A']=value
        except Exception as exc:
            result['completed']=False
            result['error']=str(exc)
            if result['stop_reason']=='not started':result['stop_reason']='execution or output error'
        return result
    results=[]
    def save():
        fields=list(dict.fromkeys(k for r in sorted(results,key=lambda r:r['case_id']) for k in r))
        with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            w.writerows(sorted(results,key=lambda r:r['case_id']))
    def save_groups():
        groups={}
        keys=('move_time_s','noise','current_tau_ms','localization_delay_ms')
        for row in results:
            groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
        grouped=[]
        for key,rows in sorted(groups.items()):
            good=[r for r in rows if r['completed'] and not r['error']]
            entry=dict(zip(keys,key))|{'runs':len(rows),'completed_runs':len(good),
                'stopped_runs':sum(not r['completed'] and not r['error'] for r in rows),
                'error_runs':sum(bool(r['error']) for r in rows)}
            for metric in ('motion_path_xy_rms_mm','whole_run_z_rms_mm','max_z_error_mm','max_alignment_deg','max_actual_A'):
                values=[r[metric] for r in good]
                entry[metric+'_min_completed']=min(values) if values else ''
                entry[metric+'_max_completed']=max(values) if values else ''
            grouped.append(entry)
        with (output/'grouped_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(grouped[0]));writer.writeheader();writer.writerows(grouped)
    print(f'{len(cases)} runs; {args.jobs} workers. Output: {output}',flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run,case) for case in cases]):
            r=future.result();results.append(r);save();save_groups()
            print(f"[{len(results)}/{len(cases)}] move={r['move_time_s']:g}s {r['noise']} seed={r['seed']} tau={r['current_tau_ms']:g}ms delay={r['localization_delay_ms']:g}ms completed={r['completed']} XY_RMS={r.get('motion_path_xy_rms_mm',float('nan')):.3f}mm Zmax={r.get('max_z_error_mm',float('nan')):.3f}mm {r['stop_reason']} {r['error']}",flush=True)
    print(f"Completed inside ROI: {sum(bool(r['completed']) for r in results)}/{len(cases)}")
    print('Summary:',output/'summary.csv')
    print('Grouped completed-run ranges:',output/'grouped_summary.csv')
    if any(r['error'] for r in results):sys.exit(1)


if __name__=='__main__':main()
