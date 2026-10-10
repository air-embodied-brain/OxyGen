"""Derive replacement figures/tables from a validated run without copying paper numbers."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from validate import module,read


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def table(folder,name,headers,rows,speed_columns=()):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.csv')).open('w') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    def fmt(v):
        return f'{v:.2f}' if isinstance(v,float) else str(v).replace('_',r'\_')
    lines=[r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(fmt(v)+(r'$\times$' if i in speed_columns and isinstance(v,(float,int)) else '') for i,v in enumerate(row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    (folder/(name+'.tex')).write_text('\n'.join(lines)+'\n')


def build(root,tp2=False):
    paper=root/'paper'; tables=paper/'tables'
    # Frozen driver reference: retain its per-frame mean and actual completion windows.
    reference=root/'pi_reference/index.json'
    if reference.exists():
        reference_rows=[]; completions=[]
        for name in read(reference)['files']:
            r=read(reference.parent/name);params=r['params'];n=params['max_decoding_steps'];k=params.get('steps_per_frame')
            frames=[f for f in r['frames'] if not f['is_warmup']]
            latency=statistics.fmean(f['frame_ms'] for f in frames)
            reference_rows.append([r['setting'],n,k if k else '--',latency,10000/latency])
            if k:
                byframe={f['frame_idx']:f['frame_ms'] for f in frames};length=n//k
                times=[sum(byframe[j] for j in range(i,i+length))/1000 for i in sorted(byframe) if all(j in byframe for j in range(i,i+length))]
                if times:completions.append([n,k,len(times),statistics.median(times),max(times)])
        table(tables,'pi_reference',['Mode','$N$','$k$','Frame (ms)','Action frequency (Hz)'],reference_rows)
        table(tables,'completion_time',['$N$','$k$','Requests','Median (s)','Max (s)'],completions)
    # Same median-of-repeat-means as the formal Xiaomi summary.
    x=read(root/'xiaomi/raw.json'); groups=defaultdict(list)
    for r in x['results']: groups[(r['mode'],r['max_decoding_steps'],r['steps_per_frame'])].append(r)
    xagg={k:statistics.median(r['mean_frame_ms'] for r in rows) for k,rows in groups.items()}
    xrows=[]; cross=[]
    for n in [5,10,15,20,30]:
        b=xagg['baseline',n,None]; shared=xagg['shared_kv',n,None]
        speed=[b/xagg['continuous_batching',n,k] if n%k==0 else '--' for k in [1,5,10]]
        xrows.append([n,b,shared,*speed])
        if n==30: cross.append(['Xiaomi-Robotics-0',2,b,*speed[:2]])
    table(tables,'xiaomi_full',['$N$','Baseline (ms)','Shared KV (ms)','$k=1$','$k=5$','$k=10$'],xrows,speed_columns=(3,4,5))
    # Preserve synchronized stage timing for diagnosis; end-to-end wall-clock
    # timing is the manuscript-facing comparison and gets its own reduction.
    spread=[]
    for (mode,n,k),reps in groups.items():
        vals=[r['mean_frame_ms'] for r in reps]
        spread.append([mode,n,k if k else '--',statistics.median(vals),min(vals),max(vals)])
    table(tables,'xiaomi_repeat_spread',['Mode','$N$','$k$','Median (ms)','Min (ms)','Max (ms)'],spread)
    s=read(root/'starvla/raw.json'); srows=[]
    for row in sorted(s['summary'],key=lambda r:(r['expert_count'],r['language_tokens'])):
        b=row['separate']['latency_ms']; cs={p['steps_per_frame']:p for p in row['persistent_continuous']}
        speed=[b/cs[k]['mean_frame_ms'] if k in cs else '--' for k in [1,5,10]]
        srows.append([row['expert_count'],row['language_tokens'],b,row['shared']['latency_ms'],*speed])
        if row['language_tokens']==30: cross.append(['StarVLA',row['expert_count'],b,*speed[:2]])
    table(tables,'starvla_full',['$E$','$N$','Baseline (ms)','Shared KV (ms)','$k=1$','$k=5$','$k=10$'],srows,speed_columns=(4,5,6))
    table(tables,'model_transfer',['Model','$E$','Baseline (ms)','$k=1$','$k=5$'],cross,speed_columns=(3,4))
    qroot=root/'qwen35_scaling_tp_formal_20260728'; analyzer=module('analyze_qwen35_scaling_tp')
    qa={}; qr=[]; signatures={}
    for tp in ([1,2] if tp2 else [1]):
        for size,key in [('0.8B','0p8b'),('2B','2b'),('4B','4b'),('9B','9b')]:
            d=read(qroot/f'formal_{key}_tp{tp}.json')
            signature=d['metadata']['input_signature']
            if size in signatures and signatures[size]!=signature: raise ValueError('TP input mismatch: '+size)
            signatures[size]=signature
            agg=analyzer.aggregate(d); qa[f'{size}-TP{tp}']={f'{m}|N={n}|k={k}':v for (m,n,k),v in agg.items()}
            for n in [10,20,30]:
                b=agg['separate',n,None]['latency_ms']
                qr.append([size,tp,n,b,*[b/agg['continuous_batching',n,k]['latency_ms'] for k in [1,5,10]]])
    dump(qroot/'aggregated.json',qa)
    table(tables,'model_scaling',['Backbone','TP','$N$','Baseline (ms)','$k=1$','$k=5$','$k=10$'],qr,speed_columns=(4,5,6))
    # Convert manager schema consumed by the approved plot; no hard-coded measurements.
    managers={}; mr=[]
    mapping={'prefix_ms':'prefix_ms_mean','denoise_ms':'denoise_ms_mean','decode_ms':'decode_ms_mean','kv_stack_ms':'kv_stack_ms_mean','kv_split_ms':'kv_split_ms_mean','crud_ms':'crud_total_ms_mean','manager_ms':'manager_total_ms_mean','mean_ms':'frame_ms_mean','manager_fraction':'manager_fraction_of_full_frame'}
    for backend in ['jax','pytorch']:
        d=read(root/f'manager/{backend}.json'); points={}
        for k,p in d['points'].items():
            summary=p['profiled_summary']; converted={a:summary[b] for a,b in mapping.items()}
            points['N30_k'+k]={'synchronized_full_path':converted,'natural_end_to_end':p['natural_summary']}
            mr.append([backend,int(k),p['natural_summary']['frame_ms_mean'],summary['frame_ms_mean'],summary['kv_stack_ms_mean'],summary['kv_split_ms_mean'],summary['manager_total_ms_mean'],100*summary['manager_fraction_of_full_frame']])
        managers[backend]={'points':points}
    dump(root/'kv_manager_fullpath_aligned_20260729/aggregate.json',managers)
    table(tables,'manager_all',['Backend','$k$','Natural (ms)','Synchronized (ms)','Stack (ms)','Split (ms)','Manager (ms)','Manager (\%)'],mr)
    jax=read(root/'manager/jax.json')
    summaries=[jax['points'][str(k)]['profiled_summary'] for k in [5,1]]
    fields=[('Prefix','prefix_ms_mean'),('Denoise','denoise_ms_mean'),('Language decode','decode_ms_mean'),('Request admission','admission_total_ms_mean'),('KV stack','kv_stack_ms_mean'),('KV split','kv_split_ms_mean'),('Request bookkeeping','crud_total_ms_mean'),('Other frame work','unattributed_ms_mean'),('Synchronized full frame','frame_ms_mean'),('Manager total','manager_total_ms_mean')]
    rows=[[label,*[(f'{v[key]:.3f}' if key=='crud_total_ms_mean' else v[key]) for v in summaries]] for label,key in fields]
    rows.append(['Manager / synchronized frame',*[f"{100*v['manager_fraction_of_full_frame']:.2f}\\%" for v in summaries]])
    table(tables,'manager_overhead',['Measurement','$k=5$, batch 6','$k=1$, batch 30'],rows)
    sn=read(root/'small_n/profile.json'); sr=[]
    for r in sn['results']:
        b=r['baseline'];s=r['shared_kv']
        mean=lambda mode,key:statistics.fmean(x[key] for x in mode['stages'])
        bs=mean(b,'stage_sum_ms');ss=mean(s,'stage_sum_ms')
        sr.append([r['N'],mean(s,'prefill_ms'),mean(s,'denoise_ms'),mean(s,'decode_ms'),bs,ss,100*(1-ss/bs),b['e2e_mean_ms'],s['e2e_mean_ms']])
    table(tables,'small_n_profile',['$N$','Prefix','Denoise','AR decode','Isolated','Shared KV','Stage reduction (\%)'],[r[:7] for r in sr if r[0]<=5])
    e2e_rows=[r[:7]+[r[7],r[8],100*(1-r[8]/r[7])] for r in sr]
    table(tables,'small_n_e2e',['$N$','Prefix (ms)','Denoise (ms)','Decode (ms)','Stage isolated (ms)','Stage shared (ms)','Stage reduction (\%)','E2E isolated (ms)','E2E shared (ms)','E2E reduction (\%)'],e2e_rows)
    allocator=read(root/'allocator/profile.json')
    table(tables,'allocation',['Experts','Initialization (ms)','Per expert (ms)','Memory (MiB)'],[[r['append_expert_count'],r['total_init_ms'],r['init_ms_per_expert'],r['bytes_in_use_delta']/2**20] for r in allocator['alias_probe']['append_expert_scaling']])
    lr=[]; robust=module('analyze_robust_latency')
    for backend in ['jax','pytorch']:
        for scenario in ['steady','fluctuating']:
            folder=root/f'eccv_submission_raw/kv_manager_longrun_20260726/raw/bs32/{backend}_{scenario}_bs32'
            rows=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]; rows=[r for r in rows if not r['is_warmup']]
            period=3 if scenario=='steady' else 119; edge=100 if scenario=='steady' else 5
            stats=robust.period_analysis(rows,period,edge); mem=robust.memory_analysis(rows,backend,period,edge)
            dump(folder/'robust_stats.json',{'phase_aligned':stats,'memory':mem,'measured':robust.aggregate_rows(rows)})
            key='bytes_in_use' if backend=='jax' else 'allocated_bytes'; pool='pool_bytes' if backend=='jax' else 'reserved_bytes'
            cycles=robust.complete_periods(rows,period)
            trim=lambda vals:statistics.fmean(sorted(vals)[int(len(vals)*.05):len(vals)-int(len(vals)*.05)])
            first=trim([r['frame_ms'] for c in cycles[:edge] for r in c]);last=trim([r['frame_ms'] for c in cycles[-edge:] for r in c])
            lr.append([backend,scenario,mem[key]['min']/2**30,mem[key]['max']/2**30,mem[pool]['max']/2**30,first,last])
    table(tables,'longrun_all',['Backend','Workload','Min (GiB)','Max (GiB)','Pool (GiB)','First (ms)','Last (ms)'],lr)
    table(tables,'longrun',['Backend','Active memory (GiB)','Pool (GiB)','First / last latency (ms)'],[[r[0].upper() if r[0]=='jax' else 'PyTorch',f'{r[2]:.3f}--{r[3]:.3f}',r[4],f'{r[5]:.1f} / {r[6]:.1f}'] for r in lr if r[1]=='fluctuating'])
    dump(paper/'all_metrics.json',{'xiaomi':xrows,'starvla':srows,'model_transfer':cross,'scaling':qr,'manager':mr,'short_output':sr,'longrun':lr})
    (paper/'README.md').write_text('Generated from this run only. Tables provide tabular bodies; retain the reviewed paper captions and labels. Figure PDFs use the approved full-width canvas. Review corresponding prose and captions when replacing numbers. TP=2 comparisons must use TP=1 from this same run/host. No historical result is merged automatically.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--tp2',action='store_true');g.add_argument('--single-gpu',action='store_true')
    a=p.parse_args();build(a.root,a.tp2)
