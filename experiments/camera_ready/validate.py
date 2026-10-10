"""Coverage and protocol checks; never require an expected speedup or trend."""
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics

BUNDLE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def positive(values):
    require(bool(values) and all(math.isfinite(v) and v > 0 for v in values), 'invalid frame timings')


def module(name):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / 'runners' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate(task, root):
    path = Path(root) / task['output']
    d = read(path)
    kind = task['kind']
    if kind == 'pi_reference':
        import hashlib
        points=set()
        require(len(d['files'])==23,'incomplete pi reference sweep')
        for name,sha in d['files'].items():
            f=path.parent/name
            require(hashlib.sha256(f.read_bytes()).hexdigest()==sha,'pi reference file changed')
            r=read(f); params=r['params']; mode=r['setting']; n=params['max_decoding_steps'];k=params.get('steps_per_frame')
            points.add((mode,n,k))
            require(params['num_denoise_steps']==10,'wrong pi reference S')
            frames=[x for x in r['frames'] if not x['is_warmup']]
            require(len(frames)==(50-n//k if k else 10),'pi reference frames missing')
            positive([x['frame_ms'] for x in frames])
        expected={(m,n,None) for m in ['baseline','shared_kv'] for n in [5,10,15,20,30]}
        expected|={('continuous_batching',n,k) for n in [5,10,15,20,30] for k in [1,5,10] if n%k==0}
        require(points==expected,'missing pi reference grid points')
    elif kind == 'qwen35':
        import copy
        checked=copy.deepcopy(d)
        checked['metadata']['fast_path_packages']={k:v.split('+')[0] for k,v in checked['metadata']['fast_path_packages'].items()}
        module('analyze_qwen35_scaling_tp').validate(task['label'], checked)
    elif kind == 'xiaomi':
        require(d['common_prefix_mode'] == 'libero_common', 'wrong Xiaomi input')
        require(d['optimization']['attention'] == 'flash_attention_2', 'wrong attention')
        require(not d['optimization']['torch_compile'] and not d['optimization']['cuda_graph'], 'compiled Xiaomi')
        groups = defaultdict(list)
        for row in d['results']:
            n, k = row['max_decoding_steps'], row['steps_per_frame']
            groups[row['mode'], n, k].append(row)
            require(row['denoise_steps'] == 5 and row['common_prefix_tokens'] == 199, 'Xiaomi protocol drift')
            require(row['action_prompt_tokens'] == row['language_prompt_tokens'] == 199, 'prefix mismatch')
            require(math.isfinite(row['action_checksum']), 'nonfinite action')
            positive(row['frame_times_ms'])
            require(math.isclose(statistics.fmean(row['frame_times_ms']),row['mean_frame_ms'],rel_tol=1e-9), 'mean mismatch')
            if k is not None:
                require(len(row['frame_times_ms']) == 50 and row['avg_batch_size'] == n/k, 'batch/frame mismatch')
            else:
                require(len(row['frame_times_ms']) in (10,30), 'single frame count mismatch')
        expected = {(m,n,None) for m in ['baseline','shared_kv'] for n in [5,10,15,20,30]}
        expected |= {('continuous_batching',n,k) for n in [5,10,15,20,30] for k in [1,5,10] if n%k==0}
        require(set(groups)==expected, 'incomplete Xiaomi grid')
        require(all(len(rows)==3 and {x['repeat'] for x in rows}=={0,1,2} for rows in groups.values()), 'missing repeats')
    elif kind == 'starvla':
        require(d['metadata']['prefix_tokens']==116 and d['metadata']['denoise_steps']==[4], 'StarVLA protocol drift')
        require(not d['metadata']['compile'] and not d['metadata']['cuda_graph'], 'compiled StarVLA')
        require(len(d['summary'])==15, 'incomplete expert sweep')
        require({(r['expert_count'],r['language_tokens']) for r in d['summary']}=={(e,n) for e in [2,3,4] for n in [5,10,15,20,30]}, 'missing expert points')
        for row in d['summary']:
            n=row['language_tokens']
            for mode in ['separate','shared']:
                require(len(row[mode]['repeat_means_ms'])==3, 'missing single repeats')
                positive(row[mode]['repeat_means_ms'])
                records=[x for x in d['records'] if x['expert_count']==row['expert_count'] and x['language_tokens']==n and x['mode']==mode]
                require(len(records)==3 and {x['repeat'] for x in records}=={0,1,2}, 'missing raw single repeats')
                require(all(len(x['frames'])==row['single_measured_frames_per_repeat'] for x in records), 'single frames mismatch')
                means=[statistics.fmean(f['total_ms'] for f in x['frames']) for x in records]
                require(math.isclose(statistics.median(means),row[mode]['latency_ms'],rel_tol=1e-9), 'StarVLA aggregate mismatch')
            require(row['single_correctness']['separate_vs_shared']['language']['token_ids_equal'], 'token mismatch')
            points=row['persistent_continuous']
            require({x['steps_per_frame'] for x in points}=={k for k in [1,5,10] if n%k==0}, 'missing k')
            for p in points:
                require(p['all_completed_lengths_correct'] and p['all_actions_finite'], 'invalid continuation/action')
                require(p['avg_batch_size']==n/p['steps_per_frame'], 'wrong batch')
                require(len(p['repeat_rows'])==3, 'missing CB repeats')
                for rep in p['repeat_rows']:
                    require(len(rep['frame_times_ms'])==50, 'missing CB frames')
                    positive(rep['frame_times_ms'])
    elif kind == 'small_n':
        expected=[int(x) for x in task['args'][task['args'].index('--decode-steps')+1:task['args'].index('--denoise-steps')]]
        require([r['N'] for r in d['results']]==expected, 'missing short-N points')
        for row in d['results']:
            require(row['S']==10,'wrong S')
            for m in ['baseline','shared_kv']:
                require(len(row[m]['stages'])==len(row[m]['e2e_frames'])==10,'missing small-N repeats')
                positive([x['stage_sum_ms'] for x in row[m]['stages']])
    elif kind == 'manager':
        ks=task['args'][task['args'].index('--k-values')+1:task['args'].index('--steady-warmup')]
        require(set(d['points'])==set(ks),'missing manager k')
        require(d['metadata']['backend']==task['backend'],'wrong backend')
        for k,p in d['points'].items():
            for mode in ['natural','profiled']:
                rows=[x for x in p[mode+'_rows'] if x['is_measured']]
                require(len(rows)==20 and all(x['batch_size']==30/int(k) for x in rows),'manager frame/batch mismatch')
                positive([x['frame_ms'] for x in rows])
    elif kind == 'longrun':
        rows=[json.loads(l) for l in (path.parent/'frames.jsonl').read_text().splitlines()]
        measured=[r for r in rows if not r['is_warmup']]
        require(len(measured)==3000,'expected 3000 measured frames')
        require(len({r['frame'] for r in rows})==len(rows),'duplicate frames')
        positive([r['frame_ms'] for r in measured])
        require(d['metadata']['backend']==task['backend'],'wrong longrun backend')
        require(not d['summary']['oom'],'longrun encountered OOM')
        requests=[json.loads(l) for l in (path.parent/'requests.jsonl').read_text().splitlines()]
        completed=[r for r in requests if not r['warmup']]
        require(len(completed)==d['summary']['completed_requests'],'request count mismatch')
        require(len({r['request_id'] for r in requests})==len(requests),'duplicate completed requests')
        require(all(r['frame_latency']==math.ceil(r['tokens']/(5 if task['scenario']=='steady' else 4)) for r in requests),'request progress mismatch')
        args=d['metadata']['args']; steady=task['scenario']=='steady'
        require(args['frames']==(3030 if steady else 3100) and args['warmup_frames']==(30 if steady else 100),'wrong warmup')
        require(args['arrival_pattern']=='5,5,6' if steady else args['arrival_scale']==3,'wrong arrivals')
        require(args['decode_steps']==(30 if steady else 100) and args['steps_per_frame']==(5 if steady else 4),'wrong workload')
    elif kind == 'allocator':
        require({x['append_expert_count'] for x in d['alias_probe']['append_expert_scaling']}=={1,2,4},'missing allocation points')
        require(len(d['churn_probes'])==2,'missing allocation churn probe')
    elif kind == 'xiaomi_correctness':
        require(d['num_requests']==5, 'wrong Xiaomi request count')
        require(d.get('validation_policy')=='performance_implementation_v2', 'missing implementation validation policy')
        audit=d['implementation_audit']
        require(audit['policy']=='fixed_input_matched_batch_v2' and audit['passed'], 'Xiaomi implementation audit failed')
        require(audit['ages']==[0,1,3,5,8] and audit['steps']==30, 'incomplete Xiaomi audit coverage')
        require(audit['atol']==1e-4 and audit['rtol']==1e-3, 'changed matched-batch tolerance')
        require(audit['stack_split_roundtrip'] and audit['negative_row_swap_detected'] and audit['recycle_isolation'], 'cache isolation check failed')
        require(len(audit['records'])==150 and {(x['step'],x['row']) for x in audit['records']}=={(s,r) for s in range(30) for r in range(5)}, 'missing audit steps/rows')
        require(all(x['logits_close'] and x['cache_close'] and x['progress_correct'] and math.isfinite(x['logits_max_abs']) and math.isfinite(x['cache_max_abs']) for x in audit['records']), 'invalid cache/logits/progress')
        require(d['all_request_lengths_correct'], 'wrong request lengths')
        require(all(r['prefix_tokens']==199 and len(r['generated_token_ids'])==30 for r in d['requests']),'wrong correctness input/output')
    elif kind == 'native':
        require(d['metadata']['world_size']==task['tp'],'native TP mismatch')
        expected={('prefill',b) for b in [1,3,6]}|{('incremental_decode',b) for b in [1,3,6,30]}
        require({(r['stage'],r['batch_size']) for r in d['records']}==expected and len(d['records'])==7,'incomplete native decoder grid')
        for r in d['records']:
            require(r['finite'] and len(r['rank_max_times_ms'])==5,'native diagnostic invalid')
            positive(r['rank_max_times_ms'])
    else:
        raise ValueError('Unknown task kind '+kind)
    return d
