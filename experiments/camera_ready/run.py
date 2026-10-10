#!/usr/bin/env python3
"""Portable camera-ready suite: plan, preflight, run, resume, audit and render."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

from validate import read, require, validate

BUNDLE=Path(__file__).resolve().parent
REPO=BUNDLE.parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)

def tasks_for(config):
    return [t for t in read(BUNDLE/'protocol.json')['tasks'] if t.get('tp',1)==1 or config['include_tp2']]

def context(config):
    return dict(config['checkpoints'],bundle=str(BUNDLE),out=str(Path(config['output']).resolve()),physical_gpu=config['devices'][0],random_init='--random-init' if config.get('random_init', False) else '')

def command(task,config):
    cmd=[config['python'][task['environment']]]
    if task.get('tp'):
        cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node='+str(task['tp'])]
    args = [s.format(**context(config)) for s in task['args']
            if s != '{random_init}' or config.get('random_init', False)]
    return cmd+[str(BUNDLE/'runners'/task['script'])]+args

def environment(task,config):
    # Start from an allowlist: ROS/conda, loader paths, Python startup hooks,
    # distributed launch settings and unrelated CUDA options must not leak in.
    allowed = ['HOME','USER','LOGNAME','LANG','LC_ALL','TZ','TMPDIR',
               'HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','HF_MODULES_CACHE',
               'OPENPI_DATA_HOME','XDG_CACHE_HOME','JAX_COMPILATION_CACHE_DIR',
               'CUDA_CACHE_PATH','MPLCONFIGDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR',
               'NUMBA_CACHE_DIR']
    env={key:os.environ[key] for key in allowed if key in os.environ}
    env['PATH']=str(Path(config['python'][task['environment']]).parent)+':/usr/bin:/bin'
    env['PYTHONNOUSERSITE']='1'
    env['PYTHONFAULTHANDLER']='1'
    env.update(CUDA_VISIBLE_DEVICES=','.join(config['devices'][:task.get('tp',1)]),PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    pi=BUNDLE/'vendor/oxygen'
    paths=[BUNDLE/'runners',BUNDLE/'vendor/starvla',pi/'src',pi,pi/'packages/openpi-client/src']
    env['PYTHONPATH']=os.pathsep.join(map(str,paths))
    cache=Path(config['output'])/'cache'/task['environment']
    env['TRITON_CACHE_DIR']=str(cache/'triton'); env['TORCHINDUCTOR_CACHE_DIR']=str(cache/'inductor')
    env.update(task.get('environment_overrides',{}))
    return env

def asset_files(path):
    path=Path(path).resolve()
    # Expert checkpoints need their sibling model config and normalization stats.
    root=path.parent.parent if path.is_file() and path.parent.name=='checkpoints' else path
    require(root.exists(),f'Missing checkpoint: {root}')
    if root.is_file(): return root,{root.name:digest(root)}
    files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['.cache','.git','__pycache__'])}
    require(bool(files),f'Empty checkpoint: {root}')
    return root,files

def assets(config,create=False):
    if config.get('random_init', False):
        # Performance-only runs deliberately do not hash trained parameters.
        # Model directories remain required by each runner for config, code,
        # tokenizer/processor, and action-schema metadata.
        for key, path in config['checkpoints'].items():
            require(Path(path).exists(), f'Missing model metadata directory: {path}')
        manifest = {}
        for key, path in config['checkpoints'].items():
            _, files = asset_files(path)
            manifest[key] = files
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest={}
    for key,path in config['checkpoints'].items():
        print('Hash checkpoint:',key,flush=True)
        _,files=asset_files(path); manifest[key]=files
    lock=Path(config['assets_lock'])
    if create:
        require(not lock.exists(),'Asset lock already exists; choose a new filename rather than overwriting it.')
        write(lock,manifest)
    else:
        require(read(lock)==manifest,'Checkpoint contents differ from the source asset lock.')
    return digest(lock)

def source_check():
    entries=read(BUNDLE/'source-lock.json')
    for name,sha in entries.items():
        require(digest(REPO/name)==sha,f'Source/input changed: {name}')
    return digest(BUNDLE/'source-lock.json')

def preflight(config):
    require(not (config.get('random_init', False) and config['include_tp2']), 'Random-init TP2 is not implemented; use the single-GPU protocol.')
    require(len(config['devices'])>= (2 if config['include_tp2'] else 1),'Insufficient configured devices for TP2')
    require(len(set(config['devices']))==len(config['devices']),'Duplicate devices')
    result={'source_lock':source_check(),'assets_lock':assets(config),'hostname':platform.node(),'platform':platform.platform(),'environments':{}}
    # Current frequency/scaling is telemetry, not immutable resume identity.
    cpu_info=subprocess.check_output(['lscpu'],text=True)
    result['cpu']='\n'.join(line for line in cpu_info.splitlines()
                            if not line.startswith('CPU(s) scaling MHz:'))+'\n'
    result['cpu_affinity']=sorted(os.sched_getaffinity(0))
    if 'cpu_affinity' in config:
        require(result['cpu_affinity']==config['cpu_affinity'], 'CPU affinity differs from approved configuration')
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader'],text=True)
    result['topology']=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
    for name in sorted({t['environment'] for t in tasks_for(config)}):
        task=next(t for t in tasks_for(config) if t['environment']==name)
        cmd=[config['python'][name],str(BUNDLE/'check_environment.py'),name]
        output=subprocess.check_output(cmd,env=environment(task,config),cwd=BUNDLE/'vendor/oxygen',text=True)
        result['environments'][name]=json.loads(output.strip().splitlines()[-1])
        commands=[]
        for selected in tasks_for(config):
            if selected['environment']==name:
                cmd=command(selected,config);start=cmd.index(str(BUNDLE/'runners'/selected['script']))
                commands.append(cmd[start:])
        subprocess.run([config['python'][name],str(BUNDLE/'check_commands.py')],input=json.dumps(commands),text=True,env=environment(task,config),cwd=BUNDLE/'vendor/oxygen',check=True,stdout=subprocess.DEVNULL)

    return result

def idle(config):
    devices=','.join(config.get('exclusive_devices',config['devices']))
    output=subprocess.check_output(['nvidia-smi','-i',devices,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    require(not output,'Exclusive GPU set has running compute processes: '+output)

def products(task,root):
    path=Path(root)/task['output']; paths=[path]
    if task['kind']=='longrun': paths.extend(sorted(path.parent.glob('*.jsonl')))
    if task['kind']=='xiaomi': paths.append(path.with_suffix('.jsonl'))
    if task['kind']=='pi_reference': paths.extend(path.parent/n for n in read(path)['files'])
    return {str(p.relative_to(root)):digest(p) for p in paths if p.exists()}

def archive_attempt(task,root,attempt):
    root=Path(root); path=root/task['output']
    paths=[path]
    if task['kind']=='xiaomi': paths.append(path.with_suffix('.jsonl'))
    if task['kind']=='longrun': paths=list(path.parent.glob('*.json'))+list(path.parent.glob('*.jsonl'))
    if task['kind']=='pi_reference': paths.append(path.parent/'raw')
    for p in paths:
        if p.exists():
            target=root/'attempts'/task['id']/attempt/p.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(p),str(target))

def execute(config,resume):
    root=Path(config['output']).resolve(); root.mkdir(parents=True,exist_ok=True)
    # Shared lock between all invocations on this host, independent of output dir.
    lock=open(Path('/tmp')/f'oxygen-camera-ready-{os.getuid()}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle(config)
    proof=preflight(config)
    run_spec={'config':config,'protocol':read(BUNDLE/'protocol.json'),'environment':proof}
    manifest=root/'run.json'
    if manifest.exists():
        require(resume,'Output already belongs to a run; use --resume or choose a new output directory.')
        require(read(manifest)==run_spec,'Resume configuration, machine, code, weights or environment differs.')
    else:
        require(not resume,'Cannot resume without run.json')
        write(manifest,run_spec)
    failures=[]
    for task in tasks_for(config):
        status=root/'status'/f"{task['id']}.json"
        if resume and status.exists():
            old=read(status)
            if old.get('ok'):
                validate(task,root)
                require(products(task,root)==old['products'],'Completed output changed: '+task['id'])
                print('SKIP verified:',task['id'],flush=True); continue
        idle(config)
        out=root/task['output']; out.parent.mkdir(parents=True,exist_ok=True)
        attempt=time.strftime('%Y%m%dT%H%M%S')+'-'+str(time.time_ns())
        archive_attempt(task,root,attempt)
        log=root/'logs'/f"{task['id']}-{attempt}.log"; log.parent.mkdir(exist_ok=True)
        cmd=command(task,config)
        info={'command':cmd,'cwd':str(BUNDLE/'vendor/oxygen'),'started':attempt,'log':str(log),'ok':False}
        write(status,info)
        print('RUN',task['id'],shlex.join(cmd),flush=True)
        try:
            with log.open('w') as stream:
                subprocess.run(cmd,cwd=BUNDLE/'vendor/oxygen',env=environment(task,config),stdout=stream,stderr=subprocess.STDOUT,check=True)
            validate(task,root)
            info.update(ok=True,products=products(task,root))
        except (subprocess.CalledProcessError,ValueError,KeyError,AssertionError,FileNotFoundError) as exc:
            info['error']=str(exc); failures.append(task['id'])
            print('FAILED',task['id'],'see',log,flush=True)
        finally:
            info['finished']=time.strftime('%Y%m%dT%H%M%S'); write(status,info)
    require(not failures,'Incomplete suite: '+', '.join(failures))
    publish(config)

def publish(config):
    root=Path(config['output'])
    for task in tasks_for(config):
        validate(task,root)
        status=root/'status'/f"{task['id']}.json"
        require(status.exists() and read(status).get('ok'),'Task not marked complete: '+task['id'])
        require(products(task,root)==read(status)['products'],'Raw output changed: '+task['id'])
    subprocess.run([sys.executable,str(BUNDLE/'report.py'),'--root',str(root),'--tp2' if config['include_tp2'] else '--single-gpu'],check=True)
    env=os.environ.copy(); env['PYTHONPATH']=str(REPO)
    subprocess.run([config['python'].get('plot',config['python']['pi']),'-m','experiments.analysis.plot_camera_ready_appendix','--results-root',str(root),'--output-dir',str(root/'paper/figures')],env=env,cwd=REPO,check=True)
    print('Complete:',root/'paper',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','seal-assets','preflight','run','report'])
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); config=read(a.config)
    if 'cpu_affinity' in config:
        os.sched_setaffinity(0,config['cpu_affinity'])
    if a.action=='plan':
        for task in tasks_for(config): print(task['id']+': '+shlex.join(command(task,config)))
        print(f"{len(tasks_for(config))} tasks; TP2 {'included' if config['include_tp2'] else 'explicitly excluded on single GPU'}")
    elif a.action=='seal-assets': assets(config,create=True)
    elif a.action=='preflight': print(json.dumps(preflight(config),indent=2))
    elif a.action=='run': execute(config,a.resume)
    elif a.action=='report':
        source_check()
        publish(config)

if __name__=='__main__': main()
