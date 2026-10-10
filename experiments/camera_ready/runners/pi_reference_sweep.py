"""Run the frozen official driver for a complete paired LIBERO reference sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',required=True);p.add_argument('--random-init',action='store_true');p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from random_init_support import pi_checkpoint
    a.checkpoint_dir = pi_checkpoint(a.checkpoint_dir, a.random_init)
    folder=a.output.parent/'raw'
    if folder.exists():
        shutil.move(str(folder),str(folder.with_name('previous-'+str(time.time_ns()))))
    subprocess.run([sys.executable,'-m','experiments.run_experiments','--settings','baseline','shared_kv','continuous_batching','--policies','pi05_o2_libero','--checkpoint-dir',a.checkpoint_dir,'--results-dir',str(folder),'--gpu',a.gpu,'--prompt','pick the red cup','--num-denoise-steps','10','--max-decoding-steps','5','10','15','20','30','--steps-per-frame','1','5','10','--total-frames','50','--arrival-pattern','uniform_arrivals(rate=1)','--num-measured-runs','10','--warmup-runs','2'],check=True)
    files={str(p.relative_to(a.output.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*.json'))}
    a.output.write_text(json.dumps({'files':files},indent=2)+'\n')

if __name__=='__main__': main()
