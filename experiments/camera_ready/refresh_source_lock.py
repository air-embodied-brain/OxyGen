"""Maintainer-only: regenerate the reviewed source manifest after intentional edits."""
import hashlib
import json
from pathlib import Path

bundle=Path(__file__).resolve().parent
repo=bundle.parents[1]
files={}
for p in sorted(bundle.rglob('*')):
    if not p.is_file() or p.name=='source-lock.json' or '.venv' in p.parts or '__pycache__' in p.parts or p.suffix=='.pyc':continue
    files[str(p.relative_to(repo))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['experiments/analysis/plot_camera_ready_appendix.py','experiments/analysis/plot_utils.py']:
    files[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
(bundle/'source-lock.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
print('Locked',len(files),'files')
