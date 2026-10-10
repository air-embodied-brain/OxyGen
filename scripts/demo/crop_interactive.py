"""Extract unannotated, synchronized views from the existing comparison films.

The source painter places 868 x 651 views at (64,144) and (988,144).
Keep their shared gutter; trim one bottom pixel for H.264's even dimensions.
Frequent keyframes make dragging the web timeline responsive.
"""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
for name in ('white-mug-and-chocolate-pudding', 'alphabet-soup-and-tomato-sauce', 'moka-pots'):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(root / 'public/videos' / f'{name}.mp4'),
                    '-vf', 'crop=1792:650:64:144', '-an', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '20', '-g', '15', '-movflags', '+faststart',
                    str(root / 'public/videos' / f'{name}-interactive.mp4')], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(root / 'public/videos' / f'{name}-interactive.mp4'),
                    '-frames:v', '1', str(root / 'public/posters' / f'{name}-interactive.webp')], check=True)
