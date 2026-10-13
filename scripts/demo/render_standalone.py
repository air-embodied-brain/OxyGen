"""CLI-only still/full-film compositor. Default: three review PNGs only."""
import argparse, hashlib, io, json, math, re, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SITE = HERE.parents[1]  # website/scripts/demo/render_standalone.py
DATA = json.loads((SITE / 'public/data/demo-timelines.json').read_text())
SITE_TEXT = (SITE / 'src/data/site.ts').read_text()
TASKS = {a: (b,c) for a,b,c in re.findall(r'id: "([^"]+)",\s+title: "([^"]+)",\s+task: "([^"]+)"', SITE_TEXT)}
SAMPLES = dict(zip(DATA, [15.5,17.8,27.0]))
C = dict(paper='#fffefa', deep='#f3f1eb', ink='#292825', soft='#454440', muted='#696760', hair='#dedbd2', action='#315f78', language='#a33a31', move='#8b877f')
FONT_FILES = {'sans': SITE/'public/fonts/SourceSans3VF-Upright.ttf.woff2', 'display': SITE/'public/fonts/LibreBaskerville-VariableFont_wght.woff2', 'serif': SITE/'node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-wght-normal.woff2'}
FONTS = {}
def font(family,size,weight=400):
    key=(family,size,weight)
    if key not in FONTS:
        f=ImageFont.truetype(str(FONT_FILES[family]),size)
        axes=f.get_variation_axes()
        f.set_variation_by_axes([weight if a['name']==b'Weight' else a['default'] for a in axes])
        FONTS[key]=f
    return FONTS[key]

def state(run,t,dt):
    seg=next((s for s in run['segments'] if s['start']<=t<s['end']),None)
    step=min(run['lastStep'],seg['step']+(int((t-seg['start'])/dt)+1 if seg['kind']=='move' else 0)) if seg else run['lastStep']
    elapsed=sum(max(0,min(s['end'],t)-s['start']) for s in run['segments'] if s['kind']!='move')
    memory=next((m for m in reversed(run['memory']) if m['time']<=t),None)
    return dict(kind=seg['kind'] if seg else 'move',phase={'action':'Action inference','language':'Language inference','move':'Executing actions'}[seg['kind']] if seg else 'Task complete',step=step,inferenceElapsed=elapsed,output=memory['text'] if memory else 'Waiting for completed output',outputTime=memory['time'] if memory else None)

def compose(slug,t,footage):
    r=DATA[slug];t=min(t,r['total']);im=Image.new('RGB',(1920,1080),C['paper']);d=ImageDraw.Draw(im)
    def text(x,y,s,size=32,color='ink',family='sans',weight=400,anchor='mt'):
        d.text((x,y),s,font=font(family,size,weight),fill=C.get(color,color),anchor=anchor)
    text(960,24,TASKS[slug][0],42,family='display')
    text(960,85,TASKS[slug][1],32,'soft')
    d.line((960,140,960,870),fill=C['hair'],width=2)
    for i,side in enumerate(('oxygen','baseline')):
        x=72+i*912;cx=x+432;run=r[side];st=state(run,t,r['controlDt']);accent='action' if i==0 else 'move'
        text(cx,134,'OxyGen' if i==0 else 'Isolated baseline',38,accent if i==0 else 'ink',weight=650)
        infer=sum(s['end']-s['start'] for s in run['segments'] if s['kind']!='move')
        text(cx,180,f'Inference {infer:.1f} s  ·  End-to-end {run["duration"]:.1f} s',30,'muted')
        view=footage.crop((0 if i==0 else 924,0,868 if i==0 else 1792,650)).resize((640,479),Image.Resampling.LANCZOS)
        im.paste(view,(cx-320,224))
        text(x,719,st['phase'],32,st['kind'],weight=600,anchor='lt')
        text(x+864,721,f'{st["inferenceElapsed"]:.2f} s inference · Step {st["step"]}',30,'muted',anchor='rt')
        d.rounded_rectangle((x,765,x+864,875),radius=6,fill=C['deep'])
        d.rectangle((x,765,x+5,875),fill=C[accent])
        words=st['output'].split();lines=['']
        for word in words:
            trial=(lines[-1]+' '+word).strip()
            if d.textlength(trial,font=font('serif',34))>816:lines.append(word)
            else:lines[-1]=trial
        assert len(lines)<=2,(slug,st['output'])
        # Center the actual ink bounding box, not the font's ascender box.
        block='\n'.join(lines);f=font('serif',34)
        box=d.multiline_textbbox((0,0),block,font=f,spacing=5,align='center')
        d.multiline_text((cx-(box[2]-box[0])/2-box[0],820-(box[3]-box[1])/2-box[1]),block,font=f,fill=C['soft'],spacing=5,align='center')
    text(72,895,f'Replay  {t:.2f} / {r["total"]:.2f} s',32,weight=600,anchor='lt')
    for x,kind,label in [(748,'action','Action inference'),(1120,'language','Language inference'),(1550,'move','Execution')]:
        d.rectangle((x,905,x+14,919),fill=C[kind]);text(x+26,895,label,29,'muted',anchor='lt')
    left=250;width=1598
    for i,side in enumerate(('oxygen','baseline')):
        y=942+i*36;text(72,y-4,'OxyGen' if i==0 else 'Baseline',29,'muted',anchor='lt')
        d.rectangle((left,y,left+width,y+23),fill=C['deep'])
        for s in r[side]['segments']:
            d.rectangle((left+s['start']/r['total']*width,y,left+s['end']/r['total']*width,y+23),fill=C[s['kind']])
    head=left+t/r['total']*width
    d.line((head,936,head,1004),fill=C['paper'],width=6);d.line((head,936,head,1004),fill=C['ink'],width=2)
    for tick in [v for v in range(0,int(r['total']),5) if r['total']-v>=2]+[r['total']]:
        text(left+tick/r['total']*width,1005,f'{tick:g}' if tick!=r['total'] else f'{tick:.2f} s',24,'muted',anchor='mt' if tick!=r['total'] else 'rt')
    text(960,1045,'LIBERO-10 · π0.5 · RTX 4090 · k = 1   |   Logged inference; estimated subphases; 20 Hz execution',26,'muted')
    return im

def source(slug):return SITE/'public/videos'/f'{slug}-interactive.mp4'
def frame(slug,t):
    # Floor to the source's 30 fps clock, so footage never advances beyond annotations.
    stamp=math.floor(t*30)/30
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(stamp),'-i',str(source(slug)),'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','1','-'])
    return Image.open(io.BytesIO(raw)).convert('RGB')

def full(slug,out):
    print(f'Rendering {slug} -> {out}', flush=True)
    r=DATA[slug];n=math.floor((r['total']+2)*30)
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(source(slug)),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'],stdout=subprocess.PIPE)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    try:
        for i in range(n):
            raw=decoder.stdout.read(1792*650*3)
            if len(raw)!=1792*650*3:raise RuntimeError('Source footage ended early')
            encoder.stdin.write(compose(slug,i/30,Image.frombytes('RGB',(1792,650),raw)).tobytes())
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Encoder failed')
        print(f'Finished {slug}: {n} frames', flush=True)
    finally:
        decoder.stdout.close();decoder.terminate();decoder.wait()
        if encoder.poll() is None:encoder.terminate();encoder.wait()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['stills','video'],default='stills');p.add_argument('--task',choices=['all',*DATA],default='all');p.add_argument('--time',type=float);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);manifest={}
    for slug in DATA if a.task=='all' else [a.task]:
        if a.mode=='video':full(slug,a.output/(slug+'.mp4'));continue
        t=SAMPLES[slug] if a.time is None else a.time
        assert 0<=t<=DATA[slug]['total']
        dest=a.output/(slug+'.png');compose(slug,t,frame(slug,t)).save(dest)
        manifest[slug]={'sampleTime':t,'sourceFrame':math.floor(t*30),'source':str(source(slug)),'preview':str(dest),'states':{s:state(DATA[slug][s],t,DATA[slug]['controlDt']) for s in ['oxygen','baseline']}}
        print(dest)
    if a.mode=='stills':(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
