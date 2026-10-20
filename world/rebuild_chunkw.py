"""Worldwide corpus chunk build. Builds days [a:b) from alld.json at the worldwide grid,
uploads one tar per chunk to the given release tag, then a chunk-N.done marker.
Usage: VD_POINTS=... VD_CELLS=... python3 rebuild_chunkw.py <alld.json> <a> <b> <outdir> <tag> <chunk>"""
import os,sys,json,subprocess
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import concurrent.futures as cf
import vdlibw as v
LEAD=int(os.environ.get('VD_LEAD','1'))
alld=json.load(open(sys.argv[1]));a=int(sys.argv[2]);b=int(sys.argv[3])
outdir=sys.argv[4];tag=sys.argv[5];chunk=sys.argv[6]
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
os.makedirs(outdir,exist_ok=True)
pool=cf.ThreadPoolExecutor(48)
built=[];errs=[]
for date in alld[a:b]:
    try:
        arr,nb=v.fetch_gfs_day(date,pool,LEAD);f,raw=v.assemble(date,arr);t=v.fetch_era_day(date)
        if not np.isfinite(t).all():raise ValueError('non-finite ERA5')
    except Exception as e:
        errs.append((date,repr(e)[:150]));print('ERR',date,repr(e)[:150],flush=True);continue
    p=os.path.join(outdir,f'{date}.npz')
    np.savez_compressed(p+'.tmp.npz',F=f,RR=raw,T=t.astype(np.float32));os.replace(p+'.tmp.npz',p)
    built.append(p)
    if len(built)%8==0:print('built',len(built),'errs',len(errs),flush=True)
if built:
    tarf=os.path.join(outdir,f'world-chunk-{chunk}.tar.gz')
    subprocess.run(['tar','czf',tarf,'-C',outdir]+[os.path.basename(p) for p in built],check=True)
    print('tar',tarf,os.path.getsize(tarf),flush=True)
    r=subprocess.run(['gh','release','upload',tag,tarf,'--clobber','--repo',REPO],capture_output=True,text=True)
    if r.returncode:raise SystemExit('upload failed: '+r.stderr[:300])
marker=os.path.join(outdir,f'chunk-{chunk}.done')
open(marker,'w').write(json.dumps({'built':len(built),'lead':LEAD,'errors':errs}))
r=subprocess.run(['gh','release','upload',tag,marker,'--clobber','--repo',REPO],capture_output=True,text=True)
if r.returncode:raise SystemExit('marker upload failed: '+r.stderr[:300])
print('CHUNK DONE',chunk,'built',len(built),'errors',len(errs),json.dumps(errs)[:400],flush=True)
if errs:raise SystemExit(2)
