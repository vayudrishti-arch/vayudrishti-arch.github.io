"""Backtrack backfill part 2: raw GFS 48h/72h from the world corpus.
Streams each world chunk tar once (lead-2 and lead-3 releases), extracts ONLY the RR
row at each site's nearest world-grid point (precomputed from pointsw1m.json), and
extends app/history/<site>/<year>.json rows with raw_gfs_48h_mm / raw_gfs_72h_mm.
Only dates already present in the year files are extended; missing days stay gaps.
Corrected 48/72h before 2026-10-22 remains honestly 'not archived'."""
import os,json,io,time,math,tarfile,zipfile,urllib.request,subprocess
import numpy as np

REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W=os.environ.get('VD_WORK','/tmp/vdl23');os.makedirs(W,exist_ok=True)
def log(*a):print(*a,flush=True)
def ghapi(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'vd-l23'}),timeout=120))
def get(url,tries=5):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'vd-l23'}),timeout=120).read()
        except Exception:
            if t==tries-1:raise
            time.sleep(4*(t+1))

SITES=[["pune",18.51957,73.85535],["mumbai",19.076,72.8777],["delhi",28.6139,77.209],["chennai",13.0827,80.2707],
["srinagar",34.0837,74.7973],["shimla",31.1048,77.1734],["chandigarh",30.7333,76.7794],["jaipur",26.9124,75.7873],
["jodhpur",26.2389,73.0243],["ahmedabad",23.0225,72.5714],["bhopal",23.2599,77.4126],["nagpur",21.1458,79.0882],
["raipur",21.2514,81.6296],["lucknow",26.8467,80.9462],["patna",25.5941,85.1376],["kolkata",22.5726,88.3639],
["bhubaneswar",20.2961,85.8245],["guwahati",26.1445,91.7362],["shillong",25.5788,91.8933],["imphal",24.817,93.9368],
["hyderabad",17.385,78.4867],["bengaluru",12.9716,77.5946],["thiruvananthapuram",8.5241,76.9366],["port-blair",11.6234,92.7265]]

# world grid points (1M: main 980,692 + ext 19,308)
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
import gzip
open(f'{W}/pt.json.gz','wb').write(get(wurls['pointsw1m.json.gz']))
pts=json.load(gzip.open(f'{W}/pt.json.gz'))
plat=np.array([p['latitude'] for p in pts]);plon=np.array([p['longitude'] for p in pts])
del pts
N_MAIN=980692
def site_idx(la,lo):
    # nearest in main grid and in ext grid; return (index, distance_deg)
    best=None
    for lo_,hi_ in ((0,N_MAIN),(N_MAIN,len(plat))):
        d=(plat[lo_:hi_]-la)**2+(plon[lo_:hi_]-lo)**2
        i=int(d.argmin())+lo_
        if best is None or d.min()<best[1]:best=(i,float(d.min()))
    return best[0],math.sqrt(best[1])
SIDX=[];SDIST=[]
for name,la,lo in SITES:
    i,d=site_idx(la,lo);SIDX.append(i);SDIST.append(round(d,3))
log('site->world grid:',list(zip([s[0] for s in SITES],SIDX,SDIST))[:3],'...')

# day -> chunk map (needed only to know which chunks exist; we stream ALL chunks anyway)
alld=json.loads(get(wurls['alld_world.json']).decode())
LAGGED={'2026-10-10','2026-10-11'}
full_days=[d for d in alld if d not in LAGGED]
log('world corpus days:',len(full_days),full_days[0],'..',full_days[-1])

series={2:{name:{} for name,_,_ in SITES},3:{name:{} for name,_,_ in SITES}}
for lead,tag in ((2,'data-days-world-l2'),(3,'data-days-world-l3')):
    rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}')
    chunks=sorted(a['browser_download_url'] for a in rel['assets'] if a['name'].startswith('world-chunk-') and a['name'].endswith('.tar.gz'))
    log(tag,'chunks:',len(chunks))
    t0=time.time()
    for u in chunks:
        fn=f"{W}/{u.rsplit('/',1)[-1]}"
        if not os.path.exists(fn):
            open(fn,'wb').write(get(u))
        tf=tarfile.open(fn,mode='r:gz')
        for m in tf.getmembers():
            d=m.name[:10]
            z=zipfile.ZipFile(io.BytesIO(tf.extractfile(m).read()))
            RR=np.load(io.BytesIO(z.read('RR.npy')))
            for k,(name,_,_) in enumerate(SITES):
                series[lead][name][d]=round(float(RR[SIDX[k]]),2)
        tf.close();os.remove(fn)
        log(tag,'done',u.rsplit('/',1)[-1],'elapsed',int(time.time()-t0),'s')

# merge into app/history/<site>/<year>.json (extend existing dates only)
OUT=os.path.join(os.getcwd(),'app','history')
extended=0
for k,(name,la,lo) in enumerate(SITES):
    for y in os.listdir(os.path.join(OUT,name)):
        if y=='originals' or not y.endswith('.json'):continue
        fp=os.path.join(OUT,name,y)
        doc=json.load(open(fp))
        if len(doc['columns'])==8:continue   # already extended
        assert doc['columns']==['date','raw_gfs_24h_mm','corrected_24h_mm_replay','observed_rain_mm_era5','observed_tmax_c_era5','observed_tmin_c_era5'],doc['columns']
        doc['columns']=doc['columns']+['raw_gfs_48h_mm','raw_gfs_72h_mm']
        for r in doc['rows']:
            v2=series[2][name].get(r[0]);v3=series[3][name].get(r[0])
            r+=[v2,v3]
            if v2 is not None or v3 is not None:extended+=1
        json.dump(doc,open(fp,'w'),separators=(',',':'))
# index labels
idxp=os.path.join(OUT,'index.json')
idx=json.load(open(idxp))
idx['labels']['raw_48_72']='Raw GFS 48h/72h forecast for that day from the worldwide archived corpus (nearest world-grid point). Corrected 48h/72h was not archived before 2026-10-22.'
idx['generated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
json.dump(idx,open(idxp,'w'),indent=1)
log('DONE extended rows:',extended)
