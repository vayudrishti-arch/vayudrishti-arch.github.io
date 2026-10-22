"""VayuDrishti backtrack gap fill: appends MISSING days to app/history/<site>/<year>.json.
raw 24h + 24h replay: GFS grib (NOAA S3 archive) at the nearest 9,112-point India grid cell,
  identical fetch math to the corpus pipeline (vdlib); replay = current live 24h model on those
  inputs, same as the main backfill.
raw 48/72h: same grib at the nearest world-grid (1M points) cell, leads 2/3 - same source the
  48/72 backfill used (world corpus day files), so conventions match.
observed rain/temp: Open-Meteo archive API with models=era5 ONLY (final ERA5). Days ERA5 has not
  published yet stay null and are filled by the nightly archiver once ERA5 publishes. The seamless
  (IFS-blended) tail is never written as observed.
Existing rows are never overwritten. Runs in Actions (history-gapfill.yml)."""
import os,sys,json,io,math,time,datetime as dt,urllib.request,gzip,threading
import numpy as np
import concurrent.futures
import eccodes

REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
TOK=os.environ.get('GH_TOKEN','')
W='/tmp/vdgap';os.makedirs(W,exist_ok=True)
START,END=sys.argv[1],sys.argv[2]
BUNDLE='https://github.com/user-attachments/files/33634084/1-rebuild_bundle2.tar.gz'
SITES=[["pune",18.51957,73.85535],["mumbai",19.076,72.8777],["delhi",28.6139,77.209],["chennai",13.0827,80.2707],
["srinagar",34.0837,74.7973],["shimla",31.1048,77.1734],["chandigarh",30.7333,76.7794],["jaipur",26.9124,75.7873],
["jodhpur",26.2389,73.0243],["ahmedabad",23.0225,72.5714],["bhopal",23.2599,77.4126],["nagpur",21.1458,79.0882],
["raipur",21.2514,81.6296],["lucknow",26.8467,80.9462],["patna",25.5941,85.1376],["kolkata",22.5726,88.3639],
["bhubaneswar",20.2961,85.8245],["guwahati",26.1445,91.7362],["shillong",25.5788,91.8933],["imphal",24.817,93.9368],
["hyderabad",17.385,78.4867],["bengaluru",12.9716,77.5946],["thiruvananthapuram",8.5241,76.9366],["port-blair",11.6234,92.7265]]
def log(*a):print(*a,flush=True)
def get(url,rng=None,tries=4):
    for t in range(tries):
        try:
            req=urllib.request.Request(url,headers={'Range':rng} if rng else {'User-Agent':'vd-gapfill'})
            with urllib.request.urlopen(req,timeout=40) as r:
                if rng and r.status!=206:raise ValueError('Range not honored')
                return r.read()
        except Exception:
            if t==tries-1:raise
            time.sleep(2*(t+1))
def ghapi(url):
    req=urllib.request.Request(url,headers={'User-Agent':'vd-gapfill','Authorization':'Bearer '+TOK} if TOK else {'User-Agent':'vd-gapfill'})
    return json.loads(urllib.request.urlopen(req,timeout=60).read())

# ---------- grids ----------
import tarfile,subprocess
subprocess.run(['curl','-sL','-o',f'{W}/rb.tgz',BUNDLE],check=True)
tarfile.open(f'{W}/rb.tgz').extractall(f'{W}/rb')
p9=json.load(open(f'{W}/rb/points9112.json'))
lat9=np.array([p['latitude'] for p in p9]);lon9=np.array([p['longitude'] for p in p9])
c9=json.load(open(f'{W}/rb/cells9112.json'))
I13_9=np.array(c9['row13'])*3072+np.array(c9['col13']);I25_9=np.array(c9['row25'])*1440+np.array(c9['col25'])
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
open(f'{W}/pw.gz','wb').write(get(wurls['pointsw1m.json.gz']));pw=json.load(gzip.open(f'{W}/pw.gz'))
latw=np.array([p['latitude'] for p in pw]);lonw=np.array([p['longitude'] for p in pw])
open(f'{W}/cw.gz','wb').write(get(wurls['cellsw1m.json.gz']));cw=json.load(gzip.open(f'{W}/cw.gz'))
I13_w=np.array(cw['row13'])*3072+np.array(cw['col13']);I25_w=np.array(cw['row25'])*1440+np.array(cw['col25'])
N_MAIN=980692
def nearest(lat,lon,la,lo,ext_ok=False):
    best=None
    for a,b in ((0,N_MAIN),(N_MAIN,len(lat))):
        d=(lat[a:b]-la)**2+(lon[a:b]-lo)**2;i=int(d.argmin())+a
        if best is None or d.min()<best[1]:best=(i,float(d.min()))
    return best[0]
def nearest9(la,lo):
    d=(lat9-la)**2+(lon9-lo)**2;return int(d.argmin())
sidx9=[nearest9(la,lo) for _,la,lo in SITES]
sidxw=[nearest(latw,lonw,la,lo,True) for _,la,lo in SITES]
log('site->9112:',sidx9[:3],'... site->world:',sidxw[:3],'...')
SLAT9=lat9[sidx9];SLON9=lon9[sidx9];SELEV9=np.array([p9[i]['elevation'] for i in sidx9])
SREG9=[p9[i]['region'] for i in sidx9]

# ---------- live 24h model (for the labeled replay) ----------
html=get('https://vayudrishti-arch.github.io/app/').decode()
i=html.find('const ENSEMBLE=')+len('const ENSEMBLE=');d=0
for p in range(i,len(html)):
    if html[p]=='{':d+=1
    elif html[p]=='}':
        d-=1
        if d==0:break
E=json.loads(html[i:p+1]);M1=[m for m in E['models'] if m['lead']==1][0]
log('24h replay model:',E['version'])
def tree_predict_row(trees,x,base):
    y=float(base)
    for t in trees:
        lc=t['left_children'];rc=t['right_children'];si=t['split_indices'];sc=t['split_conditions'];dl=t['default_left']
        n=0
        for _ in range(15):
            l=lc[n]
            if l==-1:break
            v=x[si[n]]
            left=(dl[n] if v!=v else v<sc[n])
            n=l if left else rc[n]
        y+=sc[n]
    return y
def replay24(F,RR):
    out=np.zeros(len(RR),np.float32)
    for k in range(len(RR)):
        raw=float(RR[k])
        if raw==0:out[k]=0;continue
        mp=M1['qm'].get(SREG9[k]) or M1['qm']['global']
        qm=float(np.interp(raw,mp['source'],mp['target']))
        x=list(F[k])+[raw,qm,math.log1p(raw),math.log1p(qm)]
        a=max(0.0,qm+tree_predict_row(M1['trees'],x,M1['base']))
        if M1.get('clf'):
            pr=1/(1+math.exp(-tree_predict_row(M1['clf']['trees'],x,M1['clf']['base'])))
            if pr>=M1['clf']['threshold']:a=max(a,64.5)
        out[k]=a
    return out

# ---------- grib fetch (vdlib step_job, parameterized) ----------
lock=threading.Lock()
FIELDS=['temperature_k','specific_humidity','surface_pressure_pa','u10','v10','precipitation_mm','showers_mm','frozen_percent','pressure_msl_hpa','cape']
NEED0=['TMP:2 m above ground:','SPFH:2 m above ground:','PRES:surface:','UGRD:10 m above ground:','VGRD:10 m above ground:','PRATE:surface:','CPRAT:surface:','CPOFP:surface:']
NEED1=['MSLET:mean sea level:','CAPE:surface:']
def indexed(base):return [(int(s.split(':')[1]),':'.join(s.split(':')[3:])) for s in get(base+'.idx').decode().splitlines()]
def extract(base,index,prefix,idx):
    i=next(i for i,x in enumerate(index) if x[1].startswith(prefix));b=get(base,f'bytes={index[i][0]}-{index[i+1][0]-1}')
    with lock:g=eccodes.codes_new_from_message(b);v=eccodes.codes_get_values(g)[idx].astype(np.float32);eccodes.codes_release(g)
    return v,index[i][1]
def layout(date):return 'atmos/' if date>='2021-03-23' else ''
def step_job(date,hour,lead,idx13,idx25,precip_only):
    utc=dt.datetime.fromisoformat(date)+dt.timedelta(hours=hour-5)
    run=utc.replace(hour=utc.hour//6*6)-dt.timedelta(days=lead)
    step=int((utc-run).total_seconds()/3600);arrays={}
    want0=NEED0[5:] if precip_only else NEED0
    jobs=[(f'sfluxgrbf{step:03d}.grib2',want0,idx13)]+([] if precip_only else [(f'pgrb2.0p25.f{step:03d}',NEED1,idx25)])
    for kind,nn,idx in jobs:
        base=f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/{layout(f"{run:%Y-%m-%d}")}gfs.t{run:%H}z.{kind}'
        index=indexed(base);previndex=None
        for prefix in nn:
            v,desc=extract(base,index,prefix,idx)
            if prefix.startswith(('PRATE:','CPRAT:')):
                a,b=map(int,desc.split(':')[-2].split(' ')[0].split('-'))
                if b-a>1:
                    pb=base.replace(f'sfluxgrbf{step:03d}',f'sfluxgrbf{step-1:03d}');previndex=previndex or indexed(pb)
                    prev,_=extract(pb,previndex,prefix,idx);v=v*np.float32(b-a)-prev*np.float32(b-a-1)
                v=np.maximum(v,0)*np.float32(3600)
            if prefix.startswith('MSLET:'):v/=np.float32(100)
            arrays[FIELDS[(NEED0+NEED1).index(prefix)]]=v
    return hour,arrays
def fetch_day(date,pool,lead,idx13,idx25,precip_only):
    res=list(pool.map(lambda h:step_job(date,h,lead,idx13,idx25,precip_only),range(24)))
    return {k:np.stack([r[1][k] for r in sorted(res,key=lambda r:r[0])]) for k in (FIELDS[5:8] if precip_only else FIELDS)}
q=lambda a,scale:np.copysign(np.floor(abs(a)*scale+.5),a)/scale
def raw_only(a):
    prec=q(a['precipitation_mm'],10);show=q(a['showers_mm'],10)
    sf=np.clip(a['frozen_percent']/100*a['precipitation_mm'],0,a['precipitation_mm'])
    sf=np.where((a['frozen_percent']>=50)&(prec>=.21)&(prec-q(sf,10)<=.11),a['precipitation_mm'],sf)
    snow=q(sf,10);rain=np.maximum(prec-snow-show,0)
    return (rain+show).sum(axis=0)
def assemble(date,a):
    T=a['temperature_k'];p=a['surface_pressure_pa'];spec=a['specific_humidity'];f=np.clip((T-250.16)/23,0,1)**2
    esi=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-.00728332*T);esw=np.exp(54.842763-6763.22/T-4.210*np.log(T)+.000367*T+np.tanh(.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+.014025*T))
    rh=q(np.clip(100*spec*p/(.622+.378*spec)/(f*esw+(1-f)*esi),0,100),1)
    pr=q(a['pressure_msl_hpa'],10);cape=q(a['cape'],.1);u=q(a['u10'],10);v=q(a['v10'],10)
    wind=q(np.hypot(u,v)*3.6,10);direction=q((np.degrees(np.arctan2(u,v))+180)%360,1)
    raw=raw_only(a);mo=int(date[5:7])
    feats=np.stack([a['precipitation_mm'].sum(0),a['precipitation_mm'].max(0),(a['precipitation_mm']>=.1).sum(0),cape.mean(0),cape.max(0),rh.mean(0),rh.min(0),pr.mean(0),pr.min(0),np.ptp(pr,axis=0),pr[23]-pr[0],wind.mean(0),wind.max(0),(wind*np.cos(direction*np.pi/180)).mean(0),(wind*np.sin(direction*np.pi/180)).mean(0),SLAT9,SLON9,SELEV9,np.full(24,np.sin(2*np.pi*mo/12)),np.full(24,np.cos(2*np.pi*mo/12))],axis=-1).astype(np.float32)
    if not np.isfinite(feats).all()or (raw<0).any():raise ValueError('invalid derived data '+date)
    return feats,raw.astype(np.float32)

# ---------- observed (final ERA5 only) ----------
def observed(name,la,lo):
    url=('https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s'
         '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%%2FKolkata&models=era5')%(la,lo,START,END)
    j=json.loads(get(url))['daily'];out={}
    for t,tmx,tmn,ps in zip(j['time'],j['temperature_2m_max'],j['temperature_2m_min'],j['precipitation_sum']):
        out[t]=(None if ps is None else round(ps,2),None if tmx is None else round(tmx,1),None if tmn is None else round(tmn,1))
    return out

# ---------- main ----------
d0=dt.date.fromisoformat(START);d1=dt.date.fromisoformat(END)
days=[];d=d0
while d<=d1:days.append(d.isoformat());d+=dt.timedelta(1)
pool=concurrent.futures.ThreadPoolExecutor(16)
obs_by_site={}
for name,la,lo in SITES:
    try:obs_by_site[name]=observed(name,la,lo)
    except Exception as e:log('obs failed',name,e);obs_by_site[name]={}
    time.sleep(0.3)
new_rows={name:{} for name,_,_ in SITES}
for date in days:
    a1=fetch_day(date,pool,1,I13_9[sidx9],I25_9[sidx9],False)
    F,raw1=assemble(date,a1);corr=replay24(F,raw1)
    a2=fetch_day(date,pool,2,I13_w[sidxw],I25_w[sidxw],True);raw2=raw_only(a2)
    a3=fetch_day(date,pool,3,I13_w[sidxw],I25_w[sidxw],True);raw3=raw_only(a3)
    del a1,a2,a3
    for k,(name,la,lo) in enumerate(SITES):
        ob=obs_by_site[name].get(date,(None,None,None))
        new_rows[name][date]=[date,round(float(raw1[k]),2),round(float(corr[k]),2),ob[0],ob[1],ob[2],round(float(raw2[k]),2),round(float(raw3[k]),2)]
    log('built',date,'raw24/48/72 mumbai:',new_rows['mumbai'][date][1],new_rows['mumbai'][date][6],new_rows['mumbai'][date][7],'obs:',new_rows['mumbai'][date][3])
# ---------- merge (never overwrite) ----------
COLS=['date','raw_gfs_24h_mm','corrected_24h_mm_replay','observed_rain_mm_era5','observed_tmax_c_era5','observed_tmin_c_era5','raw_gfs_48h_mm','raw_gfs_72h_mm']
added=0
for name,_,_ in SITES:
    byyear={}
    for date,row in new_rows[name].items():byyear.setdefault(date[:4],[]).append(row)
    for y,rows in byyear.items():
        fp=os.path.join('app','history',name,y+'.json')
        if os.path.exists(fp):
            doc=json.load(open(fp))
            assert doc['columns']==COLS,fp
        else:
            doc={'site':name,'year':y,'columns':COLS,'rows':[]}
        have={r[0] for r in doc['rows']}
        add=[r for r in rows if r[0] not in have]
        if add:
            doc['rows']=sorted(doc['rows']+add)
            doc['gapfill_note']='Gap-filled by history/gapfill.py: raw GFS from the NOAA grib archive at the nearest network/world grid cell (same conventions as the corpus), 24h trained = replay of the current model, observed = final ERA5 via Open-Meteo (models=era5); days ERA5 has not published yet stay empty and fill in nightly.'
            json.dump(doc,open(fp,'w'),separators=(',',':'));added+=len(add)
log('DONE gapfill, rows added:',added)
