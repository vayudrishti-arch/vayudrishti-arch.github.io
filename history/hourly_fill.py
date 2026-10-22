"""VayuDrishti backtrack HOURLY fill: appends MISSING days to app/history6h/<site>/<year>.json.
Per day (same local-day convention as the daily rows, UTC+5), 24 hours, each hour:
  forecast raw GFS (NOAA grib archive, same fetch math as the corpus pipeline):
    lead 24h: rain mm, temp C, relative humidity %, wind km/h, cloud cover %
    lead 48h: rain mm, temp C
    lead 72h: rain mm, temp C
  observed (Open-Meteo archive API, models=era5 ONLY - final ERA5, ~5-day publish lag):
    rain mm, temp C, relative humidity %, wind km/h, cloud cover %
  Hours the final ERA5 has not published yet stay null and fill in nightly.
NO trained values: the learned correction works at full-day scale only.
Existing days are never overwritten. Runs in Actions (history6h-fill.yml)."""
import os,sys,json,io,math,time,datetime as dt,urllib.request,gzip,threading
import numpy as np
import concurrent.futures
import eccodes

REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
TOK=os.environ.get('GH_TOKEN','')
W='/tmp/vdh6';os.makedirs(W,exist_ok=True)
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
            req=urllib.request.Request(url,headers={'Range':rng} if rng else {'User-Agent':'vd-h6fill'})
            with urllib.request.urlopen(req,timeout=40) as r:
                if rng and r.status!=206:raise ValueError('Range not honored')
                return r.read()
        except Exception:
            if t==tries-1:raise
            time.sleep(2*(t+1))
def ghapi(url):
    req=urllib.request.Request(url,headers={'User-Agent':'vd-h6fill','Authorization':'Bearer '+TOK} if TOK else {'User-Agent':'vd-h6fill'})
    return json.loads(urllib.request.urlopen(req,timeout=60).read())

# ---------- grids ----------
import tarfile,subprocess
subprocess.run(['curl','-sL','-o',f'{W}/rb.tgz',BUNDLE],check=True)
tarfile.open(f'{W}/rb.tgz').extractall(f'{W}/rb')
p9=json.load(open(f'{W}/rb/points9112.json'))
lat9=np.array([p['latitude'] for p in p9]);lon9=np.array([p['longitude'] for p in p9])
c9=json.load(open(f'{W}/rb/cells9112.json'))
I13_9=np.array(c9['row13'])*3072+np.array(c9['col13']);I25_9=np.array(c9['row25'])*1440+np.array(c9['col25'])
try:
    wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
    wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
except Exception:
    wurls={n:f'https://github.com/{REPO}/releases/download/world-assets/{n}' for n in ('pointsw1m.json.gz','cellsw1m.json.gz')}
open(f'{W}/pw.gz','wb').write(get(wurls['pointsw1m.json.gz']));pw=json.load(gzip.open(f'{W}/pw.gz'))
latw=np.array([p['latitude'] for p in pw]);lonw=np.array([p['longitude'] for p in pw])
open(f'{W}/cw.gz','wb').write(get(wurls['cellsw1m.json.gz']));cw=json.load(gzip.open(f'{W}/cw.gz'))
I13_w=np.array(cw['row13'])*3072+np.array(cw['col13']);I25_w=np.array(cw['row25'])*1440+np.array(cw['col25'])
N_MAIN=980692
def nearest(lat,lon,la,lo):
    best=None
    for a,b in ((0,N_MAIN),(N_MAIN,len(lat))):
        d=(lat[a:b]-la)**2+(lon[a:b]-lo)**2;i=int(d.argmin())+a
        if best is None or d.min()<best[1]:best=(i,float(d.min()))
    return best[0]
def nearest9(la,lo):
    d=(lat9-la)**2+(lon9-lo)**2;return int(d.argmin())
sidx9=[nearest9(la,lo) for _,la,lo in SITES]
sidxw=[nearest(latw,lonw,la,lo) for _,la,lo in SITES]
log('grids ready')

# ---------- grib fetch (vdlib step_job, hourly output) ----------
lock=threading.Lock()
# field modes: FULL (lead1: rain parts+temp+rh+wind u/v + cloud), RT (leads 2/3: rain parts+temp)
F_FULL=['TMP:2 m above ground:','SPFH:2 m above ground:','PRES:surface:','UGRD:10 m above ground:','VGRD:10 m above ground:','PRATE:surface:','CPRAT:surface:','CPOFP:surface:']
P_FULL=['TCDC:entire atmosphere:']
F_RT=['TMP:2 m above ground:','PRATE:surface:','CPRAT:surface:','CPOFP:surface:']
def indexed(base):return [(int(s.split(':')[1]),':'.join(s.split(':')[3:])) for s in get(base+'.idx').decode().splitlines()]
def extract(base,index,prefix,idx):
    i=next(i for i,x in enumerate(index) if x[1].startswith(prefix));b=get(base,f'bytes={index[i][0]}-{index[i+1][0]-1}')
    with lock:g=eccodes.codes_new_from_message(b);v=eccodes.codes_get_values(g)[idx].astype(np.float32);eccodes.codes_release(g)
    return v,index[i][1]
def layout(date):return 'atmos/' if date>='2021-03-23' else ''
def step_job(date,hour,lead,idx13,idx25,mode):
    utc=dt.datetime.fromisoformat(date)+dt.timedelta(hours=hour-5)
    run=utc.replace(hour=utc.hour//6*6)-dt.timedelta(days=lead)
    step=int((utc-run).total_seconds()/3600);arrays={}
    nn=F_FULL if mode=='FULL' else F_RT
    jobs=[(f'sfluxgrbf{step:03d}.grib2',nn,idx13)]
    if mode=='FULL':jobs.append((f'pgrb2.0p25.f{step:03d}',P_FULL,idx25))
    for kind,want,idx in jobs:
        base=f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/{layout(f"{run:%Y-%m-%d}")}gfs.t{run:%H}z.{kind}'
        index=indexed(base);previndex=None
        for prefix in want:
            v,desc=extract(base,index,prefix,idx)
            if prefix.startswith(('PRATE:','CPRAT:')):
                a,b=map(int,desc.split(':')[-2].split(' ')[0].split('-'))
                if b-a>1:
                    pb=base.replace(f'sfluxgrbf{step:03d}',f'sfluxgrbf{step-1:03d}');previndex=previndex or indexed(pb)
                    prev,_=extract(pb,previndex,prefix,idx);v=v*np.float32(b-a)-prev*np.float32(b-a-1)
                v=np.maximum(v,0)*np.float32(3600)
            key=('TMP' if prefix.startswith('TMP:') else 'SPFH' if prefix.startswith('SPFH:') else
                 'UGRD' if prefix.startswith('UGRD:') else 'VGRD' if prefix.startswith('VGRD:') else
                 'PRATE' if prefix.startswith('PRATE:') else 'CPRAT' if prefix.startswith('CPRAT:') else
                 'CPOFP' if prefix.startswith('CPOFP:') else 'PRES' if prefix.startswith('PRES:') else 'TCDC')
            arrays[key]=v
    return hour,arrays
def fetch_day(date,pool,lead,idx13,idx25,mode):
    res=list(pool.map(lambda h:step_job(date,h,lead,idx13,idx25,mode),range(24)))
    keys=list(res[0][1].keys())
    return {k:np.stack([r[1][k] for r in sorted(res,key=lambda r:r[0])]) for k in keys}
def rain_hourly(a):
    prec=a['PRATE'];show=a['CPRAT']
    sf=np.clip(a['CPOFP']/100*a['PRATE'],0,a['PRATE'])
    sf=np.where((a['CPOFP']>=50)&(np.round(prec,1)>=.21)&(np.round(prec,1)-np.round(sf,1)<=.11),a['PRATE'],sf)
    return np.maximum(prec-sf-show,0)+show
def rh_hourly(T,p,spec):
    f=np.clip((T-250.16)/23,0,1)**2
    esi=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-.00728332*T)
    esw=np.exp(54.842763-6763.22/T-4.210*np.log(T)+.000367*T+np.tanh(.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+.014025*T))
    return np.clip(100*spec*p/(.622+.378*spec)/(f*esw+(1-f)*esi),0,100)

# ---------- observed (final ERA5 only, hourly, shifted UTC -> local UTC+5 day) ----------
def observed_hours(la,lo,days):
    # one call per ~90 days to stay inside API limits
    days=sorted(days);dayset=set(days);out={}
    d0=dt.date.fromisoformat(days[0])-dt.timedelta(days=1);d1=dt.date.fromisoformat(days[-1])  # pad one UTC day: local hour 0 = UTC 19:00 prev day
    cur=d0
    while cur<=d1:
        end=min(d1,cur+dt.timedelta(days=89))
        url=('https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s'
             '&hourly=precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover'
             '&timezone=GMT&models=era5')%(la,lo,cur.isoformat(),end.isoformat())
        j=json.loads(get(url))['hourly'];base=dt.datetime.fromisoformat(j['time'][0])
        for i,t in enumerate(j['time']):
            u=dt.datetime.fromisoformat(t)
            ldt=u+dt.timedelta(hours=5)          # corpus local day boundary = UTC+5
            ds=ldt.date().isoformat()
            if ds in days:
                h=ldt.hour
                out.setdefault(ds,[ [None]*5 for _ in range(24) ])
                vals=[j['precipitation'][i],j['temperature_2m'][i],j['relative_humidity_2m'][i],j['wind_speed_10m'][i],j['cloud_cover'][i]]
                out[ds][h]=[None if v is None else round(float(v),2) for v in vals]
        cur=end+dt.timedelta(days=1)
        time.sleep(0.3)
    return out

# ---------- main ----------
d0=dt.date.fromisoformat(START);d1=dt.date.fromisoformat(END)
days=[];d=d0
while d<=d1:days.append(d.isoformat());d+=dt.timedelta(1)
dayset=set(days)
pool=concurrent.futures.ThreadPoolExecutor(16)
obs_by_site={}
for name,la,lo in SITES:
    try:obs_by_site[name]=observed_hours(la,lo,dayset)
    except Exception as e:log('obs failed',name,e);obs_by_site[name]={}
    time.sleep(0.3)
log('observed fetched for',len(obs_by_site),'sites')
FCOLS=['rain24_mm','temp24_c','rh24_pct','wind24_kmh','cloud24_pct','rain48_mm','temp48_c','rain72_mm','temp72_c']
OCOLS=['rain_mm','temp_c','rh_pct','wind_kmh','cloud_pct']
new={name:{} for name,_,_ in SITES}
for date in days:
    a1=fetch_day(date,pool,1,I13_9[sidx9],I25_9[sidx9],'FULL')
    r1=rain_hourly(a1);t1=a1['TMP']-273.15
    rh1=rh_hourly(a1['TMP'],a1['PRES'],a1['SPFH'])
    w1=np.hypot(a1['UGRD'],a1['VGRD'])*3.6;c1=a1['TCDC']
    a2=fetch_day(date,pool,2,I13_w[sidxw],I25_w[sidxw],'RT');r2=rain_hourly(a2);t2=a2['TMP']-273.15
    a3=fetch_day(date,pool,3,I13_w[sidxw],I25_w[sidxw],'RT');r3=rain_hourly(a3);t3=a3['TMP']-273.15
    del a1,a2,a3
    for k,(name,la,lo) in enumerate(SITES):
        frows=[]
        for h in range(24):
            frows.append([round(float(r1[h,k]),2),round(float(t1[h,k]),1),round(float(rh1[h,k]),0),round(float(w1[h,k]),1),round(float(c1[h,k]),0),
                          round(float(r2[h,k]),2),round(float(t2[h,k]),1),round(float(r3[h,k]),2),round(float(t3[h,k]),1)])
        new[name][date]={'f':frows,'o':obs_by_site[name].get(date)}
    log('built',date,'pune h14 f:',new['pune'][date]['f'][14],'o:',(new['pune'][date]['o'] or [None])[14] if new['pune'][date]['o'] else None)
# ---------- merge (never overwrite) ----------
added=0
for name,_,_ in SITES:
    byyear={}
    for date,payload in new[name].items():byyear.setdefault(date[:4],[]).append((date,payload))
    for y,items in byyear.items():
        fp=os.path.join('app','history6h',name,y+'.json')
        if os.path.exists(fp):doc=json.load(open(fp))
        else:doc={'site':name,'year':y,'hours':{'f':FCOLS,'o':OCOLS},'days':{}}
        add=0
        for date,payload in items:
            if date in doc['days']:continue
            doc['days'][date]=payload;add+=1
        if add:
            doc['days']=dict(sorted(doc['days'].items()))
            doc['note']='Hourly backtrack: raw GFS from the NOAA grib archive at the nearest network/world grid cell (same conventions as the daily rows); observed = final ERA5 via Open-Meteo (models=era5) - hours ERA5 has not published yet stay null and fill in nightly. No trained values: the learned correction works at full-day scale only. Local day convention matches the daily rows (UTC+5).'
            os.makedirs(os.path.dirname(fp),exist_ok=True)
            json.dump(doc,open(fp,'w'),separators=(',',':'));added+=add
log('DONE history6h fill, days added:',added)
