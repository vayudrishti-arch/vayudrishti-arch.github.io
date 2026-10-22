"""VayuDrishti backtrack backfill: per-site daily history from the verified corpus.
Real data only: raw GFS (RR) and ERA5 rain truth (T) come straight from the corpus day files.
The 24h 'corrected' column is a REPLAY of the current live model on archived features -
always labeled reconstruction, never the original live forecast.
Observed temperature comes from Open-Meteo's ERA5 archive (labeled, same ~5-6 day lag).
Outputs app/history/<site>/<year>.json + app/history/index.json. Runs in Actions."""
import os,json,io,math,time,datetime as dt,urllib.request,subprocess,gzip
import numpy as np
import concurrent.futures

REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W=os.environ.get('VD_WORK','/tmp/vdh');os.makedirs(W,exist_ok=True)
def log(*a):print(*a,flush=True)

SITES=[["pune",18.51957,73.85535],["mumbai",19.076,72.8777],["delhi",28.6139,77.209],["chennai",13.0827,80.2707],
["srinagar",34.0837,74.7973],["shimla",31.1048,77.1734],["chandigarh",30.7333,76.7794],["jaipur",26.9124,75.7873],
["jodhpur",26.2389,73.0243],["ahmedabad",23.0225,72.5714],["bhopal",23.2599,77.4126],["nagpur",21.1458,79.0882],
["raipur",21.2514,81.6296],["lucknow",26.8467,80.9462],["patna",25.5941,85.1376],["kolkata",22.5726,88.3639],
["bhubaneswar",20.2961,85.8245],["guwahati",26.1445,91.7362],["shillong",25.5788,91.8933],["imphal",24.817,93.9368],
["hyderabad",17.385,78.4867],["bengaluru",12.9716,77.5946],["thiruvananthapuram",8.5241,76.9366],["port-blair",11.6234,92.7265]]
REGS=["central","east","islands","north","northeast","south","west"]
BUNDLE='https://github.com/user-attachments/files/33634084/1-rebuild_bundle2.tar.gz'

def get(url,tries=6):
    for t in range(tries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'vayudrishti-backfill'})
            return urllib.request.urlopen(req,timeout=90).read()
        except Exception:
            if t==tries-1:raise
            time.sleep(3*(t+1))

# grid + site mapping
subprocess.run(['curl','-sL','-o',f'{W}/rb.tgz',BUNDLE],check=True)
import tarfile
tarfile.open(f'{W}/rb.tgz').extractall(f'{W}/rb')
pts=json.load(open(f'{W}/rb/points9112.json'))
plat=np.array([p['latitude'] for p in pts]);plon=np.array([p['longitude'] for p in pts])
preg=[p['region'] for p in pts]
coord2reg={(round(p['latitude'],4),round(p['longitude'],4)):p['region'] for p in pts}
sidx=[];sreg=[];sdist=[]
for name,la,lo in SITES:
    d=(plat-la)**2+(plon-lo)**2;i=int(d.argmin())
    sidx.append(i);sreg.append(preg[i]);sdist.append(round(float(math.hypot(plat[i]-la,plon[i]-lo)),3))
log('sites mapped:',list(zip([s[0] for s in SITES],sidx,sreg,sdist))[:3],'...')

# current live 24h model: parse ENSEMBLE from the live app page (same source the site serves)
html=get('https://vayudrishti-arch.github.io/app/').decode()
i=html.find('const ENSEMBLE=')+len('const ENSEMBLE=')
d=0
for p in range(i,len(html)):
    if html[p]=='{':d+=1
    elif html[p]=='}':
        d-=1
        if d==0:break
E=json.loads(html[i:p+1])
M1=[m for m in E['models'] if m['lead']==1][0]
log('24h model from live page:',E['version'])

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

def replay24(F,RR,gregion):
    """F: (24,20) site rows; RR: (24,) raw; returns corrected (24,) replaying the live 24h model."""
    out=np.zeros(len(RR),np.float32)
    for k in range(len(RR)):
        raw=float(RR[k])
        if raw==0:out[k]=0;continue
        mp=M1['qm'].get(gregion[k]) or M1['qm']['global']
        q=float(np.interp(raw,mp['source'],mp['target']))
        x=list(F[k])+[raw,q,math.log1p(raw),math.log1p(q)]
        a=max(0.0,q+tree_predict_row(M1['trees'],x,M1['base']))
        if M1.get('clf'):
            pr=1/(1+math.exp(-tree_predict_row(M1['clf']['trees'],x,M1['clf']['base'])))
            if pr>=M1['clf']['threshold']:a=max(a,64.5)
        out[k]=a
    return out

# asset list per release via gh api
def release_assets(tag):
    out=[]
    for page in (1,2,3,4):
        r=subprocess.run(['gh','api',f'repos/{REPO}/releases/tags/{tag}','--paginate','-q','.assets[]|.name+" "+.browser_download_url'],capture_output=True,text=True)
        for line in r.stdout.splitlines():
            n,u=line.split(' ',1);out.append((n,u))
        break
    return out
assets={}
for tag in ['data-days-9112','data-days-9112-r2','data-nightly']:
    for n,u in release_assets(tag):
        if n.endswith('.npz'):assets[n]=u
days=sorted(n[:-4] for n in assets)
log('corpus day files:',len(days),days[0],'..',days[-1])

# per-day extraction (threaded downloads, sequential npz parse)
series={name:{} for name,_,_ in SITES}
def one(date):
    b=get(assets[date+'.npz'])
    z=np.load(io.BytesIO(b))
    Fa=z['F']
    lat=Fa[:,15];lon=Fa[:,16]
    idx=[];regs=[]
    for name,la,lo in SITES:
        d=(lat-la)**2+(lon-lo)**2;i=int(d.argmin());idx.append(i)
        key=(round(float(lat[i]),4),round(float(lon[i]),4))
        rg=coord2reg.get(key)
        if rg is None:
            j=int(((plat-lat[i])**2+(plon-lon[i])**2).argmin());rg=preg[j]
        regs.append(rg)
    return date,Fa[idx],z['RR'][idx],z['T'][idx],regs
t0=time.time();done=0
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    for date,F,RR,T,regs in ex.map(one,days):
        corr=replay24(F,RR,regs)
        for k,(name,_,_) in enumerate(SITES):
            series[name][date]=[round(float(RR[k]),2),round(float(corr[k]),2),round(float(T[k]),2)]
        done+=1
        if done%200==0:log('days done',done,'elapsed',int(time.time()-t0),'s')

# observed temperature from Open-Meteo ERA5 archive (labeled source; same lag family)
temp={}
for name,la,lo in SITES:
    temp[name]={}
    try:
        for y0,y1 in ((2021,2022),(2023,2024),(2025,2026)):
            url=(f'https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}'
                 f'&start_date={y0}-01-01&end_date={y1}-12-31&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FKolkata')
            j=json.loads(get(url))
            for t,mx,mn in zip(j['daily']['time'],j['daily']['temperature_2m_max'],j['daily']['temperature_2m_min']):
                if mx is not None:temp[name][t]=[round(mx,1),round(mn,1)]
            time.sleep(1)
    except Exception as e:
        log('temp archive FAILED for',name,'- continuing without it:',repr(e)[:120])
    log('temp archive',name,len(temp[name]))

# write chunked history files
OUT=os.path.join(os.getcwd(),'app','history')
os.makedirs(OUT,exist_ok=True)
index={'corpus_days':len(days),'corpus_note':'The archived corpus does not contain every calendar day; only days listed in the files are real archived data.','generated_utc':dt.datetime.utcnow().isoformat(timespec='seconds')+'Z','sites':{},'years':{},
 'labels':{'raw':'Raw GFS 24h forecast for that day (as archived in the verified corpus)',
  'corrected_24h':'RECONSTRUCTION: current live 24h model replayed on archived data. Not the original live forecast (predictions were not stored before 2026-10-22).',
  'observed_rain':'Observed daily rain, ERA5 reanalysis (our verified corpus). ERA5 publishes with a lag of several days; recent days may be missing.',
  'observed_temp':'Observed daily max/min temperature, ERA5 via Open-Meteo archive. Same lag family.',
  'point':'Values at the nearest VayuDrishti network point (within about 15 km of the named place).'}}
for k,(name,la,lo) in enumerate(SITES):
    d0=os.path.join(OUT,name);os.makedirs(d0,exist_ok=True)
    byyear={}
    for date,row in sorted(series[name].items()):
        byyear.setdefault(date[:4],[]).append([date]+row+temp[name].get(date,[None,None]))
    for y,rows in byyear.items():
        json.dump({'site':name,'year':y,'columns':['date','raw_gfs_24h_mm','corrected_24h_mm_replay','observed_rain_mm_era5','observed_tmax_c_era5','observed_tmin_c_era5'],'rows':rows},open(os.path.join(d0,y+'.json'),'w'),separators=(',',':'))
    index['sites'][name]={'lat':la,'lon':lo,'grid_index':sidx[k],'grid_region':sreg[k],'grid_distance_deg':sdist[k],'years':sorted(byyear)}
    index['years'][name]=sorted(byyear)
json.dump(index,open(os.path.join(OUT,'index.json'),'w'),indent=1)
log('DONE wrote history for',len(SITES),'sites')
