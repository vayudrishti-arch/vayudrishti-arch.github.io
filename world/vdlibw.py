"""VayuDrishti lead-1 data pipeline (GFS lead-1 forecast + ERA5 reference -> daily rows for 4,000 points).
Rebuilt Oct 5 2026. points.json holds the frozen 4,000 points; cells are derived by nearest-cell rule."""
import json,pathlib,numpy as np,datetime as dt,time,urllib.request,eccodes,threading,concurrent.futures,os,sys
P=pathlib.Path(__file__).parent
import os
pts=json.load(open(os.environ.get('VD_POINTS',str(P/'points.json'))))
lat=np.array([p['latitude'] for p in pts]);lon=np.array([p['longitude'] for p in pts])
G13=np.array(json.load(open(P/'gfs013_lats.json')))
row13=np.empty(len(lat),np.int64)
for _s in range(0,len(lat),20000):
    _e=min(_s+20000,len(lat));row13[_s:_e]=np.abs(G13[None,:].astype(np.float32)-lat[_s:_e,None].astype(np.float32)).argmin(1)
col13=np.round((lon%360)/0.1171875).astype(int)%3072
IDX13=row13*3072+col13
row25=np.round((90-lat)/0.25).astype(int);col25=np.round((lon%360)/0.25).astype(int)%1440
IDX25=row25*1440+col25

import os as _os
if _os.environ.get('VD_CELLS'):
    _c=json.load(open(_os.environ['VD_CELLS']))
    IDX13=np.array(_c['row13'])*3072+np.array(_c['col13']);IDX25=np.array(_c['row25'])*1440+np.array(_c['col25']);IYE=np.array(_c['rowE']);IXE=np.array(_c['colE'])
else:
    _c=json.load(open(P/'cells.json'))
IDX13=np.array(_c['row13'])*3072+np.array(_c['col13']);IDX25=np.array(_c['row25'])*1440+np.array(_c['col25']);IYE=np.array(_c['rowE']);IXE=np.array(_c['colE'])  # cells recovered by matching the seed data
lock=threading.Lock()
FIELDS=['temperature_k','specific_humidity','surface_pressure_pa','u10','v10','precipitation_mm','showers_mm','frozen_percent','pressure_msl_hpa','cape']
NEED=[['TMP:2 m above ground:','SPFH:2 m above ground:','PRES:surface:','UGRD:10 m above ground:','VGRD:10 m above ground:','PRATE:surface:','CPRAT:surface:','CPOFP:surface:'],['MSLET:mean sea level:','CAPE:surface:']]
def get(url,rng=None,tries=4):
    for t in range(tries):
        try:
            req=urllib.request.Request(url,headers={'Range':rng}if rng else {})
            with urllib.request.urlopen(req,timeout=40)as r:
                if rng and r.status!=206:raise ValueError('Range not honored')
                return r.read()
        except Exception:
            if t==tries-1:raise
            time.sleep(2*(t+1))
def indexed(base):return [(int(s.split(':')[1]),':'.join(s.split(':')[3:]))for s in get(base+'.idx').decode().splitlines()]
def extract(base,index,prefix,idx):
    i=next(i for i,x in enumerate(index)if x[1].startswith(prefix));b=get(base,f'bytes={index[i][0]}-{index[i+1][0]-1}')
    with lock:g=eccodes.codes_new_from_message(b);v=eccodes.codes_get_values(g)[idx].astype(np.float32);eccodes.codes_release(g)
    return v,len(b),index[i][1]
def layout(date):  # GFS bucket moved files under atmos/ on 2021-03-23
    return 'atmos/' if date>='2021-03-23' else ''
def step_job(date,hour,lead=1):
    utc=dt.datetime.fromisoformat(date)+dt.timedelta(hours=hour-5);run=utc.replace(hour=utc.hour//6*6)-dt.timedelta(days=lead);step=int((utc-run).total_seconds()/3600);arrays={};nb=0
    for kind,nn,idx in [(f'sfluxgrbf{step:03d}.grib2',NEED[0],IDX13),(f'pgrb2.0p25.f{step:03d}',NEED[1],IDX25)]:
        base=f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/{layout(f"{run:%Y-%m-%d}")}gfs.t{run:%H}z.{kind}';index=indexed(base);previndex=None
        for prefix in nn:
            v,n,desc=extract(base,index,prefix,idx);nb+=n
            if prefix.startswith(('PRATE:','CPRAT:')):
                a,b=map(int,desc.split(':')[-2].split(' ')[0].split('-'))
                if b-a>1:
                    pb=base.replace(f'sfluxgrbf{step:03d}',f'sfluxgrbf{step-1:03d}');previndex=previndex or indexed(pb);prev,n,_=extract(pb,previndex,prefix,idx);nb+=n;v=v*np.float32(b-a)-prev*np.float32(b-a-1)
                v=np.maximum(v,0)*np.float32(3600)
            if prefix.startswith('MSLET:'):v/=np.float32(100)
            arrays[FIELDS[(NEED[0]+NEED[1]).index(prefix)]]=v
    return hour,arrays,nb
def fetch_gfs_day(date,pool,lead=1):
    res=list(pool.map(lambda h:step_job(date,h,lead),range(24)))
    return {k:np.stack([r[1][k] for r in sorted(res,key=lambda r:r[0])]) for k in FIELDS},sum(r[2] for r in res)
_meta=None
def era_meta():
    global _meta
    if _meta is None:
        import numcodecs
        m=json.load(urllib.request.urlopen(ERA+'/.zmetadata',timeout=40))['metadata']
        _meta=(numcodecs.get_codec(m['total_precipitation/.zarray']['compressor']),dt.datetime.fromisoformat(m['time/.zattrs']['units'].split('since ')[1].replace(' ','T')))
    return _meta
ERA='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
def fetch_era_day(date):
    codec,origin=era_meta();d=dt.date.fromisoformat(date);utc=dt.datetime.combine(d,dt.time())-dt.timedelta(hours=5);n=len(pts);tot=np.empty((24,n),np.float32);snow=np.empty((24,n),np.float32)
    for h in range(24):
        idx=int((utc+dt.timedelta(hours=h)-origin).total_seconds()/3600)
        for var,out in [('total_precipitation',tot),('snowfall',snow)]:
            arr=np.frombuffer(codec.decode(get(f'{ERA}/{var}/{idx}.0.0')),dtype='<f4').reshape(721,1440);out[h]=np.maximum(arr[IYE,IXE],0)*np.float32(1000)
    q=lambda v:np.floor(v*10+.5)/10
    rain=np.maximum(q(tot)-q(snow),0)
    return rain.sum(axis=0)
q=lambda a,scale:np.copysign(np.floor(abs(a)*scale+.5),a)/scale
def assemble(date,a):
    T=a['temperature_k'];p=a['surface_pressure_pa'];spec=a['specific_humidity'];f=np.clip((T-250.16)/23,0,1)**2
    esi=np.exp(9.550426-5723.265/T+3.53068*np.log(T)-.00728332*T);esw=np.exp(54.842763-6763.22/T-4.210*np.log(T)+.000367*T+np.tanh(.0415*(T-218.8))*(53.878-1331.22/T-9.44523*np.log(T)+.014025*T))
    rh=q(np.clip(100*spec*p/(.622+.378*spec)/(f*esw+(1-f)*esi),0,100),1)
    pr=q(a['pressure_msl_hpa'],10);cape=q(a['cape'],.1);u=q(a['u10'],10);v=q(a['v10'],10);wind=q(np.hypot(u,v)*3.6,10);direction=q((np.degrees(np.arctan2(u,v))+180)%360,1)
    prec=q(a['precipitation_mm'],10);show=q(a['showers_mm'],10);sf=np.clip(a['frozen_percent']/100*a['precipitation_mm'],0,a['precipitation_mm'])
    sf=np.where((a['frozen_percent']>=50)&(prec>=.21)&(prec-q(sf,10)<=.11),a['precipitation_mm'],sf);snow=q(sf,10);rain=np.maximum(prec-snow-show,0);raw=(rain+show).sum(axis=0);mo=int(date[5:7])
    ones=np.ones(len(pts))
    feats=np.stack([prec.sum(0),prec.max(0),(prec>=.1).sum(0),cape.mean(0),cape.max(0),rh.mean(0),rh.min(0),pr.mean(0),pr.min(0),np.ptp(pr,axis=0),pr[23]-pr[0],wind.mean(0),wind.max(0),(wind*np.cos(direction*np.pi/180)).mean(0),(wind*np.sin(direction*np.pi/180)).mean(0),lat,lon,np.array([p['elevation'] for p in pts]),ones*np.sin(2*np.pi*mo/12),ones*np.cos(2*np.pi*mo/12)],axis=-1).astype(np.float32)
    if not np.isfinite(feats).all()or (raw<0).any():raise ValueError('invalid derived data '+date)
    return feats,raw.astype(np.float32)
