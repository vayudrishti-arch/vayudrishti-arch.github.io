"""VayuDrishti nightly archiver: records the ORIGINAL as-served forecasts.
Replays the live app's exact serve path (same ENSEMBLE/MODEL consts parsed from the
live page, same Open-Meteo fetch, same f32 xgb math) so tonight's capture equals what
the site serves tonight for every one of the 24 network sites and leads 24/48/72h.
Also refreshes observed rain/temp (ERA5 via Open-Meteo archive) for the trailing days
in the per-year history files written by history/backfill.py.
Outputs app/history/originals/<site>/<year>.json + updates app/history/index.json.
Runs in Actions on a nightly cron. Free APIs only."""
import os,json,math,time,datetime as dt,urllib.request
import numpy as np
from zoneinfo import ZoneInfo

UA={'User-Agent':'vayudrishti-archiver'}
def get(url,tries=4):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=60).read()
        except Exception:
            if t==tries-1:raise
            time.sleep(2*(t+1))
def log(*a):print(*a,flush=True)
f32=lambda v:float(np.float32(v))

# --- parse the live page: ENSEMBLE + MODEL (exact same source the site serves) ---
def extract_const(html,name):
    i=html.find('const '+name+'=')+len('const '+name+'=')
    d=0
    for p in range(i,len(html)):
        if html[p]=='{':d+=1
        elif html[p]=='}':
            d-=1
            if d==0:return json.loads(html[i:p+1])
    raise RuntimeError(name+' not found')
html=get('https://vayudrishti-arch.github.io/app/').decode()
ENSEMBLE=extract_const(html,'ENSEMBLE')
MODEL=extract_const(html,'MODEL')
log('live page models:',ENSEMBLE['version'],'/',MODEL['version'])
SITES=MODEL['locations']  # [name, lat, lon, region]

# --- exact ports of the app's JS serve math ---
def interpolate(x,a,b):
    if x<a[0]:return b[0]
    if x>=a[-1]:return b[-1]
    i=0
    while i<len(a)-1 and a[i+1]<=x:i+=1
    return b[i]+(b[i+1]-b[i])*(x-a[i])/(a[i+1]-a[i])
def xgb_predict(x,m):
    x=[f32(v) for v in x];y=f32(m['base'])
    for t in m['trees']:
        lc=t['left_children'];rc=t['right_children'];si=t['split_indices'];sc=t['split_conditions'];dl=t['default_left']
        n=0
        for _ in range(20):
            l=lc[n]
            if l==-1:break
            v=x[si[n]]
            left=dl[n] if v!=v else v<f32(sc[n])
            n=l if left else rc[n]
        y=f32(y+f32(sc[n]))
    return y
def weights(x,e):
    s=[1+3*(x<1),1+3*(e['pressure']<1005 and e['wind']>15),1+3*(e['elevation']>700 and e['rh']>75),1+3*(e['rh']>75 and x>=1)]
    sm=sum(s);return [v/sm for v in s]
def incumbent(date,lead,raw,env,lat,lon):
    m=next((m for m in MODEL['models'] if m['lead']==lead),None)
    if not m or not all(map(math.isfinite,[raw,env['rh'],env['pressure'],env['wind'],env['elevation']])):return None
    w=weights(raw,env);z=math.log1p(raw);mo=int(date[5:7])
    f=[1,z,z*w[0],z*w[1],z*w[2],(env['rh']-70)/30,(env['pressure']-1010)/20,env['wind']/30,(lat-22)/12,(lon-82)/15,math.sin(2*math.pi*mo/12),math.cos(2*math.pi*mo/12)]
    return max(0.0,math.expm1(min(8.0,sum(v*c for v,c in zip(f,m['coefficients']))))),w
def ensemble_amount(raw,rich,inc,region,lead):
    m=next((m for m in ENSEMBLE['models'] if m['lead']==lead),None)
    if m is None:return None
    if raw==0:return {'amount':0.0,'zero_gate':1,'heavy_flag':0}
    mp=m['qm'].get(region) or m['qm']['global']
    q=interpolate(raw,mp['source'],mp['target'])
    x=list(rich)+[raw,q,math.log1p(raw),math.log1p(q)]
    a=max(0.0,q+xgb_predict(x,m));flag=0
    if m.get('clf'):
        pr=1/(1+math.exp(-xgb_predict(x,m['clf'])))
        if pr>=m['clf']['threshold']:flag=1;a=max(a,64.5)
    return {'amount':m['weight']*a+(1-m['weight'])*inc,'zero_gate':0,'heavy_flag':flag}

def local_noon_ms(date,tzname):
    y,mn,d=map(int,date.split('-'))
    target=dt.datetime(y,mn,d,12,tzinfo=dt.timezone.utc).timestamp()*1000
    t=target
    for _ in range(3):
        loc=dt.datetime.fromtimestamp(t/1000,tz=dt.timezone.utc).astimezone(ZoneInfo(tzname))
        shown=dt.datetime(loc.year,loc.month,loc.day,loc.hour,loc.minute,loc.second,tzinfo=dt.timezone.utc).timestamp()*1000
        t+=target-shown
    return t

def mean(a):return sum(a)/len(a)

# --- capture tonight's as-served forecasts for every site ---
issued_ist=dt.datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
loaded=time.time()*1000
captured=0
for name,lat,lon,region in SITES:
    url=('https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&models=gfs_seamless'
         '&hourly=rain,showers,precipitation,cape,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m'
         '&timezone=auto&forecast_days=5&precipitation_unit=mm')%(lat,lon)
    a=json.loads(get(url));h=a['hourly'];tz=a['timezone']
    now=dt.datetime.now(ZoneInfo(tz));today=now.date().isoformat()
    dates=sorted(set(t[:10] for t in h['time'] if t[:10]>today))
    rows=[]
    for date in dates:
        ids=[i for i,t in enumerate(h['time']) if t.startswith(date)]
        arr=lambda k:[h[k][i] for i in ids]
        allf=lambda k:len(ids)==24 and all(v is not None and math.isfinite(v) for v in arr(k))
        if not(allf('rain') and allf('showers')):continue
        raw=sum(arr('rain'))+sum(arr('showers'))
        hours=(local_noon_ms(date,tz)-loaded)/3600000
        lead=round(hours/24)
        if lead<1 or lead>3:continue
        keys=['precipitation','cape','relative_humidity_2m','pressure_msl','wind_speed_10m','wind_direction_10m']
        if not all(allf(k) for k in keys):continue
        pr,ca,hu,ps,ws,wd=(arr(k) for k in keys)
        mo=int(date[5:7])
        rich=[sum(pr),max(pr),len([v for v in pr if v>=.1]),mean(ca),max(ca),mean(hu),min(hu),mean(ps),min(ps),
              max(ps)-min(ps),ps[23]-ps[0],mean(ws),max(ws),
              mean([v*math.cos(w*math.pi/180) for v,w in zip(ws,wd)]),
              mean([v*math.sin(w*math.pi/180) for v,w in zip(ws,wd)]),
              lat,lon,a['elevation'],math.sin(2*math.pi*mo/12),math.cos(2*math.pi*mo/12)]
        env={'rh':mean(hu),'pressure':mean(ps),'wind':mean(ws),'elevation':a['elevation']}
        ib=incumbent(date,lead,raw,env,lat,lon)
        if ib is None:continue
        en=ensemble_amount(raw,rich,ib[0],region,lead)
        if en is None:continue
        rows.append([date,lead*24,round(raw,2),round(en['amount'],2),en['zero_gate'],en['heavy_flag']])
    # merge into app/history/originals/<site>/<year>.json (first write wins per date+lead)
    outdir=os.path.join('app','history','originals',name);os.makedirs(outdir,exist_ok=True)
    byyear={}
    for r in rows:byyear.setdefault(r[0][:4],[]).append(r)
    for y,new in byyear.items():
        fp=os.path.join(outdir,y+'.json')
        if os.path.exists(fp):
            doc=json.load(open(fp))
        else:
            doc={'site':name,'year':y,'kind':'archived-original',
                 'columns':['date','lead_h','raw_gfs_mm','corrected_mm','zero_gate','heavy_flag'],
                 'note':'Original forecast as served by the live app, captured nightly at ~02:00 IST. Not a reconstruction.',
                 'model_version':ENSEMBLE['version'],'rows':[]}
        have={(r[0],r[1]) for r in doc['rows']}
        add=[r for r in new if (r[0],r[1]) not in have]
        if add:
            doc['rows']=sorted(doc['rows']+add)
            doc['issued_ist_last']=issued_ist
            json.dump(doc,open(fp,'w'),separators=(',',':'))
            captured+=len(add)
    time.sleep(0.4)
log('captured new original rows:',captured)

# --- refresh observed rain/temp for trailing days in the backfill year files ---
today_i=dt.datetime.now(ZoneInfo('Asia/Kolkata')).date()
start=(today_i-dt.timedelta(days=14)).isoformat();end=(today_i-dt.timedelta(days=5)).isoformat()
updated=0
for name,lat,lon,region in SITES:
    try:
        url=('https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s'
             '&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Asia%%2FKolkata')%(lat,lon,start,end)
        j=json.loads(get(url))['daily']
    except Exception as e:
        log('observed refresh failed',name,e);continue
    obs={t:(ps,tmx,tmn) for t,tmx,tmn,ps in zip(j['time'],j['temperature_2m_max'],j['temperature_2m_min'],j['precipitation_sum'])}
    for y in {t[:4] for t in obs}:
        fp=os.path.join('app','history',name,y+'.json')
        if not os.path.exists(fp):continue
        doc=json.load(open(fp));changed=False
        for row in doc['rows']:
            if row[0] in obs:
                ps,tmx,tmn=obs[row[0]]
                if ps is not None and row[3] is None:row[3]=round(ps,2);changed=True
                if tmx is not None and row[4] is None:row[4]=round(tmx,1);row[5]=round(tmn,1) if tmn is not None else row[5];changed=True
        if changed:
            doc['observed_recent_note']='Recent observed values: ERA5 via Open-Meteo archive API grid cell (same family as corpus ERA5), refreshed nightly once the lag clears.'
            json.dump(doc,open(fp,'w'),separators=(',',':'));updated+=1
    time.sleep(0.4)
log('observed refresh files updated:',updated)

# --- index update ---
idxp=os.path.join('app','history','index.json')
if os.path.exists(idxp):
    idx=json.load(open(idxp))
    idx['originals']={'kind':'archived-original','path':'originals/<site>/<year>.json',
        'note':'Original live forecasts captured nightly since 2026-10-22 ~02:00 IST. Earlier dates have no original - only the labeled reconstruction.',
        'sites':{s[0]:sorted({f[:-5] for f in os.listdir(os.path.join('app','history','originals',s[0]))} if os.path.isdir(os.path.join('app','history','originals',s[0])) else []) for s in SITES}}
    idx['generated_utc']=dt.datetime.utcnow().isoformat(timespec='seconds')+'Z'
    json.dump(idx,open(idxp,'w'),indent=1)
log('DONE archive_day')
