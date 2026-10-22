"""One-off: fill missing observed temps in app/history/<site>/<year>.json.
The backfill's 2025-2026 temp chunk used end_date=2026-12-31, which the ERA5 archive
API rejects (max is today minus the ~5-day ERA5 lag). Fetches the real allowed range
per site, gentle pacing. Safe to re-run: only fills rows whose temp is still None."""
import os,json,time,datetime as dt,urllib.request
SITES=[["pune",18.51957,73.85535],["mumbai",19.076,72.8777],["delhi",28.6139,77.209],["chennai",13.0827,80.2707],
["srinagar",34.0837,74.7973],["shimla",31.1048,77.1734],["chandigarh",30.7333,76.7794],["jaipur",26.9124,75.7873],
["jodhpur",26.2389,73.0243],["ahmedabad",23.0225,72.5714],["bhopal",23.2599,77.4126],["nagpur",21.1458,79.0882],
["raipur",21.2514,81.6296],["lucknow",26.8467,80.9462],["patna",25.5941,85.1376],["kolkata",22.5726,88.3639],
["bhubaneswar",20.2961,85.8245],["guwahati",26.1445,91.7362],["shillong",25.5788,91.8933],["imphal",24.817,93.9368],
["hyderabad",17.385,78.4867],["bengaluru",12.9716,77.5946],["thiruvananthapuram",8.5241,76.9366],["port-blair",11.6234,92.7265]]
def get(url,tries=6):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'vayudrishti-tempfill'}),timeout=90).read()
        except Exception:
            if t==tries-1:raise
            time.sleep(8*(t+1))
def log(*a):print(*a,flush=True)
END=(dt.date.today()-dt.timedelta(days=5)).isoformat()
filled_total=0
for name,la,lo in SITES:
    docs={}
    need=[]
    for y in ('2021','2022','2023','2024','2025','2026'):
        fp=f'app/history/{name}/{y}.json'
        if not os.path.exists(fp):continue
        doc=json.load(open(fp));docs[fp]=doc
        need+=[r[0] for r in doc['rows'] if len(r)>4 and r[4] is None]
    if not need:
        log(name,'nothing to fill');continue
    start=min(need);end=min(max(need),END)
    url=(f'https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}'
         f'&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FKolkata')
    try:
        j=json.loads(get(url))
    except Exception as e:
        log('FAILED',name,repr(e)[:100]);continue
    obs={t:(mx,mn) for t,mx,mn in zip(j['daily']['time'],j['daily']['temperature_2m_max'],j['daily']['temperature_2m_min'])}
    filled=0
    for fp,doc in docs.items():
        changed=False
        for r in doc['rows']:
            if len(r)>4 and r[4] is None and r[0] in obs and obs[r[0]][0] is not None:
                r[4]=round(obs[r[0]][0],1);r[5]=round(obs[r[0]][1],1) if obs[r[0]][1] is not None else None
                filled+=1;changed=True
        if changed:json.dump(doc,open(fp,'w'),separators=(',',':'))
    filled_total+=filled
    log(name,'filled',filled)
    time.sleep(4)
log('DONE tempfill total',filled_total)
