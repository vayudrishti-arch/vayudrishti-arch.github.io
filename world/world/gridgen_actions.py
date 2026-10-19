"""Worldwide 0.125-deg grid + cells for VayuDrishti. Runs in Actions.
Downloads the original rebuild bundle (points9112/cells9112/gfs013_lats) from the pinned
user-attachments URL, generates pointsw.json.gz + cellsw.json.gz, verifies against
cells9112, uploads both to release 'world-assets'."""
import json,subprocess,urllib.request,datetime as dt,os,tarfile,gzip,shutil
import numpy as np, eccodes

BUNDLE='https://github.com/user-attachments/files/33634084/1-rebuild_bundle2.tar.gz'
subprocess.run(['curl','-sL','-o','/tmp/rb.tgz',BUNDLE],check=True)
os.makedirs('/tmp/rb',exist_ok=True)
subprocess.run(['tar','xzf','/tmp/rb.tgz','-C','/tmp/rb'],check=True)

# latest posted GFS run (f000 is enough for LAND/HGT)
now=dt.datetime.utcnow()-dt.timedelta(hours=4)
run=now.replace(hour=now.hour//6*6,minute=0,second=0,microsecond=0)
idx=None
for _ in range(4):
    base=f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f000'
    try:
        idx=urllib.request.urlopen(base+'.idx',timeout=40).read().decode().splitlines();break
    except Exception:
        run=run-dt.timedelta(hours=6)
assert idx,'no GFS run found'
def fr(name):
    for i,l in enumerate(idx):
        if f':{name}:surface:' in l:return int(l.split(':')[1]),int(idx[i+1].split(':')[1])-1
    raise KeyError(name)
def gf(name):
    s,e=fr(name)
    b=urllib.request.urlopen(urllib.request.Request(base,headers={'Range':f'bytes={s}-{e}'}),timeout=60).read()
    g=eccodes.codes_new_from_message(b);v=eccodes.codes_get_values(g).astype(np.float32).reshape(721,1440)
    eccodes.codes_release(g);return v
LAND=gf('LAND');HGT=gf('HGT')
landmask=(LAND>=0.5);landmask[600:,:]=False

pts_in=json.load(open('/tmp/rb/points9112.json'));old=json.load(open('/tmp/rb/cells9112.json'))
in_by_parent={}
for p in pts_in:
    pr=int(round((90-p['latitude'])/0.25));pc=int(round((p['longitude']%360)/0.25))%1440
    in_by_parent[(pr,pc)]=p['region']
def region_box(lat,lon):
    if 55<=lon<100 and 3<=lat<38:return 'south_asia'
    if 28<=lon<63 and 12<=lat<42:return 'middle_east'
    if (lon>=330 or lon<40) and 36<=lat<=75:return 'europe'
    if (lon>=340 or lon<55) and -38<=lat<36:return 'africa'
    if 40<=lon<95 and 38<=lat<58:return 'central_asia'
    if 95<=lon<150 and 15<=lat<55:return 'east_asia'
    if 88<=lon<165 and -12<=lat<15:return 'southeast_asia'
    if 40<=lon<190 and 55<=lat<=83:return 'north_asia'
    if 165<=lon<300 and 5<=lat<=84:return 'north_america'
    if 300<=lon<345 and 58<=lat<=84:return 'north_america'
    if 270<=lon<340 and -60<=lat<5:return 'south_america'
    if 105<=lon<180 and -50<=lat<-12:return 'oceania'
    if 150<=lon<250 and -40<=lat<30:return 'oceania'
    return 'other'
pts=[]
for pr,pc in np.argwhere(landmask):
    h=float(round(float(HGT[pr,pc]),1));key=(int(pr),int(pc));in_india=key in in_by_parent
    for di in range(2):
        for dj in range(2):
            i=2*pr+di;j=2*pc+dj
            lat=round(89.9375-0.125*i,4);lon=round(0.0625+0.125*j,4)
            if lat<=-60:continue
            reg=in_by_parent[key] if in_india else region_box(lat,lon)
            pts.append({'region':reg,'latitude':lat,'longitude':lon,'elevation':h,'fit_eligible':bool(((i*2880+j)%8)!=0),'eligible_const':True})
for p in pts_in:
    pts.append({'region':p['region'],'latitude':round(p['latitude'],4),'longitude':round(p['longitude'],4),'elevation':p['elevation'],'fit_eligible':p['fit_eligible'],'eligible_const':p.get('eligible_const',True)})
N=len(pts)
lat_a=np.array([p['latitude'] for p in pts],np.float32);lon_a=np.array([p['longitude'] for p in pts],np.float32)
g13=np.array(json.load(open('/tmp/rb/gfs013_lats.json')),np.float32)
row13=np.empty(N,np.int32)
for s in range(0,N,20000):
    e=min(s+20000,N);row13[s:e]=np.abs(g13[None,:]-lat_a[s:e,None]).argmin(1)
col13=(np.round((lon_a%360)/0.1171875).astype(np.int64)%3072).astype(np.int32)
row25=np.round((90-lat_a)/0.25).astype(np.int32);col25=(np.round((lon_a%360)/0.25).astype(np.int64)%1440).astype(np.int32)
cells={'row13':[int(x) for x in row13],'col13':[int(x) for x in col13],'row25':[int(x) for x in row25],'col25':[int(x) for x in col25],'rowE':[int(x) for x in row25],'colE':[int(x) for x in col25]}
base=N-len(pts_in)
for k in cells:cells[k][base:]=[int(x) for x in old[k]]
ok=sum(1 for i in range(len(pts_in)) if all(cells[k][base+i]==int(old[k][i]) for k in cells))
assert ok==len(pts_in),f'india cell copy mismatch {ok}'
# day list: 2021-03-25 .. today-6, minus the two corrupt days
alld=[];d=dt.date(2021,3,25);end=dt.datetime.utcnow().date()-dt.timedelta(days=6)
while d<=end:
    if str(d) not in ('2022-11-30','2022-12-01'):alld.append(str(d))
    d+=dt.timedelta(1)
json.dump(alld,open('alld_world.json','w'))
json.dump(pts,open('pointsw.json','w'));json.dump(cells,open('cellsw.json','w'))
subprocess.run(['gzip','-9','-f','pointsw.json'],check=True);subprocess.run(['gzip','-9','-f','cellsw.json'],check=True)
print('points',N,'days',len(alld))
