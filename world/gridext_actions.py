"""Build the 1M-grid Antarctic extension from the existing 980,692-point base grid.

Expects pointsw.json, cellsw.json, gfs013_lats.json in CWD (the workflow fetches them
from the world-assets release). Fetches the SAME pinned GFS analysis used for the base
grid (gfs.20261019/06 f000 LAND/HGT) so the land mask matches exactly, then appends the
4,827 northmost Antarctic land cells (rows>=601) x 4 sub-points = 19,308 new points,
region 'antarctica'. Writes pointsw_ext/cellsw_ext (extension only) and
pointsw1m/cellsw1m (base order + extension = exactly 1,000,000 points).
"""
import json,numpy as np,urllib.request,eccodes,gzip,shutil,os,hashlib
BASE_URL='https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261019/06/atmos/gfs.t06z.pgrb2.0p25.f000'
idx=urllib.request.urlopen(BASE_URL+'.idx',timeout=60).read().decode().splitlines()
def field_range(name):
    for i,l in enumerate(idx):
        if f':{name}:surface:' in l:
            return int(l.split(':')[1]),int(idx[i+1].split(':')[1])-1
    raise KeyError(name)
def get_field(name):
    s,e=field_range(name)
    b=urllib.request.urlopen(urllib.request.Request(BASE_URL,headers={'Range':f'bytes={s}-{e}'}),timeout=60).read()
    g=eccodes.codes_new_from_message(b)
    v=eccodes.codes_get_values(g).astype(np.float32).reshape(721,1440)
    eccodes.codes_release(g)
    return v
LAND=get_field('LAND');HGT=get_field('HGT')
amask=(LAND>=0.5);amask[:601,:]=False
cells_ant=np.argwhere(amask)
print('antarctic land cells available:',len(cells_ant),'northmost:',cells_ant[0].tolist())
cells_ant=cells_ant[:4827]
print('using cells:',len(cells_ant),'last used:',cells_ant[-1].tolist())
pts=[]
for pr,pc in cells_ant:
    h=float(round(float(HGT[pr,pc]),1))
    for di in range(2):
        for dj in range(2):
            i=2*pr+di;j=2*pc+dj
            lat=round(89.9375-0.125*i,4);lon=round(0.0625+0.125*j,4)
            pts.append({'region':'antarctica','latitude':lat,'longitude':lon,'elevation':h,'fit_eligible':bool(((i*2880+j)%8)!=0),'eligible_const':True})
assert len(pts)==19308,len(pts)
lat_a=np.array([p['latitude'] for p in pts],np.float32);lon_a=np.array([p['longitude'] for p in pts],np.float32)
g13=np.array(json.load(open('gfs013_lats.json')),np.float32)
row13=np.abs(g13[None,:]-lat_a[:,None]).argmin(1).astype(np.int32)
col13=(np.round((lon_a%360)/0.1171875).astype(np.int64)%3072).astype(np.int32)
row25=np.round((90-lat_a)/0.25).astype(np.int32)
col25=(np.round((lon_a%360)/0.25).astype(np.int64)%1440).astype(np.int32)
cells_ext={'row13':[int(x) for x in row13],'col13':[int(x) for x in col13],'row25':[int(x) for x in row25],'col25':[int(x) for x in col25],'rowE':[int(x) for x in row25],'colE':[int(x) for x in col25]}
base_pts=json.load(open('pointsw.json'));base_cells=json.load(open('cellsw.json'))
assert len(base_pts)==980692
allp=base_pts+pts
cells1m={k:base_cells[k]+cells_ext[k] for k in base_cells}
assert len(allp)==1000000 and all(len(cells1m[k])==1000000 for k in cells1m)
json.dump(pts,open('pointsw_ext.json','w'));json.dump(cells_ext,open('cellsw_ext.json','w'))
json.dump(allp,open('pointsw1m.json','w'));json.dump(cells1m,open('cellsw1m.json','w'))
for f in ['pointsw_ext','cellsw_ext','pointsw1m','cellsw1m']:
    with open(f'{f}.json','rb') as fi,gzip.open(f'{f}.json.gz','wb') as fo:shutil.copyfileobj(fi,fo)
    print(f,'bytes',os.path.getsize(f'{f}.json'),'sha256',hashlib.sha256(open(f'{f}.json','rb').read()).hexdigest())
elig=sum(1 for p in allp if p['fit_eligible'])
print('total',len(allp),'fit_eligible',elig,'heldout',len(allp)-elig)
