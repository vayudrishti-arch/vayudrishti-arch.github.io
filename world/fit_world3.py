"""VayuDrishti worldwide RICHER-FEATURES fit (world-v6) + gates (1M grid).
Single-stage v4 architecture (the v5 stacking cascade was abandoned after honest
evaluation) plus two spatiotemporal feature groups, all GFS-forecast-derived (never
ERA5 truth, so the cell-level holdout stays leak-free and every feature is available
at live inference time):
  A. spatial context, same run: 3x3 0.125-deg neighborhood of each point over the raw
     forecast RR: rr_nbr_mean, rr_nbr_max, rr_nbr_wetfrac (valid neighbors only;
     longitude wraps, missing neighbors at grid/pole edges are skipped; a point with no
     valid neighbors falls back to its own RR). Computed from EXISTING lead-1 day files
     via a precomputed neighbor index - no corpus rebuild.
  B. prior-run context, same target day: rr_l2 / rr_l3 = the point's raw forecast from
     the 48h and 72h GFS runs (releases data-days-world-l2/-l3) - forecast evolution as
     the model converged on the day. Fit days missing either lead are dropped (only the
     first 1-2 days of 2021-03 and ERA5-lagged days; disclosed).
Live path (post-approval deploy work, not this script): spatial = one multi-point
gfs_seamless Open-Meteo request (9 coords); prior-run = Open-Meteo Previous Runs API.
Same honesty framework as v4: pinned eval window 2026-08-12..2026-10-12, cell-level
holdout crc32%8==7, 5 interleaved day-fold OOF calibration, thresholds on pooled OOF
over WET ROWS ONLY (matching live/eval inference; the v4 all-rows counting is fixed and
disclosed), cap FA<=102% of raw GFS with max hits, same 28 gates. The eval window is
never touched for any choice. Uploads model + metrics to release world-fit-v6.
Nothing public. Usage: python3 fit_world3.py"""
import os,json,math,time,gc,io,tarfile,zipfile,datetime as dt,urllib.request,subprocess
import numpy as np
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W='/tmp/fitw3';os.makedirs(W,exist_ok=True)
EVAL_DAYS=60;MIN_HEAVY=300;HEAVY=64.5
FIT_DAY_STRIDE=int(os.environ.get('VD_DAY_STRIDE','16'));FIT_POINT_STRIDE=int(os.environ.get('VD_POINT_STRIDE','4'))
WIN_START='2026-08-12';WIN_END='2026-10-12'   # pinned to world-v4's window
def log(*a):print(*a,flush=True)
def dl(url,path):
    if os.path.exists(path):return
    import time
    for att in range(8):
        try:
            with urllib.request.urlopen(url,timeout=600) as r,open(path+'.part','wb') as f:
                while True:
                    b=r.read(1<<24)
                    if not b:break
                    f.write(b)
            os.replace(path+'.part',path)
            return
        except Exception as e:
            log('dl retry',att,path.split('/')[-1],repr(e)[:120])
            try:os.remove(path+'.part')
            except OSError:pass
            time.sleep(10*(att+1))
    raise RuntimeError('dl failed after retries: '+path)
log('loading release indexes')
def ghapi(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'vd-fit'}),timeout=120))
urls={}
for tag in ['data-days-world','data-days-world-l2','data-days-world-l3']:
    rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}')
    urls[tag]={a['name']:a['browser_download_url'] for a in rel['assets']}
    log(tag,'assets',len(urls[tag]))
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
dl(wurls['pointsw1m.json.gz'],f'{W}/pointsw1m.json.gz')
import gzip as gz
pts=json.load(gz.open(f'{W}/pointsw1m.json.gz'))
REGIONS=sorted(set(p['region'] for p in pts))
log('regions:',REGIONS)
ridx={r:i for i,r in enumerate(REGIONS)}
reg_of=np.array([ridx[p['region']] for p in pts],np.int8)
# --- neighbor index on the 1M grid: 8 neighbors at +-0.125 deg, longitude wraps ---
log('building neighbor index')
t0=time.time()
key2i={}
for i,p in enumerate(pts):
    key2i[(int(round(p['latitude']*8)),int(round(p['longitude']*8))%2880)]=i
NBR=np.full((len(pts),8),-1,np.int32)
off=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
for i,p in enumerate(pts):
    la=int(round(p['latitude']*8));lo=int(round(p['longitude']*8))%2880
    for j,(da,db) in enumerate(off):
        k=key2i.get((la+da,(lo+db)%2880))
        if k is not None:NBR[i,j]=k
del key2i,pts;gc.collect()
log('neighbor index built','elapsed',int(time.time()-t0),'s')
def nbr_stats(RR):
    # (n,3): neighbor mean, max, wet fraction over VALID neighbors (RR>0.1 counts wet)
    g=np.where(NBR>=0,RR[np.maximum(NBR,0)],np.nan).astype(np.float32)
    cnt=(~np.isnan(g)).sum(axis=1)
    mn=np.where(cnt>0,np.nanmean(g,axis=1),RR)
    mx=np.where(cnt>0,np.nanmax(g,axis=1),RR)
    wf=np.where(cnt>0,np.nansum(g>0.1,axis=1)/np.maximum(cnt,1),(RR>0.1).astype(np.float32))
    return mn,mx,wf.astype(np.float32)
# holdout must be cell-level: sibling points share one 0.25-deg ERA5 truth cell,
# so a point-level split leaks truth cells across fit/holdout. Whole-cell split.
# (Neighbor features carry forecast RR only - never truth - so the holdout stays clean.)
dl(wurls['cellsw1m.json.gz'],f'{W}/cellsw1m.json.gz')
cellsj=json.load(gz.open(f'{W}/cellsw1m.json.gz'))
import zlib
cid=np.array([r*1440+c for r,c in zip(cellsj['rowE'],cellsj['colE'])],np.int64)
del cellsj;gc.collect()
uc=np.unique(cid)
hold_cells=set(int(c) for c in uc if zlib.crc32(int(c).to_bytes(8,'little'))%8==7)
elig=np.array([int(c) not in hold_cells for c in cid],bool)
log('cell-level holdout: cells',len(uc),'held',len(hold_cells),'pts held',int((~elig).sum()))
del cid,uc;gc.collect()
alld=json.load(urllib.request.urlopen(wurls['alld_world.json'],timeout=60))
LAGGED={'2026-10-10','2026-10-11'}   # still ERA5-lagged (404); 2026-10-13 is present but pinned out of fit+eval
full_days=[d for d in alld if d not in LAGGED]
built_days=sorted(full_days+['2021-03-24'])
win_days=[d for d in built_days if WIN_START<=d<=WIN_END]
assert len(win_days)==EVAL_DAYS,f'pinned window has {len(win_days)} days, expected {EVAL_DAYS}'
win_start=WIN_START
log('eval window (pinned to v4):',win_days[0],'to',win_days[-1],'days',len(win_days),'corpus days total',len(built_days))
fit_days=set(d for i,d in enumerate(built_days) if d<win_start and i%FIT_DAY_STRIDE==0)
log('fit days:',len(fit_days))
def main_chunk(d): return None if d=='2021-03-24' else full_days.index(d)//32
# pass 1: accumulate fit subsample
sub=np.zeros(1000000,bool);sub[::FIT_POINT_STRIDE]=True
nsub=int(sub.sum())
FL=[];RL=[];TL=[];DL=[];N1=[];N2=[];N3=[];L2=[];L3=[]
t0=time.time()
def eat_npz(b,date,dstore,rr_only=False):
    z=zipfile.ZipFile(io.BytesIO(b))
    if rr_only:
        RR=np.load(io.BytesIO(z.read('RR.npy')));dstore[date]=RR;return
    F=np.load(io.BytesIO(z.read('F.npy')));RR=np.load(io.BytesIO(z.read('RR.npy')));T=np.load(io.BytesIO(z.read('T.npy')))
    dstore[date]=(F,RR,T)
def process_chunk(tag,cname,dayset,store,rr_only=False):
    dl(urls[tag][cname],f'{W}/{tag}-{cname}')
    tf=tarfile.open(f'{W}/{tag}-{cname}',mode='r:gz')
    for m in tf.getmembers():
        d=m.name[:10]
        if d in dayset and d not in store:
            eat_npz(tf.extractfile(m).read(),d,store,rr_only)
    tf.close();os.remove(f'{W}/{tag}-{cname}')
m1={};e1={};m2={};e2={};m3={};e3={}
stores={('data-days-world',0):m1,('data-days-world',1):e1,('data-days-world-l2',0):m2,('data-days-world-l2',1):e2,('data-days-world-l3',0):m3,('data-days-world-l3',1):e3}
n_main=980692
skipped=[]
# chunk-paired streaming (cancel fix, 2026-10-22): tag-major pass 1 stored every fit
# day's full grids at once (~110MB x ~490 days) and got the job killed around the
# 26th streamed chunk. Same days, same rows, same append order - but each main chunk
# id is now streamed with its ext/l2/l3 partners and reduced to the fit subsample
# immediately, so peak memory stays near one chunk instead of the whole corpus.
days_by_mc={}
for d in fit_days:
    mc=main_chunk(d)
    if mc is None:continue
    days_by_mc.setdefault(mc,set()).add(d)
def flush_fit_days(dlist):
    for d in sorted(dlist):
        ok=(d in m1 and d in e1 and d in m2 and d in e2 and d in m3 and d in e3)
        if not ok:skipped.append((d,'missing lead coverage'));continue
        F=np.concatenate([m1[d][0],e1[d][0]]);RR=np.concatenate([m1[d][1],e1[d][1]]);T=np.concatenate([m1[d][2],e1[d][2]])
        r2=np.concatenate([m2[d],e2[d]]);r3=np.concatenate([m3[d],e3[d]])
        if not(len(RR)==1000000 and len(r2)==1000000 and len(r3)==1000000):
            skipped.append((d,'short rows'));continue
        mn,mx,wf=nbr_stats(RR)
        FL.append(F[sub]);RL.append(RR[sub]);TL.append(T[sub]);DL.append(np.full(nsub,int(d.replace('-','')),np.int32))
        N1.append(mn[sub]);N2.append(mx[sub]);N3.append(wf[sub]);L2.append(r2[sub]);L3.append(r3[sub])
        del F,RR,T,r2,r3,mn,mx,wf
        for st in (m1,e1,m2,e2,m3,e3):st.pop(d,None)
        if len(FL)%20==0:gc.collect();log('fit buf days',len(FL),'elapsed',int(time.time()-t0),'s')
for mc in sorted(days_by_mc):
    dset=days_by_mc[mc]
    process_chunk('data-days-world',f'world-chunk-{mc}.tar.gz',dset,m1)
    log('streamed data-days-world',f'world-chunk-{mc}.tar.gz','elapsed',int(time.time()-t0),'s')
    if mc<63:
        process_chunk('data-days-world',f'world-chunk-{64+mc}.tar.gz',dset,e1)
        log('streamed data-days-world',f'world-chunk-{64+mc}.tar.gz','elapsed',int(time.time()-t0),'s')
    for ptag,pstores in (('data-days-world-l2',(m2,e2)),('data-days-world-l3',(m3,e3))):
        process_chunk(ptag,f'world-chunk-{mc}.tar.gz',dset,pstores[0],rr_only=True)
        log('streamed',ptag,f'world-chunk-{mc}.tar.gz','elapsed',int(time.time()-t0),'s')
        if mc<63:
            process_chunk(ptag,f'world-chunk-{64+mc}.tar.gz',dset,pstores[1],rr_only=True)
            log('streamed',ptag,f'world-chunk-{64+mc}.tar.gz','elapsed',int(time.time()-t0),'s')
    flush_fit_days(dset);gc.collect()
if '2021-03-24' in fit_days:skipped.insert(0,('2021-03-24','special day not in l2/l3'))
log('fit days used:',len(FL),'skipped:',skipped)
n_days_used=len(FL)
F=np.concatenate(FL);RR=np.concatenate(RL);T=np.concatenate(TL);D=np.concatenate(DL)
NB1=np.concatenate(N1);NB2=np.concatenate(N2);NB3=np.concatenate(N3);RL2=np.concatenate(L2);RL3=np.concatenate(L3)
del FL,RL,TL,DL,N1,N2,N3,L2,L3
G=np.tile(reg_of[sub],len(RR)//nsub);E=np.tile(elig[sub],len(RR)//nsub)
gc.collect()
log('fit matrix',F.shape,'elapsed',int(time.time()-t0),'s')
from xgboost import XGBRegressor,XGBClassifier
def build_X(F,RR,G,qm,nb1,nb2,nb3,rl2,rl3):
    QM=np.zeros(len(RR),np.float32)
    for gi in range(len(REGIONS)):
        m=G==gi;mp=qm.get(REGIONS[gi]) or qm['global'];QM[m]=np.where(RR[m]==0,0,np.interp(RR[m],mp['source'],mp['target']))
    X=np.column_stack([F,RR,QM,np.log1p(RR),np.log1p(QM),nb1,nb2,nb3,rl2,rl3]).astype(np.float32)
    return X,QM
def fit_model(F,RR,T,G,nb1,nb2,nb3,rl2,rl3):
    wet=(RR>0)
    q=np.linspace(0,1,101);gl=(np.quantile(RR[wet],q),np.quantile(T[wet],q));qm={}
    for gi,r in enumerate(REGIONS):
        w=wet&(G==gi)
        s_,t_=(np.quantile(RR[w],q),np.quantile(T[w],q)) if w.sum()>=100 else gl
        qm[r]={'source':[float(x) for x in s_],'target':[float(x) for x in t_]}
    qm['global']={'source':[float(x) for x in gl[0]],'target':[float(x) for x in gl[1]]}
    X,QM=build_X(F[wet],RR[wet],G[wet],qm,nb1[wet],nb2[wet],nb3[wet],rl2[wet],rl3[wet])
    reg=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=.04,reg_lambda=50,min_child_weight=20,subsample=.85,colsample_bytree=.9,n_jobs=4,random_state=42,tree_method='hist').fit(X,(T[wet]-QM))
    yh=(T[wet]>=HEAVY).astype(int)
    clf=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,min_child_weight=5,subsample=.85,colsample_bytree=.9,scale_pos_weight=min(50,(len(yh)-yh.sum())/max(1,yh.sum()))**.5,n_jobs=4,random_state=42,tree_method='hist',eval_metric='logloss').fit(X,yh)
    return reg,clf,qm
F=F[E];RR=RR[E];T=T[E];G=G[E];D=D[E];NB1=NB1[E];NB2=NB2[E];NB3=NB3[E];RL2=RL2[E];RL3=RL3[E];del E;gc.collect()
log('fit matrix after cell-holdout filter',F.shape)
# --- pre-window out-of-fold calibration (K=5 interleaved day folds) ---
# QM/trees are refit per fold on the other 4 folds; affine terms and heavy thresholds
# are selected on pooled OUT-OF-FOLD predictions - honest pre-window estimates across
# all 5.4 years of seasons. The eval window is never touched for any choice.
ud=sorted(set(D.tolist()))
fmap={d:i%5 for i,d in enumerate(ud)}
fold=np.array([fmap[int(x)] for x in D],np.int8)
del D;gc.collect()
p0o=np.zeros(len(RR),np.float32);pro=np.zeros(len(RR),np.float32)
for k in range(5):
    tr=fold!=k;te=~tr
    regk,clfk,qmk=fit_model(F[tr],RR[tr],T[tr],G[tr],NB1[tr],NB2[tr],NB3[tr],RL2[tr],RL3[tr])
    Xk,QMk=build_X(F[te],RR[te],G[te],qmk,NB1[te],NB2[te],NB3[te],RL2[te],RL3[te])
    p0o[te]=np.maximum(0,QMk+regk.predict(Xk).astype(np.float32))
    pro[te]=clfk.predict_proba(Xk)[:,1]
    del regk,clfk,qmk,Xk,QMk,tr,te;gc.collect()
    log('oof fold',k,'done','elapsed',int(time.time()-t0),'s')
# final model on ALL fit rows
reg,clf,qm=fit_model(F,RR,T,G,NB1,NB2,NB3,RL2,RL3)
del F;gc.collect()
log('final model fitted','elapsed',int(time.time()-t0),'s')
# affine per region on pooled out-of-fold; kept only if pooled oof MAE improves
a_arr=np.ones(len(REGIONS),np.float32);b_arr=np.zeros(len(REGIONS),np.float32)
acc_aff={}
for gi,r in enumerate(REGIONS):
    m=G==gi
    if int(m.sum())<5000:acc_aff[r]=False;continue
    A=np.column_stack([p0o[m],np.ones(int(m.sum()),np.float32)])
    sol,_,_,_=np.linalg.lstsq(A,T[m],rcond=None)
    a=float(np.clip(sol[0],0.8,1.2));b=float(np.clip(sol[1],-2.0,2.0))
    mae0=float(abs(T[m]-np.maximum(0,p0o[m])).mean())
    mae1=float(abs(T[m]-np.maximum(0,a*p0o[m]+b)).mean())
    if mae1<mae0:a_arr[gi]=a;b_arr[gi]=b;acc_aff[r]=True
    else:acc_aff[r]=False
p1o=np.maximum(0,a_arr[G]*p0o+b_arr[G]);del p0o;gc.collect()
# heavy thresholds on pooled out-of-fold, WET ROWS ONLY (RR>0), matching live/eval
# inference (v4 selected over all rows; the dry-row flags it counted never occur in
# eval - fixed and disclosed). Cap = the gate itself: FA<=102% of raw GFS, max hits.
heavy_t=T>=HEAVY
wall=RR>0
THS=np.linspace(0.05,1.0,39)
def sweep_th(m,cap_ratio):
    m=m&wall
    raw_fa=int(((RR[m]>=HEAVY)&(~heavy_t[m])).sum());cap=cap_ratio*raw_fa
    p1m=p1o[m];prm=pro[m];htm=heavy_t[m]
    best=(-1,1.0)
    for th in THS:
        ph=(p1m>=HEAVY)|(prm>=th)
        fa=int((ph&~htm).sum())
        if fa<=cap:
            h=int((ph&htm).sum())
            if h>best[0]:best=(h,float(th))
    return best[0],best[1],raw_fa
MINCAL_TH=5000
hg,TG,rfa_g=sweep_th(np.ones(len(G),bool),1.02)
thr_arr=np.full(len(REGIONS),TG,np.float32)
calrep={'method':'out-of-fold 5-fold interleaved day folds; thresholds on pooled OOF over WET ROWS ONLY (RR>0), matching live/eval inference','global':{'th':float(TG),'oof_hits':int(hg),'oof_raw_fa':rfa_g}}
for gi,r in enumerate(REGIONS):
    m=G==gi
    if int((m&wall).sum())>=MINCAL_TH and int(((RR[m&wall]>=HEAVY)&(~heavy_t[m&wall])).sum())>=50:
        h,th,rfa=sweep_th(m,1.02)
        thr_arr[gi]=th
    else:
        h,th,rfa=-1,float(TG),-1
    calrep[r]={'th':float(thr_arr[gi]),'oof_hits':int(h),'oof_raw_fa':int(rfa),'affine':[float(a_arr[gi]),float(b_arr[gi])],'affine_accepted':bool(acc_aff[r])}
log('oof calibration',json.dumps(calrep)[:1800])
del G,RR,T,p1o,pro,heavy_t,fold,wall,NB1,NB2,NB3,RL2,RL3;gc.collect()
TH=float(TG)
log('calibrated, elapsed',int(time.time()-t0),'s')
# pass 2: eval window day-by-day (full 1M) - same pinned window as v4
need={}
for d in win_days:
    mc=full_days.index(d)//32
    for tag in ['data-days-world','data-days-world-l2','data-days-world-l3']:
        need.setdefault((tag,f'world-chunk-{mc}.tar.gz'),set()).add(d)
        need.setdefault((tag,f'world-chunk-{64+mc}.tar.gz'),set()).add(d)
stores={('data-days-world',0):m1,('data-days-world',1):e1,('data-days-world-l2',0):m2,('data-days-world-l2',1):e2,('data-days-world-l3',0):m3,('data-days-world-l3',1):e3}
for (tag,cname),dset in sorted(need.items()):
    ci=int(cname.split('-')[2].split('.')[0])
    process_chunk(tag,cname,dset,stores[(tag,1 if ci>=64 else 0)],tag!='data-days-world')
    log('streamed',tag,cname,'elapsed',int(time.time()-t0),'s')
_missing=[d for d in win_days if not(d in m1 and d in e1 and d in m2 and d in e2 and d in m3 and d in e3)]
assert not _missing,f'eval window days missing from corpora: {_missing}'
def met_acc():
    return {'n':0,'ae':0.0,'se':0.0,'h':0,'hh':0,'fa':0}
def add(m,y,p):
    m['n']+=len(y);m['ae']+=float(abs(y-p).sum());m['se']+=float(((y-p)**2).sum())
    h=y>=HEAVY;f=p>=HEAVY;m['h']+=int(h.sum());m['hh']+=int((h&f).sum());m['fa']+=int((~h&f).sum())
def fin(m):
    return dict(n=m['n'],mae=round(m['ae']/max(1,m['n']),4),rmse=round(math.sqrt(m['se']/max(1,m['n'])),4),heavy_days=m['h'],hits=m['hh'],false_alarms=m['fa'])
acc={k:met_acc() for k in ['raw','cand']}
hold={k:met_acc() for k in ['raw','cand']}
per=[[met_acc(),met_acc()] for _ in REGIONS]  # [raw,cand] per region
pkmax=0.0;finite_ok=True
for d in win_days:
    F=np.concatenate([m1[d][0],e1[d][0]]);RR=np.concatenate([m1[d][1],e1[d][1]]);T=np.concatenate([m1[d][2],e1[d][2]])
    r2=np.concatenate([m2[d],e2[d]]);r3=np.concatenate([m3[d],e3[d]])
    for st in (m1,e1,m2,e2,m3,e3):st.pop(d,None)
    mn,mx,wf=nbr_stats(RR)
    wet=RR>0
    X,QM=build_X(F[wet],RR[wet],reg_of[wet],qm,mn[wet],mx[wet],wf[wet],r2[wet],r3[wet])
    del F,mn,mx,wf,r2,r3
    Gw_=reg_of[wet]
    p0=np.maximum(0,QM+reg.predict(X).astype(np.float32))
    p1=np.maximum(0,a_arr[Gw_]*p0+b_arr[Gw_])
    pr=clf.predict_proba(X)[:,1]
    pw=np.where(pr>=thr_arr[Gw_],np.maximum(p1,HEAVY),p1)
    p=np.zeros(len(RR),np.float32);p[wet]=pw
    if not np.isfinite(p).all():finite_ok=False
    pkmax=max(pkmax,float(p.max()))
    add(acc['raw'],T,RR);add(acc['cand'],T,p)
    hm=~elig
    add(hold['raw'],T[hm],RR[hm]);add(hold['cand'],T[hm],p[hm])
    for gi in range(len(REGIONS)):
        m=reg_of==gi
        add(per[gi][0],T[m],RR[m]);add(per[gi][1],T[m],p[m])
    del RR,T,X,p;gc.collect()
raw=fin(acc['raw']);mk=fin(acc['cand'])
hold_m={k:fin(v) for k,v in hold.items()}
regm={REGIONS[gi]:{'raw':fin(per[gi][0]),'candidate':fin(per[gi][1])} for gi in range(len(REGIONS))}
gates=[]
def gate(name,ok,detail):gates.append({'gate':name,'pass':bool(ok),'detail':detail})
gate('evidence: enough heavy-rain point-days in window',mk['heavy_days']>=MIN_HEAVY,f"{mk['heavy_days']} heavy point-days (need {MIN_HEAVY})")
gate('all-site MAE better than raw GFS',mk['mae']<raw['mae'],f"{mk['mae']} vs raw {raw['mae']}")
gate('all-site heavy-rain hits >= raw GFS',mk['hits']>=raw['hits'],f"{mk['hits']} vs raw {raw['hits']}")
gate('all-site false alarms <= 102% of raw GFS',mk['false_alarms']<=1.02*raw['false_alarms'],f"{mk['false_alarms']} vs raw {raw['false_alarms']}")
gate('unseen-site MAE better than raw GFS',hold_m['cand']['mae']<hold_m['raw']['mae'],f"{hold_m['cand']['mae']} vs raw {hold_m['raw']['mae']}")
gate('unseen-site heavy hits >= 97% of raw GFS',hold_m['cand']['hits']>=0.97*hold_m['raw']['hits'],f"{hold_m['cand']['hits']} vs raw {hold_m['raw']['hits']}")
gate('unseen-site false alarms <= 102% of raw GFS',hold_m['cand']['false_alarms']<=1.02*hold_m['raw']['false_alarms'],f"{hold_m['cand']['false_alarms']} vs raw {hold_m['raw']['false_alarms']}")
for gi,rn in enumerate(REGIONS):
    rm_=regm[rn]['raw']['mae'];cm_=regm[rn]['candidate']['mae']
    ok=(cm_<=1.03*rm_) if rm_>=0.05 else (cm_<=rm_+0.02)
    gate(f'per-region MAE within 3% of raw GFS: {rn}',bool(ok),f'candidate {cm_} vs raw {rm_}')
gate('predictions finite and sane',bool(finite_ok and pkmax<2000),f"max {pkmax:.1f} mm")
out={'regions':REGIONS,'evaluation':{'window':f"{win_days[0]} to {win_days[-1]}",'raw':raw,'candidate':mk,'unseen_sites':hold_m,'per_region':regm},'gates':gates,'all_gates_pass':all(g['pass'] for g in gates),'fit':{'days':n_days_used,'days_skipped':[d for d,_ in skipped],'folds':5,'day_stride':FIT_DAY_STRIDE,'point_stride':FIT_POINT_STRIDE,'fit_through':win_start,'window_pinned_to':'world-v4 (2026-08-12..2026-10-12); 2026-10-13 backfilled after v4 and excluded from fit+eval','threshold':TH,'th_regions':{REGIONS[gi]:float(thr_arr[gi]) for gi in range(len(REGIONS))},'affine':{REGIONS[gi]:[float(a_arr[gi]),float(b_arr[gi])] for gi in range(len(REGIONS))},'features':'24 v4 base + 3 spatial (rr_nbr_mean/max/wetfrac over the 3x3 0.125-deg same-run neighborhood, valid neighbors only, own-RR fallback) + 2 prior-run (rr_l2, rr_l3 raw forecasts of the same target day from the 48h/72h runs)','holdout':'cell-level: whole ERA5 (rowE,colE) cells held out via crc32%8==7; ~12.5% of cells, no sibling leak; all new features are forecast-derived (never truth), so the holdout stays leak-free','calibration':'pre-window out-of-fold: 5 interleaved day folds over the fit period; QM/trees refit per fold on the other 4; affine kept per region only if pooled out-of-fold MAE improves; per-region heavy thresholds = max out-of-fold hits s.t. out-of-fold FA<=102% of raw GFS (the gate rule itself, on honest pre-window estimates), counted on WET ROWS ONLY (v4 counted all rows; fixed and disclosed); eval window never touched for any choice'},'note':'Richer-features worldwide fit (world-v6): v4 single-stage architecture + spatial and prior-run context. Champion baseline = raw GFS. Extra 0.125-deg points share 0.25-deg ERA5 truth cells (not independently verified). antarctica region: precip almost all snow, rain target ~always 0, climatologically quiet.'}
json.dump(out,open(f'{W}/world-fit-v6-metrics.json','w'),indent=1)
log(json.dumps(out['evaluation'],indent=1)[:2000])
log('GATES',json.dumps(gates,indent=1)[:1500])
# export model (run.py-compatible tree format + native boosters)
def tr(mod):
    cfg=json.loads(mod.get_booster().save_config());bs=float(str(cfg['learner']['learner_model_param']['base_score']).strip('[]'))
    j=json.loads(mod.get_booster().save_raw('json'))
    return bs,[{k:t[k] for k in ['left_children','right_children','split_indices','split_conditions','default_left']} for t in j['learner']['gradient_booster']['model']['trees']]
bs,rt=tr(reg);cb,ct=tr(clf)
model={'lead':1,'weight':1.0,'regions':REGIONS,'qm':qm,'base':bs,'trees':rt,'clf':{'base':math.log(cb/(1-cb)),'trees':ct,'threshold':TH,'th_regions':{REGIONS[gi]:float(thr_arr[gi]) for gi in range(len(REGIONS))}},'affine':{REGIONS[gi]:[float(a_arr[gi]),float(b_arr[gi])] for gi in range(len(REGIONS))},'features_v6':{'extra':['rr_nbr_mean','rr_nbr_max','rr_nbr_wetfrac','rr_l2','rr_l3'],'neighborhood':'3x3 at 0.125 deg, longitude wraps, valid neighbors only, own-RR fallback','prior_runs':'same target day, 48h and 72h runs, raw RR'},'fit_through':win_start,'version':'world-v6-'+win_days[-1]}
json.dump(model,open(f'{W}/world-lead1-model-v6.json','w'),separators=(',',':'))
reg.save_model(f'{W}/world-v6-reg.json');clf.save_model(f'{W}/world-v6-clf.json')
if os.environ.get('VD_UPLOAD','1')=='1':
    subprocess.run(['gh','release','view',os.environ.get('VD_OUT','world-fit-v6'),'--repo',REPO],capture_output=True)
    r=subprocess.run(['gh','release','view',os.environ.get('VD_OUT','world-fit-v6'),'--repo',REPO],capture_output=True)
    if r.returncode:subprocess.run(['gh','release','create',os.environ.get('VD_OUT','world-fit-v6'),'--repo',REPO,'--title','World fit v6 artifacts','--notes','Richer-features (spatial + prior-run) worldwide fit: model + metrics. Not deployed.'],check=True)
    subprocess.run(['gh','release','upload',os.environ.get('VD_OUT','world-fit-v6'),f'{W}/world-fit-v6-metrics.json',f'{W}/world-lead1-model-v6.json',f'{W}/world-v6-reg.json',f'{W}/world-v6-clf.json','--clobber','--repo',REPO],check=True)
    log('uploaded to world-fit-v6')
log('DONE all_gates_pass',out['all_gates_pass'])
