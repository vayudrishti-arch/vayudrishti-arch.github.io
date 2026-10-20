"""VayuDrishti worldwide lead-parameterized fit + gates (1M grid). Same pipeline and
honesty framework as fit_world.py (world-v4): stratified subsample, cell-level holdout,
5-fold out-of-fold calibration, gates vs raw GFS. Differences: corpus tag, lead and
output release come from env (VD_TAG / VD_LEAD / VD_OUT); the eval window is PINNED to
world-v4's (2026-08-12..2026-10-12) for cross-lead comparability; threshold selection
counts flags on WET ROWS ONLY (RR>0), matching live/eval inference (v4's all-rows
counting included dry-row flags the eval never produces - fixed and disclosed). Days
missing from the corpus (early-2021 GFS archive edge, ERA5-lagged days) are skipped and
disclosed in the metrics. Nothing public. Usage: VD_LEAD=2 VD_TAG=data-days-world-l2
VD_OUT=world-fit-l2 python3 fit_worldl.py"""
import os,json,math,time,gc,io,tarfile,zipfile,datetime as dt,urllib.request,subprocess
import numpy as np
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W='/tmp/fitw';os.makedirs(W,exist_ok=True)
EVAL_DAYS=60;MIN_HEAVY=300;HEAVY=64.5
FIT_DAY_STRIDE=16;FIT_POINT_STRIDE=4
TAG=os.environ.get('VD_TAG','data-days-world');LEAD=int(os.environ.get('VD_LEAD','1'));OUT=os.environ.get('VD_OUT','world-fit')
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
log('loading release index')
import re
def ghapi(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'vd-fit'}),timeout=120))
rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
urls={a['name']:a['browser_download_url'] for a in rel['assets']}
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
dl(wurls['pointsw1m.json.gz'],f'{W}/pointsw1m.json.gz')
import gzip as gz
pts=json.load(gz.open(f'{W}/pointsw1m.json.gz'))
REGIONS=sorted(set(p['region'] for p in pts))
log('regions:',REGIONS)
ridx={r:i for i,r in enumerate(REGIONS)}
reg_of=np.array([ridx[p['region']] for p in pts],np.int8)
# holdout must be cell-level: sibling points share one 0.25-deg ERA5 truth cell,
# so a point-level split leaks truth cells across fit/holdout. Whole-cell split.
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
del pts;gc.collect()
alld=json.load(urllib.request.urlopen(wurls['alld_world.json'],timeout=60))
LAGGED={'2026-10-10','2026-10-11'}   # still ERA5-lagged; 2026-10-13 present but pinned out of fit+eval
full_days=[d for d in alld if d not in LAGGED]
HAS_FULLGRID='world-chunk-128.tar.gz' in urls   # 2021-03-24 exists only as a full-grid day in the l1 corpus; its lead-2/3 runs predate the free GFS archive
built_days=sorted(full_days+(['2021-03-24'] if HAS_FULLGRID else []))
win_days=[d for d in built_days if WIN_START<=d<=WIN_END]
assert len(win_days)==EVAL_DAYS,f'pinned window has {len(win_days)} days, expected {EVAL_DAYS}'
win_start=WIN_START
log('eval window (pinned to v4):',win_days[0],'to',win_days[-1],'days',len(win_days),'corpus days total',len(built_days))
fit_days=set(d for i,d in enumerate(built_days) if d<win_start and i%FIT_DAY_STRIDE==0)
log('fit days:',len(fit_days))
# day -> (main_chunk, needs_ext)
def main_chunk(d): return None if d=='2021-03-24' else full_days.index(d)//32
# pass 1: accumulate fit subsample
sub=np.zeros(1000000,bool);sub[::FIT_POINT_STRIDE]=True
nsub=int(sub.sum())
FL=[];RL=[];TL=[];DL=[]
fit_days_sorted=sorted(fit_days)
log('fit days',len(fit_days_sorted))
t0=time.time()
def eat_npz(b,date,dstore):
    z=zipfile.ZipFile(io.BytesIO(b))
    F=np.load(io.BytesIO(z.read('F.npy')));RR=np.load(io.BytesIO(z.read('RR.npy')));T=np.load(io.BytesIO(z.read('T.npy')))
    dstore[date]=(F,RR,T)
def process_chunk(cname,dayset,store):
    dl(urls[cname],f'{W}/{cname}')
    tf=tarfile.open(f'{W}/{cname}',mode='r:gz')
    for m in tf.getmembers():
        d=m.name[:10]
        if d in dayset and d not in store:
            eat_npz(tf.extractfile(m).read(),d,store)
    tf.close();os.remove(f'{W}/{cname}')
main_days={}
need_main={}
skipped=[]
if not HAS_FULLGRID:skipped.append(f'2021-03-24 (no lead-{LEAD} run in GFS archive; full-grid day exists only at lead-1)')
for d in fit_days:
    mc=main_chunk(d)
    if mc is None:continue
    need_main.setdefault(f'world-chunk-{mc}.tar.gz',set()).add(d)
    if mc is not None and mc<63:need_main.setdefault(f'world-chunk-{64+mc}.tar.gz',set()).add(d)  # ext pair
if '2021-03-24' in fit_days:need_main.setdefault('world-chunk-128.tar.gz',set()).add('2021-03-24')
# build paired 1M day rows: main + ext per date
ext_days={}
for cname,dset in need_main.items():
    process_chunk(cname,dset,ext_days if int(cname.split('-')[2].split('.')[0])>=64 else main_days)
    log('streamed',cname,'elapsed',int(time.time()-t0),'s')
# pair and subsample
n_main=980692
for d in sorted(fit_days):
    if d in main_days and d in ext_days:
        F=np.concatenate([main_days[d][0],ext_days[d][0]]);RR=np.concatenate([main_days[d][1],ext_days[d][1]]);T=np.concatenate([main_days[d][2],ext_days[d][2]])
    elif d in main_days:
        F,RR,T=main_days[d]
        if len(RR)==n_main:continue  # missing ext pair (should not happen for fit days)
    elif d in ext_days and len(ext_days[d][1])==1000000:
        F,RR,T=ext_days[d]  # chunk-128 full-grid day
    else:
        skipped.append(d);continue
    FL.append(F[sub]);RL.append(RR[sub]);TL.append(T[sub]);DL.append(np.full(nsub,int(d.replace('-','')),np.int32))
    del F,RR,T
    if d in main_days:del main_days[d]
    if d in ext_days:del ext_days[d]
    if len(FL)%20==0:gc.collect();log('fit buf days',len(FL),'elapsed',int(time.time()-t0),'s')
log('fit days used:',len(FL),'skipped:',skipped)
n_days_used=len(FL)
F=np.concatenate(FL);RR=np.concatenate(RL);T=np.concatenate(TL);D=np.concatenate(DL);del FL,RL,TL,DL
G=np.tile(reg_of[sub],len(RR)//nsub);E=np.tile(elig[sub],len(RR)//nsub)
gc.collect()
log('fit matrix',F.shape,'elapsed',int(time.time()-t0),'s')
from xgboost import XGBRegressor,XGBClassifier
def build_X(F,RR,G,qm):
    QM=np.zeros(len(RR),np.float32)
    for gi in range(len(REGIONS)):
        m=G==gi;mp=qm.get(REGIONS[gi]) or qm['global'];QM[m]=np.where(RR[m]==0,0,np.interp(RR[m],mp['source'],mp['target']))
    X=np.column_stack([F,RR,QM,np.log1p(RR),np.log1p(QM)]).astype(np.float32)
    return X,QM
def fit_model(F,RR,T,G,qm_days_mask=None):
    wet=(RR>0)
    q=np.linspace(0,1,101);gl=(np.quantile(RR[wet],q),np.quantile(T[wet],q));qm={}
    for gi,r in enumerate(REGIONS):
        w=wet&(G==gi)
        s_,t_=(np.quantile(RR[w],q),np.quantile(T[w],q)) if w.sum()>=100 else gl
        qm[r]={'source':[float(x) for x in s_],'target':[float(x) for x in t_]}
    qm['global']={'source':[float(x) for x in gl[0]],'target':[float(x) for x in gl[1]]}
    X,QM=build_X(F[wet],RR[wet],G[wet],qm)
    reg=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=.04,reg_lambda=50,min_child_weight=20,subsample=.85,colsample_bytree=.9,n_jobs=4,random_state=42,tree_method='hist').fit(X,(T[wet]-QM))
    yh=(T[wet]>=HEAVY).astype(int)
    clf=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,min_child_weight=5,subsample=.85,colsample_bytree=.9,scale_pos_weight=min(50,(len(yh)-yh.sum())/max(1,yh.sum()))**.5,n_jobs=4,random_state=42,tree_method='hist',eval_metric='logloss').fit(X,yh)
    return reg,clf,qm
F=F[E];RR=RR[E];T=T[E];G=G[E];D=D[E];del E;gc.collect()
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
    regk,clfk,qmk=fit_model(F[tr],RR[tr],T[tr],G[tr])
    Xk,QMk=build_X(F[te],RR[te],G[te],qmk)
    p0o[te]=np.maximum(0,QMk+regk.predict(Xk).astype(np.float32))
    pro[te]=clfk.predict_proba(Xk)[:,1]
    del regk,clfk,qmk,Xk,QMk,tr,te;gc.collect()
    log('oof fold',k,'done','elapsed',int(time.time()-t0),'s')
# final model on ALL fit rows
reg,clf,qm=fit_model(F,RR,T,G)
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
# inference (v4 selected over all rows; fixed and disclosed). Cap = the gate itself.
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
calrep={'method':'out-of-fold 5-fold interleaved day folds','global':{'th':float(TG),'oof_hits':int(hg),'oof_raw_fa':rfa_g}}
for gi,r in enumerate(REGIONS):
    m=G==gi
    if int((m&wall).sum())>=MINCAL_TH and int(((RR[m&wall]>=HEAVY)&(~heavy_t[m&wall])).sum())>=50:
        h,th,rfa=sweep_th(m,1.02)
        thr_arr[gi]=th
    else:
        h,th,rfa=-1,float(TG),-1
    calrep[r]={'th':float(thr_arr[gi]),'oof_hits':int(h),'oof_raw_fa':int(rfa),'affine':[float(a_arr[gi]),float(b_arr[gi])],'affine_accepted':bool(acc_aff[r])}
log('oof calibration',json.dumps(calrep)[:1800])
del G,RR,T,p1o,pro,heavy_t,fold,wall;gc.collect()
TH=float(TG)
log('calibrated, elapsed',int(time.time()-t0),'s')
# pass 2: eval window day-by-day (full 1M)
need={}
for d in win_days:
    mc=full_days.index(d)//32
    need.setdefault(f'world-chunk-{mc}.tar.gz',set()).add(d);need.setdefault(f'world-chunk-{64+mc}.tar.gz',set()).add(d)
ev_main={};ev_ext={}
for cname,dset in sorted(need.items()):
    ci=int(cname.split('-')[2].split('.')[0])
    process_chunk(cname,dset,ev_main if ci<64 else ev_ext)
    log('streamed',cname,'elapsed',int(time.time()-t0),'s')
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
    Fm,RRm,Tm=ev_main[d];Fe,RRe,Te=ev_ext[d]
    F=np.concatenate([Fm,Fe]);RR=np.concatenate([RRm,RRe]);T=np.concatenate([Tm,Te])
    del ev_main[d];del ev_ext[d]
    wet=RR>0
    X,QM=build_X(F[wet],RR[wet],reg_of[wet],qm)
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
    del F,RR,T,X,p;gc.collect()
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
out={'regions':REGIONS,'evaluation':{'window':f"{win_days[0]} to {win_days[-1]} (pinned to world-v4)",'raw':raw,'candidate':mk,'unseen_sites':hold_m,'per_region':regm},'gates':gates,'all_gates_pass':all(g['pass'] for g in gates),'fit':{'days':n_days_used,'days_skipped':skipped,'folds':5,'day_stride':FIT_DAY_STRIDE,'point_stride':FIT_POINT_STRIDE,'fit_through':win_start,'threshold':TH,'th_regions':{REGIONS[gi]:float(thr_arr[gi]) for gi in range(len(REGIONS))},'affine':{REGIONS[gi]:[float(a_arr[gi]),float(b_arr[gi])] for gi in range(len(REGIONS))},'holdout':'cell-level: whole ERA5 (rowE,colE) cells held out via crc32%8==7; ~12.5% of cells, no sibling leak','calibration':'pre-window out-of-fold: 5 interleaved day folds over the fit period; QM/trees refit per fold on the other 4; affine kept per region only if pooled out-of-fold MAE improves; per-region heavy thresholds = max out-of-fold hits s.t. out-of-fold FA<=102% of raw GFS (the gate rule itself, on honest pre-window estimates), counted on WET ROWS ONLY (v4 counted all rows; fixed and disclosed); eval window never touched for any choice'},'note':f'Worldwide lead-{LEAD} fit; champion baseline = raw GFS. Extra 0.125-deg points share 0.25-deg ERA5 truth cells (not independently verified). antarctica region: precip almost all snow, rain target ~always 0, climatologically quiet.'}
json.dump(out,open(f'{W}/metrics.json','w'),indent=1)
log(json.dumps(out['evaluation'],indent=1)[:2000])
log('GATES',json.dumps(gates,indent=1)[:1500])
# export model (run.py-compatible tree format + native boosters)
def tr(mod):
    cfg=json.loads(mod.get_booster().save_config());bs=float(str(cfg['learner']['learner_model_param']['base_score']).strip('[]'))
    j=json.loads(mod.get_booster().save_raw('json'))
    return bs,[{k:t[k] for k in ['left_children','right_children','split_indices','split_conditions','default_left']} for t in j['learner']['gradient_booster']['model']['trees']]
bs,rt=tr(reg);cb,ct=tr(clf)
model={'lead':LEAD,'weight':1.0,'regions':REGIONS,'qm':qm,'base':bs,'trees':rt,'clf':{'base':math.log(cb/(1-cb)),'trees':ct,'threshold':TH,'th_regions':{REGIONS[gi]:float(thr_arr[gi]) for gi in range(len(REGIONS))}},'affine':{REGIONS[gi]:[float(a_arr[gi]),float(b_arr[gi])] for gi in range(len(REGIONS))},'fit_through':win_start,'version':f'world-l{LEAD}-v1-'+win_days[-1]}
json.dump(model,open(f'{W}/model.json','w'),separators=(',',':'))
reg.save_model(f'{W}/reg-native.json');clf.save_model(f'{W}/clf-native.json')
if os.environ.get('VD_UPLOAD','1')=='1':
    r=subprocess.run(['gh','release','view',OUT,'--repo',REPO],capture_output=True)
    if r.returncode:subprocess.run(['gh','release','create',OUT,'--repo',REPO,'--title',f'World fit lead-{LEAD} artifacts','--notes',f'Worldwide lead-{LEAD} fit: model + metrics. Not deployed.'],check=True)
    names=[(f'{W}/metrics.json',f'world-fit-l{LEAD}-metrics.json'),(f'{W}/model.json',f'world-lead{LEAD}-model.json'),(f'{W}/reg-native.json',f'world-l{LEAD}-reg.json'),(f'{W}/clf-native.json',f'world-l{LEAD}-clf.json')]
    for src,dst in names:
        subprocess.run(['cp',src,f'{W}/{dst}'],check=True)
    subprocess.run(['gh','release','upload',OUT]+[f'{W}/{d}' for _,d in names]+['--clobber','--repo',REPO],check=True)
    log('uploaded to',OUT)
log('DONE all_gates_pass',out['all_gates_pass'])
