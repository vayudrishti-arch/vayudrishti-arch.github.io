"""VayuDrishti worldwide first fit + gates (1M grid). Streams corpus chunks from the
data-days-world release, fits on a stratified subsample (every FIT_POINT_STRIDE-th point x
every FIT_DAY_STRIDE-th day), evaluates on the rolling 60-day window with xgboost native
predict, gates vs raw GFS (first fit: raw is the champion baseline). Uploads model +
metrics to release world-fit. Nothing public. Usage: python3 fit_world.py"""
import os,json,math,time,gc,io,tarfile,zipfile,datetime as dt,urllib.request,subprocess
import numpy as np
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W='/tmp/fitw';os.makedirs(W,exist_ok=True)
EVAL_DAYS=60;MIN_HEAVY=300;HEAVY=64.5
FIT_DAY_STRIDE=16;FIT_POINT_STRIDE=4
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
rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/data-days-world')
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
LAGGED={'2026-10-10','2026-10-11','2026-10-13'}
full_days=[d for d in alld if d not in LAGGED]          # 2,024 days
built_days=sorted(full_days+['2021-03-24'])             # 2,025 days
win_start=built_days[-EVAL_DAYS]
log('eval window starts',win_start,'days total',len(built_days))
fit_days=set(d for i,d in enumerate(built_days) if d<win_start and i%FIT_DAY_STRIDE==0)
log('fit days:',len(fit_days))
# day -> (main_chunk, needs_ext)
def main_chunk(d): return None if d=='2021-03-24' else full_days.index(d)//32
# pass 1: accumulate fit subsample
sub=np.zeros(1000000,bool);sub[::FIT_POINT_STRIDE]=True
nsub=int(sub.sum())
FL=[];RL=[];TL=[]
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
    else:continue
    FL.append(F[sub]);RL.append(RR[sub]);TL.append(T[sub])
    del F,RR,T
    if d in main_days:del main_days[d]
    if d in ext_days:del ext_days[d]
    if len(FL)%20==0:gc.collect();log('fit buf days',len(FL),'elapsed',int(time.time()-t0),'s')
F=np.concatenate(FL);RR=np.concatenate(RL);T=np.concatenate(TL);del FL,RL,TL;gc.collect()
G=np.tile(reg_of[sub],len(fit_days));E=np.tile(elig[sub],len(fit_days))
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
    # threshold: max F1 on a pre-window subsample (pre-window choice, no window tuning)
    idx=np.random.RandomState(42).choice(len(X),min(2000000,len(X)),replace=False)
    pr=clf.predict_proba(X[idx])[:,1];yh_s=yh[idx]
    best=(0,0.5)
    for th in np.linspace(0.05,0.95,37):
        p=pr>=th;tp=int((p&(yh_s==1)).sum());fp=int((p&(yh_s==0)).sum());fn=int(((~p)&(yh_s==1)).sum())
        f1=2*tp/(2*tp+fp+fn) if tp else 0
        if f1>best[0]:best=(f1,float(th))
    log('threshold calib F1',best)
    return reg,clf,qm,best[1]
F=F[E];RR=RR[E];T=T[E];G=G[E];del E;gc.collect()
log('fit matrix after cell-holdout filter',F.shape)
reg,clf,qm,TH=fit_model(F,RR,T,G)
del F,RR,T,G;gc.collect()
log('model fitted, TH',TH,'elapsed',int(time.time()-t0),'s')
# pass 2: eval window day-by-day (full 1M)
win_days=built_days[-EVAL_DAYS:]
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
    p0=np.maximum(0,QM+reg.predict(X).astype(np.float32))
    pr=clf.predict_proba(X)[:,1]
    pw=np.where(pr>=TH,np.maximum(p0,HEAVY),p0)
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
out={'regions':REGIONS,'evaluation':{'window':f"{win_days[0]} to {win_days[-1]}",'raw':raw,'candidate':mk,'unseen_sites':hold_m,'per_region':regm},'gates':gates,'all_gates_pass':all(g['pass'] for g in gates),'fit':{'days':len(fit_days),'day_stride':FIT_DAY_STRIDE,'point_stride':FIT_POINT_STRIDE,'fit_through':win_start,'threshold':TH,'holdout':'cell-level: whole ERA5 (rowE,colE) cells held out via crc32%8==7; ~12.5% of cells, no sibling leak'},'note':'First worldwide fit; champion baseline = raw GFS. Extra 0.125-deg points share 0.25-deg ERA5 truth cells (not independently verified). antarctica region: precip almost all snow, rain target ~always 0, climatologically quiet.'}
json.dump(out,open(f'{W}/world-fit-metrics.json','w'),indent=1)
log(json.dumps(out['evaluation'],indent=1)[:2000])
log('GATES',json.dumps(gates,indent=1)[:1500])
# export model (run.py-compatible tree format + native boosters)
def tr(mod):
    cfg=json.loads(mod.get_booster().save_config());bs=float(str(cfg['learner']['learner_model_param']['base_score']).strip('[]'))
    j=json.loads(mod.get_booster().save_raw('json'))
    return bs,[{k:t[k] for k in ['left_children','right_children','split_indices','split_conditions','default_left']} for t in j['learner']['gradient_booster']['model']['trees']]
bs,rt=tr(reg);cb,ct=tr(clf)
model={'lead':1,'weight':1.0,'regions':REGIONS,'qm':qm,'base':bs,'trees':rt,'clf':{'base':math.log(cb/(1-cb)),'trees':ct,'threshold':TH},'fit_through':win_start,'version':'world-v1-'+win_days[-1]}
json.dump(model,open(f'{W}/world-lead1-model.json','w'),separators=(',',':'))
reg.save_model(f'{W}/world-reg.json');clf.save_model(f'{W}/world-clf.json')
if os.environ.get('VD_UPLOAD','1')=='1':
    subprocess.run(['gh','release','view','world-fit','--repo',REPO],capture_output=True)
    r=subprocess.run(['gh','release','view','world-fit','--repo',REPO],capture_output=True)
    if r.returncode:subprocess.run(['gh','release','create','world-fit','--repo',REPO,'--title','World fit artifacts','--notes','First worldwide fit: model + metrics. Not deployed.'],check=True)
    subprocess.run(['gh','release','upload','world-fit',f'{W}/world-fit-metrics.json',f'{W}/world-lead1-model.json',f'{W}/world-reg.json',f'{W}/world-clf.json','--clobber','--repo',REPO],check=True)
    log('uploaded to world-fit')
log('DONE all_gates_pass',out['all_gates_pass'])
