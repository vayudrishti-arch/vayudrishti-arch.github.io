"""VayuDrishti eval-ONLY re-run: worldwide lead-2 (v5 model) on the pinned window
2026-08-12..2026-10-12 with the SERVED abstain list (antarctica, middle_east, oceania,
west). west and oceania were added to the abstain list AFTER window results were seen
(l2-v5's west per-region MAE came in 10.5% worse than raw GFS in-window); the owner
approved serving the new model everywhere else and keeping the old behaviour (raw GFS)
in west/oceania on Oct 22, 2026. This script changes NO fitted constant: it loads the
released world-fit-l2-v5 model (booster files + calibration in world-lead2-model.json)
and re-evaluates the pinned window exactly as world/fit_worldl4.py's pass 2 does, with
the served abstain list applied. Gates are re-run on the served configuration.
Usage: VD_EVAL... not needed. GH_TOKEN for release upload. python3 eval_served_l2v5.py"""
import os,json,math,time,gc,io,tarfile,zipfile,urllib.request,subprocess
import numpy as np
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W='/tmp/evals';os.makedirs(W,exist_ok=True)
EVAL_DAYS=60;MIN_HEAVY=300;HEAVY=64.5
TAG='data-days-world-l2';LEAD=2;OUT='world-fit-l2-v5'
PRIORS=[('data-days-world-l3','rr_l3')]   # same as fit_worldl4.py for LEAD=2
ABSTAIN_SERV=['antarctica','middle_east','oceania','west']
WIN_START='2026-08-12';WIN_END='2026-10-12'   # pinned to world-v4's window
def log(*a):print(*a,flush=True)
def dl(url,path):
    if os.path.exists(path):return
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
def ghapi(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'vd-fit'}),timeout=120))
log('loading release index')
rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
urls={a['name']:a['browser_download_url'] for a in rel['assets']}
purls={}
for ptag,_ in PRIORS:
    rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{ptag}')
    purls[ptag]={a['name']:a['browser_download_url'] for a in rel['assets']}
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
dl(wurls['pointsw1m.json.gz'],f'{W}/pointsw1m.json.gz')
import gzip as gz
pts=json.load(gz.open(f'{W}/pointsw1m.json.gz'))
REGIONS=sorted(set(p['region'] for p in pts))
ridx={r:i for i,r in enumerate(REGIONS)}
reg_of=np.array([ridx[p['region']] for p in pts],np.int8)
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
del key2i;gc.collect()
def nbr_stats(RR):
    g=np.where(NBR>=0,RR[np.maximum(NBR,0)],np.nan).astype(np.float32)
    cnt=(~np.isnan(g)).sum(axis=1)
    mn=np.where(cnt>0,np.nanmean(g,axis=1),RR)
    mx=np.where(cnt>0,np.nanmax(g,axis=1),RR)
    wf=np.where(cnt>0,np.nansum(g>0.1,axis=1)/np.maximum(cnt,1),(RR>0.1).astype(np.float32))
    return mn,mx,wf.astype(np.float32)
dl(wurls['cellsw1m.json.gz'],f'{W}/cellsw1m.json.gz')
cellsj=json.load(gz.open(f'{W}/cellsw1m.json.gz'))
import zlib
cid=np.array([r*1440+c for r,c in zip(cellsj['rowE'],cellsj['colE'])],np.int64)
del cellsj;gc.collect()
uc=np.unique(cid)
hold_cells=set(int(c) for c in uc if zlib.crc32(int(c).to_bytes(8,'little'))%8==7)
elig=np.array([int(c) not in hold_cells for c in cid],bool)
log('cell-level holdout: pts held',int((~elig).sum()))
del cid,uc;gc.collect()
del pts;gc.collect()
alld=json.load(urllib.request.urlopen(wurls['alld_world.json'],timeout=60))
LAGGED={'2026-10-10','2026-10-11'}
full_days=[d for d in alld if d not in LAGGED]
win_days=[d for d in full_days if WIN_START<=d<=WIN_END]
assert len(win_days)==EVAL_DAYS,f'pinned window has {len(win_days)} days, expected {EVAL_DAYS}'
log('eval window (pinned to v4):',win_days[0],'to',win_days[-1],'days',len(win_days))
# --- load the fitted v5 model from the release (no constant changed) ---
mrel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{OUT}')
murls={a['name']:a['browser_download_url'] for a in mrel['assets']}
dl(murls[f'world-lead{LEAD}-model.json'],f'{W}/model.json')
modelj=json.load(open(f'{W}/model.json'))
qm=modelj['qm']
a_arr=np.array([modelj['affine'][r][0] for r in REGIONS],np.float32)
b_arr=np.array([modelj['affine'][r][1] for r in REGIONS],np.float32)
thr_arr=np.array([modelj['clf']['th_regions'][r] for r in REGIONS],np.float32)
dl(murls[f'world-l{LEAD}-reg.json'],f'{W}/reg-native.json')
dl(murls[f'world-l{LEAD}-clf.json'],f'{W}/clf-native.json')
from xgboost import XGBRegressor,XGBClassifier
reg=XGBRegressor();reg.load_model(f'{W}/reg-native.json')
clf=XGBClassifier();clf.load_model(f'{W}/clf-native.json')
abstain=np.array([r in ABSTAIN_SERV for r in REGIONS],bool)
log('fitted model abstain was:',modelj['abstain'],'| SERVED abstain now:',ABSTAIN_SERV)
def build_X(F,RR,G,qm,XE):
    QM=np.zeros(len(RR),np.float32)
    for gi in range(len(REGIONS)):
        m=G==gi;mp=qm.get(REGIONS[gi]) or qm['global'];QM[m]=np.where(RR[m]==0,0,np.interp(RR[m],mp['source'],mp['target']))
    X=np.column_stack([F,RR,QM,np.log1p(RR),np.log1p(QM),XE]).astype(np.float32)
    return X,QM
def eat_npz(b,date,dstore,rr_only=False):
    z=zipfile.ZipFile(io.BytesIO(b))
    if rr_only:
        RR=np.load(io.BytesIO(z.read('RR.npy')));dstore[date]=RR;return
    F=np.load(io.BytesIO(z.read('F.npy')));RR=np.load(io.BytesIO(z.read('RR.npy')));T=np.load(io.BytesIO(z.read('T.npy')))
    dstore[date]=(F,RR,T)
def process_chunk(tag,cname,dayset,store,rr_only=False):
    u=(urls if tag==TAG else purls[tag])[cname]
    dl(u,f'{W}/{tag}-{cname}')
    tf=tarfile.open(f'{W}/{tag}-{cname}',mode='r:gz')
    for m in tf.getmembers():
        d=m.name[:10]
        if d in dayset and d not in store:
            eat_npz(tf.extractfile(m).read(),d,store,rr_only)
    tf.close();os.remove(f'{W}/{tag}-{cname}')
# pass 2: eval window day-by-day (full 1M) - identical to fit_worldl4.py, served abstain list
need={}
for d in win_days:
    mc=full_days.index(d)//32
    need.setdefault((TAG,f'world-chunk-{mc}.tar.gz'),set()).add(d);need.setdefault((TAG,f'world-chunk-{64+mc}.tar.gz'),set()).add(d)
    for ptag,_ in PRIORS:
        need.setdefault((ptag,f'world-chunk-{mc}.tar.gz'),set()).add(d);need.setdefault((ptag,f'world-chunk-{64+mc}.tar.gz'),set()).add(d)
ev_main={};ev_ext={}
ev_pri={(ptag,0):{} for ptag,_ in PRIORS};ev_pri.update({(ptag,1):{} for ptag,_ in PRIORS})
wcids=sorted({int(c.split('-')[2].split('.')[0])%64 for (tg,c) in need})
win_chunks=[(ci,sorted(need.get((TAG,f'world-chunk-{ci}.tar.gz'),()))) for ci in wcids]
def met_acc():
    return {'n':0,'ae':0.0,'se':0.0,'h':0,'hh':0,'fa':0}
def add(m,y,p):
    m['n']+=len(y);m['ae']+=float(abs(y-p).sum());m['se']+=float(((y-p)**2).sum())
    h=y>=HEAVY;f=p>=HEAVY;m['h']+=int(h.sum());m['hh']+=int((h&f).sum());m['fa']+=int((~h&f).sum())
def fin(m):
    return dict(n=m['n'],mae=round(m['ae']/max(1,m['n']),4),rmse=round(math.sqrt(m['se']/max(1,m['n'])),4),heavy_days=m['h'],hits=m['hh'],false_alarms=m['fa'])
acc={k:met_acc() for k in ['raw','cand']}
hold={k:met_acc() for k in ['raw','cand']}
per=[[met_acc(),met_acc()] for _ in REGIONS]
pkmax=0.0;finite_ok=True
for ci,cdays in win_chunks:
    cn=f'world-chunk-{ci}.tar.gz';en=f'world-chunk-{64+ci}.tar.gz'
    if (TAG,cn) in need:process_chunk(TAG,cn,need[(TAG,cn)],ev_main);log('streamed',TAG,cn,'elapsed',int(time.time()-t0),'s')
    if (TAG,en) in need:process_chunk(TAG,en,need[(TAG,en)],ev_ext);log('streamed',TAG,en,'elapsed',int(time.time()-t0),'s')
    for ptag,_ in PRIORS:
        if (ptag,cn) in need:process_chunk(ptag,cn,need[(ptag,cn)],ev_pri[(ptag,0)],rr_only=True);log('streamed',ptag,cn,'elapsed',int(time.time()-t0),'s')
        if (ptag,en) in need:process_chunk(ptag,en,need[(ptag,en)],ev_pri[(ptag,1)],rr_only=True);log('streamed',ptag,en,'elapsed',int(time.time()-t0),'s')
    for d in cdays:
        for ptag,_ in PRIORS:
            assert d in ev_pri[(ptag,0)] and d in ev_pri[(ptag,1)],f'eval day {d} missing prior {ptag}'
        Fm,RRm,Tm=ev_main[d];Fe,RRe,Te=ev_ext[d]
        F=np.concatenate([Fm,Fe]);RR=np.concatenate([RRm,RRe]);T=np.concatenate([Tm,Te])
        del ev_main[d];del ev_ext[d]
        prs=[np.concatenate([ev_pri[(ptag,0)][d],ev_pri[(ptag,1)][d]]) for ptag,_ in PRIORS]
        for ptag,_ in PRIORS:ev_pri[(ptag,0)].pop(d,None);ev_pri[(ptag,1)].pop(d,None)
        mn,mx,wf=nbr_stats(RR)
        wet=RR>0
        XE=np.column_stack([mn[wet],mx[wet],wf[wet]]+[p[wet] for p in prs]).astype(np.float32)
        del prs,mn,mx,wf
        X,QM=build_X(F[wet],RR[wet],reg_of[wet],qm,XE)
        Gw_=reg_of[wet]
        p0=np.maximum(0,QM+reg.predict(X).astype(np.float32))
        p1=np.maximum(0,a_arr[Gw_]*p0+b_arr[Gw_])
        pr=clf.predict_proba(X)[:,1]
        pw=np.where(pr>=thr_arr[Gw_],np.maximum(p1,HEAVY),p1)
        pw=np.where(abstain[Gw_],RR[wet],pw)   # served abstain list: these regions serve raw GFS (amount and flag)
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
out={'regions':REGIONS,'served_config_eval':{'model':modelj['version'],'abstain_regions_served':ABSTAIN_SERV,'disclosure':'eval-only re-run of the released world-fit-l2-v5 model on the pinned window (2026-08-12..2026-10-12) with the SERVED abstain list; west and oceania added after window results were seen (west failed the 3% per-region gate in-window), disclosed on the site and approved by the owner Oct 22, 2026. No fitted constant changed.'},'evaluation':{'window':f"{win_days[0]} to {win_days[-1]} (pinned to world-v4)",'raw':raw,'candidate':mk,'unseen_sites':hold_m,'per_region':regm},'gates':gates,'all_gates_pass':all(g['pass'] for g in gates)}
json.dump(out,open(f'{W}/served-metrics.json','w'),indent=1)
log(json.dumps(out['evaluation'],indent=1)[:2000])
log('GATES',json.dumps(gates,indent=1)[:1500])
if os.environ.get('VD_UPLOAD','1')=='1':
    subprocess.run(['cp',f'{W}/served-metrics.json',f'{W}/world-fit-l{LEAD}-served-metrics.json'],check=True)
    subprocess.run(['gh','release','upload',OUT,f'{W}/world-fit-l{LEAD}-served-metrics.json','--clobber','--repo',REPO],check=True)
    log('uploaded served-config metrics to',OUT)
log('DONE all_gates_pass',out['all_gates_pass'])
