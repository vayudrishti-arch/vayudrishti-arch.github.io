"""VayuDrishti ensemble eval B (research only, eval-only, no deploy).
Averages the corrections of two already-fitted, already-reported models per lead:
  A = live v5   (release world-fit-l{L}-v5)
  B = 4x trial  (release world-fit-l{L}-4x, 489-day fit)
Pipelines: lead-2 parents are both fit_worldl4.py (3 spatial + rr_l3 prior); the lead-3
v5 parent is the older base-feature fit_worldl.py architecture while the lead-3 4x parent
is fit_worldl4.py (3 spatial) - each model is scored with its own fitted feature layout.
Only the fit-day stride differed within each pair (16 vs 4). This script replays the PINNED eval window
2026-08-12..2026-10-12 (60 days, same LAGGED exclusions as every prior eval) through
the SAME pass-2 gate harness as fit_worldl4.py, and scores four candidates side by
side: raw GFS, v5 alone, 4x alone, and the ensemble.

PRE-DECLARED combination rule (fixed before any window number is seen in this run):
  amount    = 0.5 * (p1_v5 + p1_4x)            (post-affine corrected mm)
  heavy p   = 0.5 * (p_v5 + p_4x);  threshold  = 0.5 * (th_v5 + th_4x) per region
  heavy bump: where(p>=th, max(amount, HEAVY), amount)   (same rule as the harness)
  abstain   = UNION of both models' abstain regions -> raw GFS served there
Nothing is tuned on the window; if the ensemble does not beat both parents on the
gates, that is reported as-is. Usage:
  VD_LEAD=2 python3 eval_ens.py   (or VD_LEAD=3)
"""
import os,json,math,time,gc,io,tarfile,zipfile,datetime as dt,urllib.request,subprocess
import numpy as np
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W='/tmp/enseval';os.makedirs(W,exist_ok=True)
EVAL_DAYS=60;MIN_HEAVY=300;HEAVY=64.5
LEAD=int(os.environ.get('VD_LEAD','2'))
TAG=f'data-days-world-l{LEAD}'
PRIORS={2:[('data-days-world-l3','rr_l3')],3:[]}[LEAD]
ENAMES=['rr_nbr_mean','rr_nbr_max','rr_nbr_wetfrac']+[n for _,n in PRIORS]
WIN_START='2026-08-12';WIN_END='2026-10-12'
RELS={'v5':f'world-fit-l{LEAD}-v5','4x':f'world-fit-l{LEAD}-4x'}
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
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'vd-ens'}),timeout=120))
log('loading release indexes')
rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}')
urls={a['name']:a['browser_download_url'] for a in rel['assets']}
purls={}
for ptag,_ in PRIORS:
    rel=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{ptag}')
    purls[ptag]={a['name']:a['browser_download_url'] for a in rel['assets']}
wa=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/world-assets')
wurls={a['name']:a['browser_download_url'] for a in wa['assets']}
# --- load the two models (run.py-format JSON for qm/affine/thresholds/abstain + native boosters) ---
import xgboost as xgb
models={}
for key,rtag in RELS.items():
    r=ghapi(f'https://api.github.com/repos/{REPO}/releases/tags/{rtag}')
    au={a['name']:a['browser_download_url'] for a in r['assets']}
    mj=f'world-lead{LEAD}-model.json';rg=f'world-l{LEAD}-reg.json';cf=f'world-l{LEAD}-clf.json'
    for need in (mj,rg,cf):
        assert need in au,f'{rtag} missing asset {need}: {sorted(au)}'
    dl(au[mj],f'{W}/{key}-model.json');dl(au[rg],f'{W}/{key}-reg.json');dl(au[cf],f'{W}/{key}-clf.json')
    m=json.load(open(f'{W}/{key}-model.json'))
    reg=xgb.Booster();reg.load_model(f'{W}/{key}-reg.json')
    clf=xgb.Booster();clf.load_model(f'{W}/{key}-clf.json')
    got=(m.get('features_extra') or {}).get('names') or []
    assert all(n in ENAMES for n in got),f'{rtag} unknown features: {got}'
    models[key]={'spec':m,'reg':reg,'clf':clf,'extras':got}
    log(key,'model loaded from',rtag,'version',m.get('version'))
import gzip as gz
dl(wurls['pointsw1m.json.gz'],f'{W}/pointsw1m.json.gz')
pts=json.load(gz.open(f'{W}/pointsw1m.json.gz'))
REGIONS=sorted(set(p['region'] for p in pts))
for key,mo in models.items():
    assert mo['spec']['regions']==REGIONS,f'{key} region order mismatch'
ridx={r:i for i,r in enumerate(REGIONS)}
reg_of=np.array([ridx[p['region']] for p in pts],np.int8)
log('building neighbor index')
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
del cid,uc;gc.collect()
del pts;gc.collect()
alld=json.load(urllib.request.urlopen(wurls['alld_world.json'],timeout=60))
LAGGED={'2026-10-10','2026-10-11'}   # identical exclusions as every prior eval on this pinned window
full_days=[d for d in alld if d not in LAGGED]
win_days=[d for d in full_days if WIN_START<=d<=WIN_END]
assert len(win_days)==EVAL_DAYS,f'pinned window has {len(win_days)} days, expected {EVAL_DAYS}'
log('eval window (pinned):',win_days[0],'to',win_days[-1],'days',len(win_days))
# --- per-model region-indexed params ---
def prep(mo):
    s=mo['spec']
    a=np.ones(len(REGIONS),np.float32);b=np.zeros(len(REGIONS),np.float32)
    for r,(av,bv) in s['affine'].items():a[ridx[r]]=av;b[ridx[r]]=bv
    thr=np.array([s['clf']['th_regions'][r] for r in REGIONS],np.float32)
    abst=np.zeros(len(REGIONS),bool)
    for r in s.get('abstain',[]):abst[ridx[r]]=True
    return a,b,thr,abst
for key,mo in models.items():
    mo['a'],mo['b'],mo['thr'],mo['abst']=prep(mo)
    log(key,'abstain regions:',[r for r in REGIONS if mo['abst'][ridx[r]]])
def qm_arr(qm,G,RRw):
    QM=np.zeros(len(RRw),np.float32)
    for gi in range(len(REGIONS)):
        m=G==gi;mp=qm.get(REGIONS[gi]) or qm['global'];QM[m]=np.where(RRw[m]==0,0,np.interp(RRw[m],mp['source'],mp['target']))
    return QM
def predict_model(mo,F,RR,G,XE_full):
    wet=RR>0
    Fw,RRw,Gw=F[wet],RR[wet],G[wet]
    QM=qm_arr(mo['spec']['qm'],Gw,RRw)
    if mo['extras']:
        idx=[ENAMES.index(n) for n in mo['extras']]
        XE=XE_full[:,idx]
        X=np.column_stack([Fw,RRw,QM,np.log1p(RRw),np.log1p(QM),XE]).astype(np.float32)
    else:
        X=np.column_stack([Fw,RRw,QM,np.log1p(RRw),np.log1p(QM)]).astype(np.float32)
    DM=xgb.DMatrix(X)
    p0=np.maximum(0,QM+mo['reg'].predict(DM).astype(np.float32))
    p1=np.maximum(0,mo['a'][Gw]*p0+mo['b'][Gw])
    pr=mo['clf'].predict(DM).astype(np.float32)
    pw=np.where(pr>=mo['thr'][Gw],np.maximum(p1,HEAVY),p1)
    pw=np.where(mo['abst'][Gw],RRw,pw)
    p=np.zeros(len(RR),np.float32);p[wet]=pw
    return p,p1,pr,wet
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
# --- pass 2 (identical structure to fit_worldl4.py) over 4 candidates ---
t0=time.time()
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
CANDS=['raw','v5','4x','ens']
def met_acc():return {'n':0,'ae':0.0,'se':0.0,'h':0,'hh':0,'fa':0}
def add(m,y,p):
    m['n']+=len(y);m['ae']+=float(abs(y-p).sum());m['se']+=float(((y-p)**2).sum())
    h=y>=HEAVY;f=p>=HEAVY;m['h']+=int(h.sum());m['hh']+=int((h&f).sum());m['fa']+=int((~h&f).sum())
def fin(m):
    return dict(n=m['n'],mae=round(m['ae']/max(1,m['n']),4),rmse=round(math.sqrt(m['se']/max(1,m['n'])),4),heavy_days=m['h'],hits=m['hh'],false_alarms=m['fa'])
acc={k:met_acc() for k in CANDS}
hold={k:met_acc() for k in CANDS}
per={r:{k:met_acc() for k in CANDS} for r in REGIONS}
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
        preds={}
        parts={}
        for key,mo in models.items():
            p,p1,pr,wet2=predict_model(mo,F,RR,reg_of,XE)
            preds[key]=p;parts[key]=(p1,pr,wet2)
        # ensemble per the pre-declared rule
        p1a,pra,weta=parts['v5'];p1b,prb,_=parts['4x']
        p1e=0.5*(p1a+p1b);pre=0.5*(pra+prb)
        the=0.5*(models['v5']['thr']+models['4x']['thr'])
        abste=models['v5']['abst']|models['4x']['abst']
        Gw=reg_of[weta]
        pwe=np.where(pre>=the[Gw],np.maximum(p1e,HEAVY),p1e)
        pwe=np.where(abste[Gw],RR[weta],pwe)
        pe=np.zeros(len(RR),np.float32);pe[weta]=pwe
        preds['ens']=pe
        for k in CANDS:
            p=RR if k=='raw' else preds[k]
            if k!='raw' and not np.isfinite(p).all():finite_ok=False
            add(acc[k],T,p)
            hm=~elig
            add(hold[k],T[hm],p[hm])
            for gi in range(len(REGIONS)):
                m=reg_of==gi
                add(per[REGIONS[gi]][k],T[m],p[m])
            if k!='raw':pkmax=max(pkmax,float(p.max()))
        log('day',d,'done','elapsed',int(time.time()-t0),'s')
        del F,RR,T,XE,preds,parts;gc.collect()
finall={k:fin(acc[k]) for k in CANDS}
hold_m={k:fin(hold[k]) for k in CANDS}
regm={r:{k:fin(per[r][k]) for k in CANDS} for r in REGIONS}
raw=finall['raw'];mk=finall['ens']
gates=[]
def gate(name,ok,detail):gates.append({'gate':name,'pass':bool(ok),'detail':detail})
gate('evidence: enough heavy-rain point-days in window',mk['heavy_days']>=MIN_HEAVY,f"{mk['heavy_days']} heavy point-days (need {MIN_HEAVY})")
gate('all-site MAE better than raw GFS',mk['mae']<raw['mae'],f"{mk['mae']} vs raw {raw['mae']}")
gate('all-site heavy-rain hits >= raw GFS',mk['hits']>=raw['hits'],f"{mk['hits']} vs raw {raw['hits']}")
gate('all-site false alarms <= 102% of raw GFS',mk['false_alarms']<=1.02*raw['false_alarms'],f"{mk['false_alarms']} vs raw {raw['false_alarms']}")
gate('unseen-site MAE better than raw GFS',hold_m['ens']['mae']<hold_m['raw']['mae'],f"{hold_m['ens']['mae']} vs raw {hold_m['raw']['mae']}")
gate('unseen-site heavy hits >= 97% of raw GFS',hold_m['ens']['hits']>=0.97*hold_m['raw']['hits'],f"{hold_m['ens']['hits']} vs raw {hold_m['raw']['hits']}")
gate('unseen-site false alarms <= 102% of raw GFS',hold_m['ens']['false_alarms']<=1.02*hold_m['raw']['false_alarms'],f"{hold_m['ens']['false_alarms']} vs raw {hold_m['raw']['false_alarms']}")
for rn in REGIONS:
    rm_=regm[rn]['raw']['mae'];cm_=regm[rn]['ens']['mae']
    ok=(cm_<=1.03*rm_) if rm_>=0.05 else (cm_<=rm_+0.02)
    gate(f'per-region MAE within 3% of raw GFS: {rn}',bool(ok),f'ensemble {cm_} vs raw {rm_}')
gate('predictions finite and sane',bool(finite_ok and pkmax<2000),f"max {pkmax:.1f} mm")
out={'lead':LEAD,'kind':'ensemble-eval-B (eval only, not deployed)',
 'parents':{k:{'release':RELS[k],'version':models[k]['spec'].get('version'),'abstain':[r for r in REGIONS if models[k]['abst'][ridx[r]]]} for k in models},
 'combination_rule':'PRE-DECLARED before window numbers were seen in this run: amount=0.5*(p1_v5+p1_4x); heavy prob=0.5*(p_v5+p_4x); per-region threshold=0.5*(th_v5+th_4x); heavy bump where(p>=th,max(amount,64.5),amount); abstain=UNION of both parents (raw GFS served there). Nothing tuned on the window.',
 'evaluation':{'window':f"{win_days[0]} to {win_days[-1]} (pinned to world-v4; same 2 LAGGED exclusions as all prior evals)",'candidates':finall,'unseen_sites':hold_m,'per_region':regm},
 'gates':gates,'all_gates_pass':all(g['pass'] for g in gates),
 'note':'Ensemble of two existing fitted models. Feature sets used per model exactly as fitted (lead-2: both have 3 spatial + rr_l3 prior; lead-3: v5 is the base-feature architecture, 4x has 3 spatial features - each model scored with its own fitted layout). v5 fit-day-stride 16, 4x trial stride 4. Scored through the identical pass-2 gate harness as fit_worldl4.py on the identical pinned window. v5/4x columns recomputed in this same run for exact comparability. Research artifact only; no deploy without owner approval.'}
json.dump(out,open(f'{W}/metrics.json','w'),indent=1)
log(json.dumps(out['evaluation']['candidates'],indent=1))
log('UNSEEN',json.dumps(hold_m,indent=1))
log('GATES',json.dumps(gates,indent=1)[:2500])
log('DONE all_gates_pass',out['all_gates_pass'])
if os.environ.get('VD_UPLOAD','1')=='1':
    OUT=f'world-fit-l{LEAD}-ens'
    r=subprocess.run(['gh','release','view',OUT,'--repo',REPO],capture_output=True)
    if r.returncode:subprocess.run(['gh','release','create',OUT,'--repo',REPO,'--title',f'World lead-{LEAD} ensemble eval B (research only)','--notes',f'Ensemble eval B: v5 + 4x-trial corrections averaged, pinned window, same gate harness. Eval only, not deployed.'],check=True)
    subprocess.run(['cp',f'{W}/metrics.json',f'{W}/world-fit-l{LEAD}-ens-metrics.json'],check=True)
    subprocess.run(['gh','release','upload',OUT,f'{W}/world-fit-l{LEAD}-ens-metrics.json','--clobber','--repo',REPO],check=True)
    log('uploaded to',OUT)
