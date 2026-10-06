"""VayuDrishti nightly learning, lead 1 (24h). Fetches new ERA5/GFS days, retrains, evaluates against the champion,
promotes only if every gate passes AND promotion is switched on. Never auto-deploys a regression.
Usage: python3 run.py [--promote 0|1] [--local] [--rollback prev|embedded]"""
import os,sys,json,re,glob,math,time,gc,shutil,subprocess,tarfile,datetime as dt,urllib.request,traceback
import numpy as np
A=dict(a.split('=',1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
PROMOTE=A.get('--promote','0')=='1';LOCAL='--local' in sys.argv
REPO=os.environ.get('GITHUB_REPOSITORY','vayudrishti-arch/vayudrishti-arch.github.io')
W=os.environ.get('VD_WORK','/tmp/vdn');os.makedirs(W,exist_ok=True)
ROOT=os.getcwd()  # repo checkout root
STATUS=os.path.join(ROOT,'nightly','status.json')
IST=dt.timezone(dt.timedelta(hours=5,minutes=30))
NOW=dt.datetime.now(IST)
LAG=6  # ERA5 lag in days
EVAL_DAYS=60;MIN_HEAVY=300
regs=["central","east","islands","north","northeast","south","west"]
def log(*a):print(*a,flush=True)
def load_status():
    try:return json.load(open(STATUS))
    except Exception:return {'runs':[]}
def save_status(st,run):
    st['runs']=([run]+st.get('runs',[]))[:60];st['latest']=run;st['promotion_enabled']=PROMOTE
    st['note']='VayuDrishti nightly learning. Research product, not an official warning service. ERA5 reference data arrives about 6 days late, so the model learns from days about 6 days old.'
    os.makedirs(os.path.dirname(STATUS),exist_ok=True);json.dump(st,open(STATUS,'w'),indent=1)
def http(url,tries=3):
    for t in range(tries):
        try:return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'vd-nightly'}),timeout=120)
        except Exception:
            if t==tries-1:raise
            time.sleep(3*(t+1))
def assets(tag):
    try:html=http(f'https://github.com/{REPO}/releases/expanded_assets/{tag}').read().decode()
    except Exception:return {}
    return {os.path.basename(u):f'https://github.com/{u}' for u in re.findall(r'(?<=href="/)([^"]*releases/download/[^"]+)',html)}
def fetch_data():
    """Download seed + day bundles + pipeline files from releases and unpack."""
    d=os.path.join(W,'dl');os.makedirs(d,exist_ok=True)
    for tag in ['data-lead1-seed']:
        for name,url in assets(tag).items():
            p=os.path.join(d,name)
            if not os.path.exists(p):
                with open(p+'.part','wb') as f:shutil.copyfileobj(http(url),f)
                os.replace(p+'.part',p)
    groups={}
    for f in os.listdir(d):
        if f.endswith('.part'):continue
        base=re.sub(r'^\d+-','',f);m=re.match(r'(.*?)(\.tar)?[.-]?(p-?\d+|r\d+)$',base)
        groups.setdefault(m.group(1) if m else base,[]).append((base,f))
    for g,items in groups.items():
        if g.startswith('vd-') :continue
        items.sort(key=lambda x:(0 if re.search(r'p-?\d+$',x[0]) else 1,x[0]))
        out=os.path.join(W,'tars',g+'.tar');os.makedirs(os.path.dirname(out),exist_ok=True)
        if os.path.exists(out+'.done'):continue
        with open(out,'wb') as o:
            for _,f in items:o.write(open(os.path.join(d,f),'rb').read())
        try:
            tarfile.open(out).extractall(os.path.join(W,'seed' if 'seed' in g else 'days'));open(out+'.done','w').write('ok')
        except Exception as e:log('WARN tar',g,repr(e)[:120])
    # nightly day files
    nd=os.path.join(W,'days');os.makedirs(nd,exist_ok=True)
    for name,url in assets('data-nightly').items():
        if name.endswith('.npz') and not os.path.exists(os.path.join(nd,name)):
            with open(os.path.join(nd,name),'wb') as f:shutil.copyfileobj(http(url),f)
    p=os.path.join(W,'pipe');os.makedirs(p,exist_ok=True)
    if not os.path.exists(os.path.join(p,'points.json')):
        pa=assets('data-lead1-seed');name=[n for n in pa if n.endswith('vd-nightly-pipeline.tar.gz')][0]
        with open(os.path.join(W,'pipe.tgz'),'wb') as f:shutil.copyfileobj(http(pa[name]),f)
        tarfile.open(os.path.join(W,'pipe.tgz')).extractall(p)
def locate():
    if LOCAL:return '/tmp/vd/days','/tmp/seed','/tmp/vd/training'
    seed=os.path.join(W,'seed');cand=[os.path.dirname(x) for x in glob.glob(seed+'/**/lead1-central.npz',recursive=True)]
    return os.path.join(W,'days'),(cand[0] if cand else seed),os.path.join(W,'pipe')
def day_files(days_dir):
    out={}
    for f in glob.glob(days_dir+'/**/*.npz',recursive=True):
        b=os.path.basename(f)
        if re.match(r'\d{4}-\d{2}-\d{2}\.npz$',b):out[b[:10]]=f
    return out
def fetch_new(days_dir,pipe_dir,have):
    last=max(dt.date.fromisoformat(k) for k in have);target=NOW.date()-dt.timedelta(days=LAG)
    todo=[];d=last+dt.timedelta(1)
    while d<=target and len(todo)<14:todo.append(str(d));d+=dt.timedelta(1)
    got=[];errs=[]
    if not todo:return got,errs,str(last)
    sys.path.insert(0,pipe_dir);import vdlib as v,concurrent.futures as cf
    pool=cf.ThreadPoolExecutor(48)
    for date in todo:
        try:
            a,nb=v.fetch_gfs_day(date,pool);f,raw=v.assemble(date,a);tg=v.fetch_era_day(date)
            if not np.isfinite(tg).all():raise ValueError('non-finite ERA5')
        except Exception as e:
            errs.append((date,repr(e)[:160]));break   # stop at first gap so the corpus stays contiguous
        p=os.path.join(days_dir,f'{date}.npz');np.savez_compressed(p+'.tmp.npz',F=f,RR=raw,T=tg.astype(np.float32));os.replace(p+'.tmp.npz',p);got.append(date)
    return got,errs,str(last)
def upload_days(got,days_dir):
    if not got or LOCAL or not shutil.which('gh'):return
    r=subprocess.run(['gh','release','view','data-nightly','--repo',REPO],capture_output=True)
    if r.returncode:subprocess.run(['gh','release','create','data-nightly','--repo',REPO,'--title','Nightly day files','--notes','Daily ERA5/GFS day files fetched by the nightly job.'],check=True)
    subprocess.run(['gh','release','upload','data-nightly','--repo',REPO,'--clobber']+[os.path.join(days_dir,f'{d}.npz') for d in got],check=True)
# ---- model helpers
def tree_predict(trees,X,base):
    y=np.full(len(X),np.float32(base),np.float32)
    for t in trees:
        lc=np.array(t['left_children']);rc=np.array(t['right_children']);si=np.array(t['split_indices']);sc=np.array(t['split_conditions'],np.float32);dl=np.array(t['default_left'],bool)
        n=np.zeros(len(X),np.int64)
        for _ in range(12):
            l=lc[n];act=l!=-1
            if not act.any():break
            v=X[np.arange(len(X)),np.where(act,si[n],0)];left=np.where(np.isnan(v),dl[n],v<sc[n])
            n=np.where(act,np.where(left,l,rc[n]),n)
        y=(y+sc[n]).astype(np.float32)
    return y
def build_X(F,RR,G,qm):
    QM=np.zeros(len(RR),np.float32)
    for gi,r in enumerate(regs):
        m=G==gi;mp=qm.get(r) or qm['global'];QM[m]=np.where(RR[m]==0,0,np.interp(RR[m],mp['source'],mp['target']))
    X=np.column_stack([F,RR,QM,np.log1p(RR),np.log1p(QM)]).astype(np.float32);return X,QM
def predict_model(M,F,RR,G,TH=None):
    X,QM=build_X(F,RR,G,M['qm']);p0=np.where(RR==0,0,np.maximum(0,QM+tree_predict(M['trees'],X,M['base'])))
    pr=1/(1+np.exp(-tree_predict(M['clf']['trees'],X,M['clf']['base'])));pr=np.where(RR==0,0,pr)
    return np.where(pr>=M['clf']['threshold'],np.maximum(p0,64.5),p0)
def met(y,p):
    h=y>=64.5;f=p>=64.5
    return dict(n=int(len(y)),mae=round(float(np.mean(abs(y-p))),4),rmse=round(float(np.sqrt(np.mean((y-p)**2))),4),heavy_days=int(h.sum()),hits=int((h&f).sum()),false_alarms=int((~h&f).sum()))
def champion():
    for p in [os.path.join(ROOT,'app','lead1-model.json')]:
        if os.path.exists(p):return json.load(open(p)),'app/lead1-model.json'
    s=open(os.path.join(ROOT,'app','index.html')).read();i=s.index('const ENSEMBLE=')+len('const ENSEMBLE=')
    E,_=json.JSONDecoder().raw_decode(s[i:]);return [m for m in E['models'] if m['lead']==1][0],'embedded in app/index.html ('+E['version']+')'
def main():
    run={'started_ist':NOW.strftime('%Y-%m-%d %H:%M'),'promotion_enabled':PROMOTE,'result':'running'}
    st=load_status()
    try:
        if '--rollback' in A or A.get('--rollback'):
            mode=A['--rollback'];pm=os.path.join(ROOT,'app','lead1-model.json');pv=os.path.join(ROOT,'app','lead1-model.prev.json')
            if mode=='prev' and os.path.exists(pv):shutil.copy(pv,pm)
            elif os.path.exists(pm):os.remove(pm)
            run.update(result='rolled_back',detail='rollback to '+mode);save_status(st,run);return
        from xgboost import XGBRegressor,XGBClassifier
        if not LOCAL:fetch_data()
        days_dir,seed_dir,pipe_dir=locate()
        have=day_files(days_dir);log('day files',len(have))
        got,errs,last=fetch_new(days_dir,pipe_dir,have) if not LOCAL else ([],[],max(have))
        if got:upload_days(got,days_dir);have=day_files(days_dir)
        run['data']={'day_files':len(have),'new_days':got,'fetch_stopped':errs,'last_corpus_day_before':last}
        pts=json.load(open(os.path.join(pipe_dir,'points.json')));reg_of=np.array([regs.index(p['region']) for p in pts]);elig=np.array([p['fit_eligible'] for p in pts])
        seed_start=dt.date(2024,1,1);seed_days=998;fit_start=dt.date.fromisoformat(os.environ.get('VD_FIT_START','2021-03-25'))
        dates={dt.date.fromisoformat(k):('f',v) for k,v in have.items()}
        for i in range(seed_days):
            d=seed_start+dt.timedelta(i)
            if d not in dates:dates[d]=('s',i)
        alld=sorted(d for d in dates if d>=fit_start);L=alld[-1];run['data']['corpus_first']=str(alld[0]);run['data']['corpus_last']=str(L);run['data']['corpus_days']=len(alld)
        # contiguity check (two known-corrupt days are allowed)
        missing=[str(alld[i]+dt.timedelta(1)) for i in range(len(alld)-1) if (alld[i+1]-alld[i]).days>1 and str(alld[i]+dt.timedelta(1)) not in('2022-11-30',)]
        run['data']['gaps']=missing[:10]
        N=len(alld)*4000;F=np.empty((N,20),np.float32);RR=np.empty(N,np.float32);T=np.empty(N,np.float32);D=np.empty(N,np.int16)
        for j,d in enumerate(alld):
            kind,v=dates[d];sl=slice(j*4000,(j+1)*4000);D[sl]=(d-seed_start).days
            if kind=='f':z=np.load(v);F[sl]=z['F'];RR[sl]=z['RR'];T[sl]=z['T']
        off=0;npt={r:int((reg_of==i).sum()) for i,r in enumerate(regs)}
        for r in regs:
            n=npt[r];z=np.load(os.path.join(seed_dir,f'lead1-{r}.npz'));Fz=z['F'].reshape(seed_days,n,20);Rz=z['RR'].reshape(seed_days,n);Tz=z['T'].reshape(seed_days,n)
            for j,d in enumerate(alld):
                kind,v=dates[d]
                if kind=='s':sl=slice(j*4000+off,j*4000+off+n);F[sl]=Fz[v];RR[sl]=Rz[v];T[sl]=Tz[v]
            off+=n;del Fz,Rz,Tz,z;gc.collect()
        G=np.tile(reg_of,len(alld)).astype(np.int8);E=np.tile(elig,len(alld))
        ev_start=(L-dt.timedelta(EVAL_DAYS-1)-seed_start).days;Lday=(L-seed_start).days
        test=D>=ev_start;hold=test&(~E)
        champ,csrc=champion();run['champion_source']=csrc
        pc=predict_model(champ,F[test],RR[test],G[test]);yt=T[test]
        # gates need candidate fitted before the window
        def fit_model(fit_mask,TH):
            wet=fit_mask&(RR>0);q=np.linspace(0,1,101);gl=(np.quantile(RR[wet],q),np.quantile(T[wet],q));qm={}
            for gi,r in enumerate(regs):
                w=fit_mask&(G==gi)&(RR>0);s_,t_=(np.quantile(RR[w],q),np.quantile(T[w],q)) if w.sum()>=100 else gl;qm[r]={'source':[float(x) for x in s_],'target':[float(x) for x in t_]}
            qm['global']={'source':[float(x) for x in gl[0]],'target':[float(x) for x in gl[1]]}
            X,QM=build_X(F[wet],RR[wet],G[wet],qm)
            reg=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=.04,reg_lambda=50,min_child_weight=20,subsample=.85,colsample_bytree=.9,n_jobs=2,random_state=42,tree_method="hist").fit(X,(T[wet]-QM))
            yh=(T[wet]>=64.5).astype(int)
            clf=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,min_child_weight=5,subsample=.85,colsample_bytree=.9,scale_pos_weight=min(50,(len(yh)-yh.sum())/max(1,yh.sum()))**.5,n_jobs=2,random_state=42,tree_method="hist",eval_metric="logloss").fit(X,yh)
            def tr(mod):
                cfg=json.loads(mod.get_booster().save_config());bs=float(str(cfg['learner']['learner_model_param']['base_score']).strip('[]'))
                j=json.loads(mod.get_booster().save_raw('json'));return bs,[{k:t[k] for k in ['left_children','right_children','split_indices','split_conditions','default_left']} for t in j['learner']['gradient_booster']['model']['trees']]
            bs,rt=tr(reg);cb,ct=tr(clf)
            return {'lead':1,'weight':1.0,'qm':qm,'base':bs,'trees':rt,'clf':{'base':math.log(cb/(1-cb)),'trees':ct,'threshold':TH}}
        TH=champ['clf']['threshold']
        cand=fit_model((D<ev_start)&E,TH);pk=predict_model(cand,F[test],RR[test],G[test])
        raw=met(yt,RR[test]);mc=met(yt,pc);mk=met(yt,pk);Gt=G[test];Ht=hold[test]
        hold_m={k:met(yt[Ht],x[Ht]) for k,x in [('raw',RR[test]),('champion',pc),('candidate',pk)]}
        regm={r:{'champion':round(float(np.mean(abs(yt[Gt==i]-pc[Gt==i]))),4),'candidate':round(float(np.mean(abs(yt[Gt==i]-pk[Gt==i]))),4)} for i,r in enumerate(regs)}
        gates=[]
        def gate(name,ok,detail):gates.append({'gate':name,'pass':bool(ok),'detail':detail})
        gate('evidence: enough heavy-rain point-days in window',mk['heavy_days']>=MIN_HEAVY,f"{mk['heavy_days']} heavy point-days (need {MIN_HEAVY})")
        gate('all-site MAE not worse than champion',mk['mae']<=mc['mae']*1.005,f"{mk['mae']} vs {mc['mae']}")
        gate('all-site heavy-rain hits >= 98% of champion',mk['hits']>=0.98*mc['hits'],f"{mk['hits']} vs {mc['hits']}")
        gate('all-site false alarms <= 102% of champion',mk['false_alarms']<=1.02*mc['false_alarms'],f"{mk['false_alarms']} vs {mc['false_alarms']}")
        gate('unseen-site MAE not worse than champion',hold_m['candidate']['mae']<=hold_m['champion']['mae']*1.005,f"{hold_m['candidate']['mae']} vs {hold_m['champion']['mae']}")
        gate('unseen-site heavy hits >= 97% of champion',hold_m['candidate']['hits']>=0.97*hold_m['champion']['hits'],f"{hold_m['candidate']['hits']} vs {hold_m['champion']['hits']}")
        gate('unseen-site false alarms <= 102% of champion',hold_m['candidate']['false_alarms']<=1.02*hold_m['champion']['false_alarms'],f"{hold_m['candidate']['false_alarms']} vs {hold_m['champion']['false_alarms']}")
        gate('every region MAE within 3% of champion (islands 10%)',all(v['candidate']<=v['champion']*(1.10 if r=='islands' else 1.03) for r,v in regm.items()),json.dumps(regm))
        gate('beats raw GFS on MAE and heavy hits',mk['mae']<raw['mae'] and mk['hits']>=raw['hits'],f"MAE {mk['mae']} vs raw {raw['mae']}, hits {mk['hits']} vs {raw['hits']}")
        gate('predictions finite and sane',bool(np.isfinite(pk).all() and pk.max()<2000),f"max {float(pk.max()):.1f} mm")
        run['evaluation']={'window':f"{L-dt.timedelta(EVAL_DAYS-1)} to {L}",'raw':raw,'champion':mc,'candidate':mk,'unseen_sites':hold_m,'region_mae':regm,'candidate_fit_through':str(L-dt.timedelta(EVAL_DAYS))}
        run['gates']=gates;allpass=all(g['pass'] for g in gates) or (LOCAL and os.environ.get('VD_TEST_FORCE')=='1');run['all_gates_pass']=allpass
        if not allpass:
            run['result']='kept_current_model';run['why']='; '.join(g['gate']+' ('+g['detail'][:80]+')' for g in gates if not g['pass'])[:600]
        else:
            final=fit_model((D<=Lday)&E,TH);final['fit_through']=str(L);final['fit_start']=str(fit_start);final['version']='v4-nightly-'+str(L)
            sl=np.where(test&(RR>0))[0][::50][:400]
            par=[{'rich':[float(x) for x in F[sl][i]],'raw':float(RR[sl][i]),'region':regs[G[sl][i]],'py':float(predict_model(final,F[sl][i:i+1],RR[sl][i:i+1],G[sl][i:i+1])[0])} for i in range(len(sl))]
            os.makedirs(os.path.join(W,'out'),exist_ok=True);json.dump(final,open(os.path.join(W,'out','lead1-model.json'),'w'),separators=(',',':'));json.dump(par,open(os.path.join(W,'out','parity.json'),'w'))
            pj=subprocess.run(['node',os.path.join(ROOT,'nightly','parity.js'),os.path.join(W,'out','lead1-model.json'),os.path.join(W,'out','parity.json')],capture_output=True,text=True)
            ok=pj.returncode==0 and float(pj.stdout.strip().split()[-1])<1e-3 if pj.returncode==0 and pj.stdout.strip() else False
            gate('JS output matches Python (parity)',ok,(pj.stdout or pj.stderr).strip()[:120]);run['all_gates_pass']=allpass=all(g['pass'] for g in gates)
            if not allpass:run['result']='kept_current_model';run['why']='JS parity failed'
            elif PROMOTE:
                pm=os.path.join(ROOT,'app','lead1-model.json');pv=os.path.join(ROOT,'app','lead1-model.prev.json')
                if os.path.exists(pm):shutil.copy(pm,pv)
                shutil.copy(os.path.join(W,'out','lead1-model.json'),pm);run['result']='promoted';run['why']='all gates passed'
            else:
                shutil.copy(os.path.join(W,'out','lead1-model.json'),os.path.join(ROOT,'nightly','candidate-lead1.json'));run['result']='dry_run_would_promote';run['why']='all gates passed; promotion switched off, current model kept'
        run['finished_ist']=dt.datetime.now(IST).strftime('%Y-%m-%d %H:%M')
        save_status(st,run);log(json.dumps({k:run[k] for k in run if k not in('gates',)},indent=1)[:3000])
    except Exception as e:
        run.update(result='error',error=repr(e)[:300],trace=traceback.format_exc()[-800:],finished_ist=dt.datetime.now(IST).strftime('%Y-%m-%d %H:%M'));save_status(st,run);log(run['trace']);sys.exit(1)
main()
