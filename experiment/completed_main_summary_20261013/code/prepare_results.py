"""Read-only source audit and compact reporting of actually completed outputs."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, sys
import numpy as np
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();R=Path(a.root)
OLD=R/'experiment/generalization_validation_20261012';EXT=R/'experiment/generalization_extended_20261013';F=R/'experiment/completed_main_summary_20261013'
for name in ['code','results','docs','docs/report_qa']:(F/name).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(OLD/'code'))
from core import read,save,sha,utc,truth,baseline,metric,export,parity
checks=[];sources={}
def check(label,ok):
    checks.append(dict(check=label,passed=bool(ok)))
    assert ok,label
def bind(path):sources[str(path.relative_to(R))]=sha(path);return read(path)
def same_metrics(label,new,old):
    for k,v in new.items():
        if k not in old:continue
        w=old[k];ok=v==w if isinstance(v,(str,int,type(None))) else np.isclose(v,w,atol=1e-9,rtol=1e-11)
        check(label+' '+k,ok)
def bands(y,m,lo,hi):
    check('Finite ordered interval',np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all())
    return dict(interval80_nominal_percent=80,interval80_coverage_percent=float(100*((y>=lo)&(y<=hi))[m].mean()),
        interval80_mean_width=float((hi-lo)[m].mean()),interval80_winkler80=float((hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0))[m].mean()),
        interval80_below_percent=float(100*(y<lo)[m].mean()),interval80_above_percent=float(100*(y>hi)[m].mean()))
check('Prior score verification passed',bind(OLD/'results/final_verification.json')['all_passed'])
check('Prior input audit passed',bind(OLD/'results/independent_input_cache_audit.json')['all_passed'])
audit=bind(OLD/'docs/scientific_review/methodological_audit.json')
for rel,digest in audit['reference_source_sha256'].items():check('Prior source unchanged '+rel,sha(R/rel)==digest)
rec=bind(R/'当前推荐方案.json');check('R48 deployment unchanged',rec['scheme_id']=='R48_ROBUST_HDSF_USER_ADOPTED')
rows=[]
oldproto=bind(OLD/'docs/protocol.json')
with np.load(OLD/'data/nlr_15min.npz') as z:old_nlr_x=z['x'].copy()
sources[str((OLD/'data/nlr_15min.npz').relative_to(R))]=sha(OLD/'data/nlr_15min.npz')
for phase in ['test','confirm']:
    summary=bind(OLD/f'results/nlr_{phase}_summary.json');path=OLD/f'results/nlr_{phase}_predictions.npz';sources[str(path.relative_to(R))]=sha(path)
    check('NLR prediction SHA '+phase,sha(path)==summary['prediction_sha256'])
    with np.load(path) as z:
        for row in summary['rows']:
            keep=['HDSF_Google_frozen','R48_Google_frozen','HDSF_NLR_refit','expert_timesfm30','expert_timesfm30_long','expert_toto2_joint','last','mean3']
            if row['model'] not in keep:continue
            if row['model'] in ['last','mean3'] and row['model']!=('last' if row['horizon_days']==1 else'mean3'):continue
            h=row['horizon_days'];j=0 if row['target']=='facility_total_kw' else 1;name=row['target']+'_'+row['model'];yy=z[f'truth_H{h}'][j:j+1];mm=z[f'valid_H{h}'][j:j+1];p=z[f'{name}_H{h}'];b=baseline(old_nlr_x[j:j+1],oldproto['nlr_origins'][phase],h,'last' if h==1 else'mean3')
            new=metric(yy,p,mm,96,b);same_metrics('NLR '+phase+' '+name+' '+str(h),new,row)
            if f'{name}_lower80_H{h}' in z.files:
                iv=bands(yy,mm,z[f'{name}_lower80_H{h}'],z[f'{name}_upper80_H{h}']);same_metrics('NLR interval '+name,iv,row)
            else:check('Uncalibrated expert intervals not fabricated '+name,row.get('interval80_coverage_percent') is None)
            rows.append(dict(**row,evidence='completed_temporal_test',model_full_name={
                'HDSF_Google_frozen':'HDSF半日单纯形 原Google权重固定迁移',
                'R48_Google_frozen':'第48轮HDSF稳健校正 原Google权重固定迁移',
                'HDSF_NLR_refit':'NLR目标数据全局及半日单纯形权重适配',
                'expert_timesfm30_long':'TimesFM 3 原预训练权重 21天历史',
                'expert_timesfm30':'TimesFM 3 原预训练权重 14天历史',
                'expert_toto2_joint':'Toto2 原预训练权重 14天历史',
                'last':'前一天同刻','mean3':'前三天同刻均值'}[row['model']]))
for fn in ['sea_summary.json','cluster_summary.json']:
    summary=bind(OLD/'results'/fn);path=OLD/'results'/fn.replace('_summary.json','_predictions.npz');sources[str(path.relative_to(R))]=sha(path)
    with np.load(path) as z:
        for row in summary['rows']:
            h=row['horizon_days'];model=row['model']
            if fn.startswith('sea'):
                if model not in ['last','Ridge_daily','daily_simplex']:continue
                yy=z[f'truth_H{h}'];mm=z[f'valid_H{h}'];p=z[model+f'_H{h}'];b=z[('last' if h==1 else'mean3')+f'_H{h}'];new=metric(yy,p,mm,1,b)
                iv=bands(yy,mm,z[model+f'_lower80_H{h}'],z[model+f'_upper80_H{h}']);same_metrics('Sea interval '+model,iv,row)
                full={'last':'前一日电量','Ridge_daily':'日尺度Ridge回归','daily_simplex':'日尺度单纯形开发期冻结 回退前一日'}[model]
            else:
                with np.load(R/'experiment/results/data.npz') as d:yy,mm=truth(d['y'],d['valid'],list(range(22,31)) if h==1 else[22,23,24],h)
                key={'R48_fixed':'R48_fixed','R48_CPU_DEV_frozen':'R48_CPU_DEV_frozen','LightGBM_power_only':'power_only','LightGBM_power_CPU_memory':'CPU_memory'}[model]
                p=z[key+f'_H{h}'];scope=row['scope'];check('CPU scope '+scope,scope in ['cella5','all57','a5','covered5'])
                if p.shape[0]==5:yy,mm=yy[:5],mm[:5]
                elif scope!='all57':yy,mm,p=yy[:5],mm[:5],p[:5]
                new=metric(yy,p,mm,96)
                full={'R48_fixed':'第48轮HDSF稳健校正 固定推荐','R48_CPU_DEV_frozen':'第48轮HDSF加历史CPU内存差分 开发期冻结25%','LightGBM_power_only':'LightGBM 仅历史功率','LightGBM_power_CPU_memory':'LightGBM 历史功率加CPU内存'}[model]
            same_metrics(fn+' '+model+' '+str(h),new,row)
            rows.append(dict(**row,evidence='completed_daily_test' if fn.startswith('sea') else'completed_same_source_ablation',model_full_name=full))
# Google original task comparison from stored forecasts on identical masks.
with np.load(R/'experiment/results/data.npz') as d:data={k:d[k].copy() for k in d.files}
sources['experiment/results/data.npz']=sha(R/'experiment/results/data.npz')
s44=bind(R/'experiment/round44/results/summary.json');s48=bind(R/'experiment/round48/results/summary.json')
for rd in [44,48]:sources[f'experiment/round{rd}/results/predictions.npz']=sha(R/f'experiment/round{rd}/results/predictions.npz')
with np.load(R/'experiment/round44/results/predictions.npz') as z44,np.load(R/'experiment/round48/results/predictions.npz') as z48:
    for h,oo in [(1,list(range(22,31))),(7,[22,23,24])]:
        yy,mm=truth(data['y'],data['valid'],oo,h);b=baseline(data['y'],oo,h,'last' if h==1 else'mean3')
        members=[('baseline',b,'前一天同刻' if h==1 else'前三天同刻均值'),('hdsf',z44[f'hdsf_H{h}'],'HDSF半日单纯形 原版'),('R48',z48[f'DEV_frozen_R48_candidate_H{h}'],'第48轮HDSF稳健校正 当前推荐'),('timesfm30_long',z44[f'timesfm30_long_H{h}'],'TimesFM 3 原预训练权重 21天历史'),('toto2_joint',z44[f'toto2_joint_H{h}'],'Toto2 原预训练权重 14天历史')]
        for model,p,full in members:
            new=metric(yy,p,mm,96,b)
            if model in s44['scores'][str(h)]:same_metrics('Google44 '+model,new,s44['scores'][str(h)][model])
            if model=='R48':
                old=s48['scores'][str(h)]['DEV_frozen_R48_candidate'];same_metrics('Google48',new,old['metrics']);iv=bands(yy,mm,z48[f'DEV_frozen_R48_candidate_lower80_H{h}'],z48[f'DEV_frozen_R48_candidate_upper80_H{h}']);new.update(iv)
                check('R48 interval matches summary',np.isclose(iv['interval80_coverage_percent'],old['interval']['coverage_percent'],atol=1e-10))
            rows.append(dict(dataset='GooglePower2019',target='57PDU',phase='test',horizon_days=h,model=model,unit='normalized power',evidence='completed_exposed_test',model_full_name=full,**new))
parity(R,F/'results/old_evaluator_parity.json')
# Newly trained models: development results only; no test or deployment selection.
dev=[];proto=bind(EXT/'docs/protocol.json');proto_sha=sha(EXT/'docs/protocol.json');model_audits=[]
for case in ['NLR_2021','NLR_2024_2025','Google2019']:
    cfg=proto['cases'][case];cut=cfg['cutoff'];dp=EXT/'data'/f'{case}.npz';sources[str(dp.relative_to(R))]=sha(dp)
    with np.load(dp) as z:y,x,m,sc,dates,names=[z[k].copy() for k in ['y','x','valid','scales','dates','names']]
    flat=(x/sc[:,None,None]).reshape(len(x),-1)
    for family in ['timesfm3','toto2','moirai2']:
        for label in ['s16','s48']:
            jp=EXT/'models'/f'{case}_{family}_cut{cut}_{label}.json';info=bind(jp);weight=jp.with_suffix('.pt')
            check('Real updates '+jp.name,info['real_optimizer_steps']==int(label[1:]) and len(info['training_windows'])==int(label[1:]))
            check('Weight SHA '+jp.name,sha(weight)==info['weight_file_sha256']);sources[str(weight.relative_to(R))]=info['weight_file_sha256']
            check('Training bounds '+jp.name,info['training_last_label_exclusive']<=cut*96 and all(0<=w['input_start']<w['input_end_exclusive']<w['label_end_exclusive']<=cut*96 for w in info['training_windows']))
            check('Model data/protocol '+jp.name,info['data_sha256']==sha(dp) and info['protocol_sha256']==proto_sha)
            model_audits.append(dict(case=case,family=family,label=label,real_optimizer_steps=info['real_optimizer_steps'],skipped_invalid_gradients=len(info.get('skipped_invalid_gradients',[])),training_last_label_exclusive=info['training_last_label_exclusive'],cutoff_exclusive=cut*96,weight_sha256=info['weight_file_sha256']))
        for h in [1,7]:
            oo=cfg['origins']['dev'] if case.startswith('NLR') else list(range(14,22)) if h==1 else[14,15]
            check('DEV labels purged '+case,all(str(dates[o+h-1])[:4]==str(dates[o])[:4] for o in oo))
            yy,mm=truth(y,m,oo,h);b=baseline(x,oo,h,'last' if h==1 else'mean3')
            for label in ['s0','s16','s48']:
                parts=[]
                for o in oo:
                    cp=EXT/'cache'/f'{case}_{family}_cut{cut}_{label}_dev_H{h}_o{o}.npz';sources[str(cp.relative_to(R))]=sha(cp)
                    with np.load(cp) as z:
                        check('Cache bounds '+cp.name,int(z['origin'])==o and int(z['horizon'])==h and str(z['protocol_sha256'])==proto_sha)
                        days=21 if family=='timesfm3' else 14 if family=='toto2' else 7 if h==1 else 14
                        ctx=flat[:,max(0,o-days)*96:o*96].copy()
                        if family=='toto2':
                            n=max(32,math.ceil(ctx.shape[-1]/32)*32)
                            if ctx.shape[-1]<n:ctx=np.concatenate([np.repeat(ctx[:,:1],n-ctx.shape[-1],axis=1),ctx],1)
                        digest=hashlib.sha256(np.ascontiguousarray(ctx,dtype=np.float32).tobytes()).hexdigest();check('Past-only input '+cp.name,str(z['input_digest'])==digest)
                        if label!='s0':info=read(EXT/'models'/f'{case}_{family}_cut{cut}_{label}.json');check('Cache weight '+cp.name,str(z['weight_digest'])==info['trained_trainable_sha256'])
                        parts.append(z['p'].copy()*sc[:,None])
                p=np.stack(parts,1)
                for j in range(len(y)) if case.startswith('NLR') else[None]:
                    new=metric(yy if j is None else yy[j:j+1],p if j is None else p[j:j+1],mm if j is None else mm[j:j+1],96,b if j is None else b[j:j+1])
                    dev.append(dict(dataset=case,target='57PDU' if j is None else str(names[j]),phase='dev',horizon_days=h,model=family,checkpoint=label,unit='normalized power' if case=='Google2019' else'kW',model_full_name={'timesfm3':'TimesFM 3','toto2':'Toto2','moirai2':'Moirai2'}[family]+(' 原预训练权重' if label=='s0' else ' 目标数据末层及输出头微调 '+label[1:]+'步'),evidence='development_only_not_tested',origins=len(oo),**new))
export(F/'results/已完成主要方法统一全指标.csv',rows);export(F/'results/新增训练开发期全指标.csv',dev)
save(F/'results/summary.json',dict(created_utc=utc(),completed_rows=rows,development_rows=dev,methodological_audit=audit,training_audits=model_audits,current_recommendation=rec,all_data_scientific_scope='Exploratory; one external NLR facility, Sea daily energy, Cluster same-source covariates',new_target_fit_test_scored=False,scope_reduced_by_user=True))
save(F/'results/source_verification.json',dict(all_passed=True,utc=utc(),checks=len(checks),check_details=checks,source_sha256=sources,rows=len(rows),development_rows=len(dev),recommendation_sha256=sha(R/'当前推荐方案.json'),new_test_scoring_started=False))
save(F/'docs/scope.json',dict(created_utc=utc(),user_request='汇总已经完成的比较好的主要方法 不等待次要实验',remaining_old_search_cancelled=True,original_protocol_unchanged=True,raw_negative_results_preserved=True,reported_families=['HDSF original','R48 robust HDSF','NLR simplex refit','TimesFM3','Toto2','Moirai2 development only'],data_sources='Existing Google, NLR, SeaGuang and completed official cell-a projection',ranking='Retrospective descriptive test results; no new deployment choice'))
print(json.dumps(dict(all_passed=True,completed_rows=len(rows),development_rows=len(dev),checks=len(checks)),ensure_ascii=False))
