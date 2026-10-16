from pathlib import Path
import json, hashlib, gzip, base64
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'google-powerdata-2019'
if not SOURCE.exists(): SOURCE = Path(r'E:\计算中心电负荷\google-powerdata-2019')

def main():
    ys, ps, masks, pmasks, names, cells, audit = [], [], [], [], [], [], []
    manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    for info in manifest['files']:
        name = info['file']
        data = (SOURCE/'raw'/name).read_bytes()
        assert base64.b64encode(hashlib.md5(data).digest()).decode() == info['md5_base64']
        if name.startswith('machine'): continue
        assert gzip.decompress(data)==(SOURCE/'csv'/name[:-3]).read_bytes(), 'CSV differs from verified original'
        d = pd.read_csv(SOURCE/'csv'/name[:-3]).sort_values('time')
        assert len(d)==8928 and np.array_equal(d.time.to_numpy(),600000000+np.arange(8928)*300000000)
        raw = d.measured_power_util.to_numpy(float)
        prod = d.production_power_util.to_numpy(float)
        valid = ~d.bad_measurement_data.to_numpy(bool) & np.isfinite(raw) & (raw>=0) & (raw<=1)
        pv = ~d.bad_production_power_data.to_numpy(bool) & np.isfinite(prod) & (prod>=0) & (prod<=1)
        # Never fill backwards. No complete-series normalisation is used.
        filled = pd.Series(np.where(valid,raw,np.nan)).ffill().to_numpy()
        # Some domains have no reliable production measurement at all. A zero
        # placeholder plus an explicit availability mask is not a measurement.
        pfilled = pd.Series(np.where(pv,prod,np.nan)).ffill().fillna(0.0).to_numpy()
        assert np.isfinite(filled).all() and np.isfinite(pfilled).all()
        ys.append(filled.reshape(-1,3).mean(1).reshape(31,96))
        ps.append(pfilled.reshape(-1,3).mean(1).reshape(31,96))
        masks.append(valid.reshape(-1,3).all(1).reshape(31,96))
        pmasks.append(pv.reshape(-1,3).all(1).reshape(31,96))
        names.append(name[:-7]); cells.append(str(d.cell.iloc[0]))
        audit.append(dict(series=names[-1],cell=cells[-1],bad_measured_5min=int((~valid).sum()),bad_production_5min=int((~pv).sum()),valid_15min=int(masks[-1].sum())))
    np.savez_compressed(ROOT/'results/data.npz',y=np.array(ys),prod=np.array(ps),valid=np.array(masks),pvalid=np.array(pmasks),names=names,cells=cells)
    pd.DataFrame(audit).to_csv(ROOT/'results/data_audit.csv',index=False)
    total=dict(series=len(ys),days=31,points_per_day=96,original_records=57*8928,aggregated_records=57*2976,valid_test24=int(np.array(masks)[:,22:31].sum()),bad_measured_5min=sum(a['bad_measured_5min'] for a in audit),bad_production_5min=sum(a['bad_production_5min'] for a in audit),source=str(SOURCE),anomaly_policy='exclude bad_measurement_data=true from targets; historical forward fill only; aggregate mask requires all three five-minute measurements valid')
    (ROOT/'results/data_audit.json').write_text(json.dumps(total,indent=2),encoding='utf-8')
    print(json.dumps(total,indent=2),flush=True)

if __name__=='__main__': main()
