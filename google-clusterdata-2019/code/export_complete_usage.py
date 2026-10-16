"""Export all five real-usage domains; absent cells are not presented as zero load."""
import complete_usage_models as ex
import numpy as np,pandas as pd
def main():
    d,v=ex.inputs();rows=[]
    for i in np.where(d['cells']=='a')[0]:
        for o in range(31):
            for t in range(96):
                u=v['u'][i,o,t];cap=v['cap'][i,o,t];r=v['r'][i,o,t];rows.append(dict(series=d['names'][i],local_trace_time_PT=str(pd.Timestamp('2019-05-01')+pd.Timedelta(days=o,minutes=15*t)),completed_job_cpu_ncu=u[0],completed_job_memory_normalized=u[1],job_window_equivalent=u[2],job_window_records=int(u[3]),mean_samples_per_second=u[4]/u[5] if u[5]>0 else np.nan,cpu_scheduler_capacity_ncu=cap[0],memory_scheduler_capacity_normalized=cap[1],cpu_capacity_ratio=u[0]/cap[0] if cap[0]>0 else np.nan,memory_capacity_ratio=u[1]/cap[1] if cap[1]>0 else np.nan,any_job_telemetry=bool(v['present'][i,o,t]),cpu_capacity_known_fraction=r[4],memory_capacity_known_fraction=r[5],measured_power_util=d['y'][i,o,t],power_quality_valid=bool(d['valid'][i,o,t]),production_power_proxy=d['prod'][i,o,t],proxy_quality_valid=bool(d['pvalid'][i,o,t])))
    path=ex.ROOT/'aggregated/cella_complete_usage_15min.csv';pd.DataFrame(rows).to_csv(path,index=False);print(path,'rows',len(rows),flush=True)
if __name__=='__main__':main()
