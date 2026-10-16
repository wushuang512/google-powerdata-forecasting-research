from pathlib import Path
src=Path(__file__).with_name('joint_objective_experiment.py').read_text(encoding='utf-8')
src=src.replace("R=B/'round15'","R=B/'round16'")
src=src.replace("'models/timesfm3-base/model.safetensors']","'models/timesfm3-base/model.safetensors','round15/results/tf3_dev7.npz','round15/results/tf3_dev7_verification.json','round15/docs/protocol.json']")
src=src.replace('new DEV672 inference','reuse independently checked round15 DEV672 cache; identical origins and original base weights')
src=src.replace('Preserve predicted rank shape; affine gap from maximum, clipped into [0,target_peak], bisection enforces full96 predicted daily mean. Peak>=mean, power in [0,1]. This is shape consistency, not energy conservation.','Minimal squared change to each daily curve with one causally chosen maximum pinned at target peak, other points projected by a shared offset into [0,target_peak]; bisection enforces feasible full96 daily mean. It may alter peak timing after capping, which is evaluated. Shape consistency, not energy conservation.')
a=src.index('def project(');b=src.index('def candidate(',a)
src=src[:a]+'''def project(shape,mean,peak,history):
 mean=np.clip(mean,0,1);peak=np.clip(np.maximum(peak,mean),0,1);mean=np.maximum(mean,peak/96)
 rows=np.arange(len(mean));index=np.argmax(shape,-1);flat=np.ptp(shape,axis=-1)<1e-12;index[flat]=np.argmax(history[flat],-1)
 low=np.full(len(mean),-2.);high=np.full(len(mean),2.)
 for _ in range(55):
  mid=(low+high)/2;v=np.clip(shape+mid[:,None],0,peak[:,None]);v[rows,index]=peak;bad=v.mean(-1)>mean
  high=np.where(bad,mid,high);low=np.where(bad,low,mid)
 out=np.clip(shape+(.5*(low+high))[:,None],0,peak[:,None]);out[rows,index]=peak
 assert abs(out.mean(-1)-mean).max()<1e-9 and abs(out.max(-1)-peak).max()<1e-9
 return out
''' +src[b:]
Path(__file__).with_name('joint_local_experiment.py').write_text(src,encoding='utf-8')
guard=Path(__file__).with_name('run_joint_guarded.py').read_text(encoding='utf-8').replace("R=T/'experiment/round15'","R=T/'experiment/round16'").replace("['dev7','develop','test']","['develop','test']").replace('joint_objective_experiment.py','joint_local_experiment.py')
Path(__file__).with_name('run_joint_local_guarded.py').write_text(guard,encoding='utf-8')
