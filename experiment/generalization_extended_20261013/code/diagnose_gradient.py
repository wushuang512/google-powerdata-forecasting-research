"""Isolated replay of a failed training window; no production checkpoints edited."""
from pathlib import Path
import shutil,sys
root=Path(sys.argv[1]);main=root/'experiment/generalization_extended_20261013';diag=main/'diagnostic_gradient'
for folder in ['docs','data','models','results','cache']:(diag/folder).mkdir(parents=True,exist_ok=True)
for name in ['protocol.json','protocol.sha256']:shutil.copy2(main/'docs'/name,diag/'docs'/name)
shutil.copy2(main/'data/NLR_2024_2025.npz',diag/'data/NLR_2024_2025.npz')
source=main/'code/neural_stage.py';code=source.read_text(encoding='utf-8')
code=code.replace("F=ROOT/'experiment/generalization_extended_20261013';old=", "F=ROOT/'experiment/generalization_extended_20261013/diagnostic_gradient';old=")
needle='assert gradient>0 and np.isfinite(gradient);torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()'
replacement="""if not (gradient>0 and np.isfinite(gradient)):
            details=dict(step=step,gradient=gradient if np.isfinite(gradient) else str(gradient),origin_tick=s,horizon=h,input_std=float(x.std()),target_std=float(yy.std()),mask_rate=float(mm.mean()),bad_gradients=[n for n,v in adapter.model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()])
            save(F/'results/gradient_diagnostic.json',details);print('GRADIENT_DIAGNOSTIC',details,flush=True);raise RuntimeError('Isolated gradient diagnostic complete')
        torch.nn.utils.clip_grad_norm_(params,1);optimizer.step()"""
assert needle in code;code=code.replace(needle,replacement)
exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
