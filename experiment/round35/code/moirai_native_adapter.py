"""Run unchanged official neural module and forecast mathematical methods, without unused Lightning/GluonTS service wrappers."""
from pathlib import Path
import ast,inspect,importlib.util,json,math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any,Generator,List,Optional
import numpy as np
import numpy
import torch
from einops import rearrange,reduce,repeat
from jaxtyping import Bool,Float,Int
class AttrDict(dict):
 def __getattr__(self,key):
  try:return self[key]
  except KeyError:raise AttributeError(key) from None
 __setattr__=dict.__setitem__
def load_module(root):
 src=Path(root)/'code/official_uni2ts/src/uni2ts/model/moirai2/module.py';spec=importlib.util.spec_from_file_location('official_moirai2_bare_module',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.Moirai2Module
def load_forecast_class(root,Module):
 src=Path(root)/'code/official_uni2ts/src/uni2ts/model/moirai2/forecast.py';tree=ast.parse(src.read_text(encoding='utf-8'));cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Moirai2Forecast');keep={'__init__','hparams_context','past_length','context_token_length','prediction_token_length','max_patch_size','forward','predict','_patched_seq_pad','_generate_time_id','_convert','_format_preds'};cl.body=[x for x in cl.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in keep];cl.bases=[ast.parse('torch.nn.Module',mode='eval').body];ns=dict(globals(),Moirai2Module=Module);exec(compile(ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[])),str(src),'exec'),ns);cls=ns['Moirai2Forecast']
 def save_hyperparameters(self,ignore):
  local=inspect.currentframe().f_back.f_locals;fields=['prediction_length','target_dim','feat_dynamic_real_dim','past_feat_dynamic_real_dim','context_length','module_kwargs'];self.hparams=AttrDict({k:local[k] for k in fields})
 cls.save_hyperparameters=save_hyperparameters;cls.device=property(lambda self:next(self.parameters()).device);return cls
def repair_joint_initial_write(cls,root):
 # Upstream flattens target_dim*4 prediction tokens but fixes the rearrange axis to4.
 # Restrict the repair to dim>1; every univariate operation is unchanged.
 src=Path(root)/'code/official_uni2ts/src/uni2ts/model/moirai2/forecast.py';tree=ast.parse(src.read_text(encoding='utf-8'));cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Moirai2Forecast');forward=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='forward');fixed=0
 for node in ast.walk(forward):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='rearrange' and len(node.args)>1 and isinstance(node.args[1],ast.Constant) and node.args[1].value=='... predict_token num_quantiles patch_size -> ... num_quantiles predict_token patch_size':
   for kw in node.keywords:
    if kw.arg=='predict_token':kw.value=ast.parse('len(adjusted_assign_index) if self.hparams.target_dim > 1 else self.module.num_predict_token',mode='eval').body;fixed+=1
 assert fixed==1
 ns=dict(globals());exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),str(src),'exec'),ns);cls.forward=ns['forward'];return cls
def build(root,context,horizon,dim=1):
 from safetensors.torch import load_file
 root=Path(root);Module=load_module(root);cfg=json.loads((root/'models/moirai2/config.json').read_text(encoding='utf-8'));module=Module(**cfg);info=module.load_state_dict(load_file(root/'models/moirai2/model.safetensors'),strict=True);assert not info.missing_keys and not info.unexpected_keys;Cls=load_forecast_class(root,Module)
 if dim>1:Cls=repair_joint_initial_write(Cls,root)
 return Cls(prediction_length=horizon,target_dim=dim,feat_dynamic_real_dim=0,past_feat_dynamic_real_dim=0,context_length=context,module=module).eval()
