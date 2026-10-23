from pathlib import Path
import argparse,json,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--kind',choices=['horizon'],required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=Path(a.root);designpath=Path(__file__).with_name('design.json');d=json.loads(designpath.read_text());frozen=root/'instrumented_source_v2'
assert hashlib.sha256((frozen/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==d['source_manifest_sha256']
for name,digest in json.loads((frozen/'SOURCE_MANIFEST.json').read_text()).items():assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(frozen/'AAMAS2027/experiments'))
import horizon_instrumented as h
import scale_control as sc
from instrumented_ppo import RolloutDiagnostics
n=len(d['seeds']);grid=d[a.kind+'_grid'];conditions=d['horizon_conditions'] if a.kind=='horizon' else d['control_arms'];assert 0<=a.index<len(grid)*len(conditions)*n
w=grid[a.index//(len(conditions)*n)];condition=conditions[(a.index//n)%len(conditions)];seed=d['seeds'][a.index%n]
out=root/'new_runs'/('v4_e3_smoke' if a.smoke else 'v4_e3_horizon')/a.kind/f'run_{a.index:03d}'
budget=1280 if a.smoke else d[a.kind+'_budget']
if a.kind=='horizon':
 cfg=dict(w=w,condition=condition,seed=seed,checkpoints=[1280] if a.smoke else d['checkpoints'],total_budget=budget,phase='smoke' if a.smoke else 'confirmation',evaluate=True,version='explicit-demand-v2',amendment=d['amendment'])
 m=h.create(cfg,out);m.stop_after=0;(out/'protocol.json').write_text(json.dumps(cfg,indent=2));(out/'manifest.json').write_text(json.dumps(h.h.manifest(),indent=2))
else:m=sc.create(out,w,condition,seed,c=7. if condition=='S' else 1.,reference=d['reference'],budget=budget,phase='smoke' if a.smoke else 'confirmation')
(out/'lineage.json').write_text(json.dumps(dict(design_sha256=hashlib.sha256(designpath.read_bytes()).hexdigest(),orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_manifest_sha256=d['source_manifest_sha256']),indent=2))
m.learn(total_timesteps=budget,callback=RolloutDiagnostics())
(out/'COMPLETE.json').write_text(json.dumps(dict(actual_transitions=m.num_timesteps,requested=budget)))
print('COMPLETE',a.kind,a.index,m.num_timesteps,flush=True)
