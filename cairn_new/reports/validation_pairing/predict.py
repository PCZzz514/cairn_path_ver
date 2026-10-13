import os,sys,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from accelerated_four.dsn.model import DSNRareDDI
from accelerated_four.dsn.data import DDIDataset,Collator
from accelerated_four.dsn.train import evaluate
root=Path.cwd();out=root/'reports/validation_pairing';arm=sys.argv[1]
cold='drugout' in arm
for variant in ['base','path']:
 d=root/(('accelerated_four/outputs/'+('no_path_' if variant=='base' else 'path_')+arm) if cold else (('dsn_pairout_cellout' if variant=='base' else 'pairout_cellout')+'/outputs/'+arm))
 saved=torch.load(d/'best.pt',map_location='cpu',weights_only=False);c=saved['config'];torch.set_num_threads(2)
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
 prepared=Path(c['prepared']);prepared=prepared if prepared.is_absolute() else root/prepared
 assets=torch.load(prepared/'assets.pt',map_location='cpu',weights_only=False);model=DSNRareDDI(assets,c).to('cuda:0');model.load_state_dict(saved['model'],strict=True)
 dataset=DDIDataset(prepared,c['protocol'],'val');loader=DataLoader(dataset,batch_size=c['batch_size'],num_workers=2,collate_fn=Collator(assets,prepared,c['with_path']),pin_memory=True,persistent_workers=False)
 metrics,p,y,rows,masks=evaluate(model,loader,torch.device('cuda:0'),c,c['kl_weight'])
 np.savez_compressed(out/f'{arm}_{variant}_validation.npz',probabilities=p,labels=y,frozen_row_indices=rows,path_masks=masks)
 meta={'arm':arm,'variant':variant,'checkpoint':str(d/'best.pt'),'selected_epoch':saved['epoch'],'prepared':str(prepared),'metrics':metrics,'accuracy':float((p.argmax(1)==y).mean()),'stored_best_val_f1':saved['best_metric'],'inference_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'classification':'deterministic mu; frozen validation rows; own validation-selected best'}
 (out/f'{arm}_{variant}_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
 del model,assets,loader,saved;torch.cuda.empty_cache()
