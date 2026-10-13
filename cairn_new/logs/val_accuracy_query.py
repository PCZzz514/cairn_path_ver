import torch,json,sys
from pathlib import Path
from torch.utils.data import DataLoader
from cairn_dsn.train import evaluate
from cairn_dsn.model import DSNRareDDI
from cairn_dsn.data import DDIDataset,Collator
torch.set_num_threads(2)
arm=sys.argv[1]
root=Path.cwd()
saved=torch.load(root/'outputs'/arm/'best.pt',map_location='cpu',weights_only=False)
c=saved['config']; prepared=Path(c['prepared'])
assets=torch.load(prepared/'assets.pt',map_location='cpu',weights_only=False)
m=DSNRareDDI(assets,c).to('cuda:0');m.load_state_dict(saved['model'])
ds=DDIDataset(prepared,c['protocol'],'val')
loader=DataLoader(ds,batch_size=c['batch_size'],num_workers=2,collate_fn=Collator(assets,prepared,c['with_path']),pin_memory=True)
metrics,p,y,_,_=evaluate(m,loader,torch.device('cuda:0'),c,c['kl_weight'])
result={'arm':arm,'checkpoint':'best.pt','epoch':saved['epoch'],'rows':len(y),'accuracy':float((p.argmax(1)==y).mean()),'macro_f1_85':metrics['macro_f1_85']}
print(json.dumps(result),flush=True)
