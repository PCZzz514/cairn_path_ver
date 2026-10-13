from pathlib import Path
from collections import deque
import json,numpy as np,pandas as pd,torch
root=Path.cwd();out=root/'reports/validation_pairing';events=pd.read_csv(root/'data/prepared/ddi_events.csv');templates=dict(zip(events.event_id,events.template));src=pd.read_parquet(root.parent/'cairn_dev/data/experiments/no_gene_embeddings_six_relations/shared/neighbours.parquet');raw=src.groupby('drug').size().to_dict();genes=src.groupby('drug').fin_idx.nunique().to_dict()
labels=['修正_错对','退化_对错','未解决_错错','保持_对对'];summaries=[];groups=[];relgroups=[];metadata={}
def stats(f):
 c=f.outcome.value_counts();n=len(f);a,b,u,d=[int(c.get(x,0)) for x in labels];return {'样本数':n,**dict(zip(labels,[a,b,u,d])),'基础Accuracy':(b+d)/n,'含路径Accuracy':(a+d)/n,'净修正数':a-b,'净Accuracy变化':(a-b)/n,'基础错误修正率':a/(a+u) if a+u else None,'基础正确退化率':b/(b+d) if b+d else None}
def support(n):return '0' if n==0 else '1–9' if n<10 else '10–99' if n<100 else '100–999' if n<1000 else '≥1000'
def bg(n,used=False):return '0' if n==0 else '1–5' if n<=5 else ('6–15' if n<=15 else '16–30') if used else '6–30' if n<=30 else '>30'
def dist(adj,s):
 d={int(s):0};q=deque([int(s)])
 while q:
  u=q.popleft()
  if d[u]>=4:continue
  for v in adj.get(u,[]):
   if v not in d:d[v]=d[u]+1;q.append(v)
 return d
for arm in ['pairout','cellout','single_drugout','double_drugout']:
 meta={v:json.load(open(out/f'{arm}_{v}_metadata.json')) for v in ['base','path']};metadata[arm]=meta;z={v:np.load(out/f'{arm}_{v}_validation.npz') for v in meta}
 assert np.array_equal(z['base']['labels'],z['path']['labels']) and np.array_equal(z['base']['frozen_row_indices'],z['path']['frozen_row_indices'])
 prep=Path(meta['path']['prepared']);whole=pd.read_parquet(prep/arm/'ddi.parquet');idx=z['base']['frozen_row_indices'];f=whole.iloc[idx].copy().reset_index(drop=True);assert f.split.eq('val').all() and np.array_equal(f.event_id,z['base']['labels']);f.insert(0,'frozen_row_index',idx);f.insert(0,'protocol',arm)
 y=f.event_id.to_numpy();bp=z['base']['probabilities'].argmax(1);pp=z['path']['probabilities'].argmax(1);bc=bp==y;pc=pp==y;f['base_prediction']=bp;f['path_prediction']=pp;f['base_correct']=bc;f['path_correct']=pc;f['outcome']=np.select([~bc&pc,bc&~pc,~bc&~pc,bc&pc],labels,default='invalid');f['base_true_probability']=z['base']['probabilities'][np.arange(len(y)),y];f['path_true_probability']=z['path']['probabilities'][np.arange(len(y)),y]
 count=whole.loc[whole.split.eq('train'),'event_id'].value_counts();f['train_support']=f.event_id.map(count).fillna(0).astype(int);f['train_support_bin']=f.train_support.map(support)
 for col,pred in [('true_event_template','event_id'),('base_event_template','base_prediction'),('path_event_template','path_prediction')]:f[col]=f[pred].map(templates)
 assets=torch.load(prep/'assets.pt',map_location='cpu',weights_only=False);local=(assets['neighbor_entities']>=0).sum(1).numpy();fin=assets['drug_fin_idx'].numpy();relations=assets['fin_relation_ids'];entityids=assets['fin_entity_ids'];nr=len(relations)
 for side,column in [('A','drug_a'),('B','drug_b')]:
  ids=f[column].to_numpy();f[side+'_drug_id']=[assets['drug_ids'][int(i)] for i in ids];f[side+'_six_relation_count_raw']=f[column].map(raw).fillna(0).astype(int);f[side+'_unique_genes_raw']=f[column].map(genes).fillna(0).astype(int);f[side+'_neighbors_used']=local[ids];f[side+'_FIN_mapped']=fin[ids]>=0;f[side+'_raw_coverage_bin']=f[side+'_six_relation_count_raw'].map(bg);f[side+'_local_coverage_bin']=f[side+'_neighbors_used'].map(lambda n:bg(n,True))
 a=f.A_neighbors_used>0;b=f.B_neighbors_used>0;f['background_presence']=np.select([a&b,a&~b,~a&b],['双侧NAI有背景','仅A侧NAI有背景','仅B侧NAI有背景'],default='双侧NAI无背景');f['coverage_raw_pair']=f.A_raw_coverage_bin+' / '+f.B_raw_coverage_bin;f['coverage_local_pair']=f.A_local_coverage_bin+' / '+f.B_local_coverage_bin
 masks=z['path']['path_masks'];f['path_B_to_A_present']=masks[:,0];f['path_A_to_B_present']=masks[:,1];f['path_presence']=np.select([masks.all(1),masks[:,1]&~masks[:,0],masks[:,0]&~masks[:,1]],['双向存在','仅A→B','仅B→A'],default='双向缺失')
 pairids=np.load(prep/arm/'pair_indices.npy',mmap_mode='r')[idx];arrays={k:np.load(prep/(k+'.npy'),mmap_mode='r') for k in ['edge_ptr','node_ptr','edges','nodes','endpoints']};features={}
 for pid in np.unique(pairids):
  e=arrays['edges'][arrays['edge_ptr'][pid]:arrays['edge_ptr'][pid+1]];nodes=arrays['nodes'][arrays['node_ptr'][pid]:arrays['node_ptr'][pid+1]];adj={};rev={}
  for u,v,r,m in e:
   if u==v or r==2*nr or m<=0:continue
   adj.setdefault(int(u),[]).append(int(v));rev.setdefault(int(v),[]).append(int(u))
  h,t=map(int,arrays['endpoints'][pid]);feature={};types=set();families=set()
  for tag,s,target in [('AB',h,t),('BA',t,h)]:
   ds=dist(adj,s);dt=dist(rev,target);length=ds.get(target);feature[tag]=length;direction=set()
   if length is not None:
    for u,v,r,m in e:
     u,v,r=int(u),int(v),int(r)
     if u==v or r==2*nr or m<=0 or ds.get(u,999)+1+dt.get(v,999)!=length:continue
     direction.add(relations[r%nr]+(' [逆向]' if r>=nr else ' [原向]'));at=entityids[int(nodes[u])].split('::',1)[0];bt=entityids[int(nodes[v])].split('::',1)[0];families.add('—'.join(sorted([at,bt])))
   feature[tag+'_relations']=sorted(direction);types.update(direction)
  feature['relations']=sorted(types);feature['families']=sorted(families);features[int(pid)]=feature
 f['path_pair_index']=pairids
 for tag,column in [('AB','path_A_to_B_present'),('BA','path_B_to_A_present')]:
  lengths=[features[int(i)][tag] for i in pairids];assert np.array_equal(np.array([x is not None for x in lengths]),f[column].to_numpy());f['shortest_'+tag+'_length']=lengths;f['shortest_'+tag+'_bin']=[str(x) if x is not None else '无连接路径' for x in lengths];f['shortest_'+tag+'_relations']=['; '.join(features[int(i)][tag+'_relations']) for i in pairids]
 f['shortest_path_relation_types']=['; '.join(features[int(i)]['relations']) for i in pairids];f['shortest_path_entity_type_pairs']=['; '.join(features[int(i)]['families']) for i in pairids];f.to_parquet(out/(arm+'_逐样本配对.parquet'),index=False);f.to_csv(out/(arm+'_逐样本配对.csv'),index=False,encoding='utf-8-sig')
 summary={'划分':arm,'基础最佳轮次':meta['base']['selected_epoch'],'含路径最佳轮次':meta['path']['selected_epoch'],**stats(f),'基础85类Macro-F1':meta['base']['metrics']['macro_f1_85'],'含路径85类Macro-F1':meta['path']['metrics']['macro_f1_85']};summaries.append(summary)
 dimensions={'真实事件':'event_id','训练支持':'train_support_bin','单药背景是否存在':'background_presence','单药原始关系覆盖A':'A_raw_coverage_bin','单药原始关系覆盖B':'B_raw_coverage_bin','单药原始覆盖配对':'coverage_raw_pair','单药模型邻域覆盖A':'A_local_coverage_bin','单药模型邻域覆盖B':'B_local_coverage_bin','单药模型邻域覆盖配对':'coverage_local_pair','路径是否存在':'path_presence','路径长度A→B':'shortest_AB_bin','路径长度B→A':'shortest_BA_bin'}
 for dimension,column in dimensions.items():
  for value,sub in f.groupby(column,sort=True):
   row={'划分':arm,'分组维度':dimension,'分组值':str(value),**stats(sub)}
   if column=='event_id':row.update(event_id=int(value),真实事件模板=templates[int(value)],训练支持=int(count.get(value,0)))
   groups.append(row)
 for dimension,key in [('最短连接路径关系类型','relations'),('最短连接路径实体类型','families')]:
  membership={}
  for pos,pid in enumerate(pairids):
   values=features[int(pid)][key] or ['无连接路径' if not masks[pos].any() else '仅自环或零长度']
   for value in values:membership.setdefault(value,[]).append(pos)
  for value,positions in sorted(membership.items()):relgroups.append({'划分':arm,'分组维度':dimension,'分组值':value,'多重归属':True,**stats(f.iloc[positions])})
 print(json.dumps(summary,ensure_ascii=False),flush=True)
pd.DataFrame(summaries).to_csv(out/'配对总览.csv',index=False,encoding='utf-8-sig');pd.DataFrame(groups).to_csv(out/'事件支持背景路径分组.csv',index=False,encoding='utf-8-sig');pd.DataFrame(relgroups).to_csv(out/'关系类型分组_允许多重归属.csv',index=False,encoding='utf-8-sig');(out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n');(out/'inference_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
notes={'模型':'基础=DSN-DDI+RareDDIE，含路径=DSN-DDI+EmerGNN+RareDDIE。各自验证选出的最佳检查点，非同权重切换路径的因果实验。','数据':'同一冻结验证行、有序药物对和标签。PairOut与CellOut DDI数据完全相同，为独立运行。未使用测试预测。','覆盖':'六类NAI原始关系记录数/独立基因数与实际cap30输入数分别保留。FIN映射与邻域覆盖分开。','路径':'模型实际使用的受限缓存子图中，忽略自环，计算≤4跳有向最短距离；不是全DRKG无约束最短距离。关系类型为双向所有最短连接路径边的并集，逆向显式标记。','分母':'修正率=错对/基础错误，退化率=对错/基础正确，净Accuracy变化=(错对-对错)/分组样本数。无分母记空，不记0。','分组':'事件ID、训练支持0/1–9/10–99/100–999/≥1000、原始覆盖0/1–5/6–30/>30、实际输入0/1–5/6–15/16–30。关系类型允许多重归属，不可跨组求和。小样本分组仅作探索。','评价':'保留固定85类Macro-F1。验证内诊断，不在测试上筛选路径规则。'}
(out/'analysis_definition.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
