#!/usr/bin/env python3
"""
train_models.py — Huấn luyện bộ dự đoán CatBoost cho EdgeBench.

Tái hiện đúng pipeline trong notebook (52 features, 3-seed ensemble,
log1p cho Latency/Throughput/Energy). Chạy 1 lần để sinh model phục vụ web:

    python train_models.py --csv RTX_3080_results.csv    --platform rtx
    python train_models.py --csv JetsonNano_model.csv     --platform jetson

Kết quả lưu trong  models/<platform>/  gồm các file .cbm + meta.json
(meta.json chứa bảng tra cứu đặc trưng tĩnh theo tên model để /predict dùng).
"""
import argparse, json, re, os
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool

TARGETS = ['Accuracy', 'Latency', 'Throughput', 'Energy']
LOG_TARGETS = {'Accuracy': False, 'Latency': True, 'Throughput': True, 'Energy': True}
SEEDS = [42, 123, 2024]

SIZE_KEYWORDS = ['nano','pico','atto','femto','tiny','mini','small','medium',
                 'base','large','huge','giant','xlarge','xxlarge','xl','xxl']
ARCH_KEYWORDS = ['resnet','resnext','vit','swin','convnext','efficientnet','mobilenet','deit',
                 'beit','mixer','regnet','densenet','inception','xception','nfnet','coat','maxvit',
                 'eva','dino','clip','distilled','in22k','in21k','in1k','ft','pretrained',
                 'transformer','attention']

def extract_family(n):  return n.split('.')[0].split('_')[0].lower()
def extract_size(n):
    low=n.lower()
    for kw in SIZE_KEYWORDS:
        if re.search(rf'(?<![a-z]){kw}(?![a-z])', low): return kw
    return 'unknown'
def extract_patch(n):  m=re.search(r'patch(\d+)',n.lower());  return int(m.group(1)) if m else np.nan
def extract_res(n):
    m=re.search(r'_(\d{3,4})(?:[._]|$)',n)
    if m:
        v=int(m.group(1))
        if 96<=v<=1024: return v
    return np.nan
def extract_depth(n):  m=re.search(r'([a-z]+)(\d{1,3})',n.lower());  return int(m.group(2)) if m else np.nan
def extract_pretrain(n): return n.split('.',1)[1].lower() if '.' in n else 'none'

def engineer_features(df):
    out=df.copy(); names=out['Model Name'].astype(str)
    out['family_parsed']=names.apply(extract_family)
    out['size_variant'] =names.apply(extract_size)
    out['patch_size']   =names.apply(extract_patch)
    out['res_from_name']=names.apply(extract_res)
    out['depth_hint']   =names.apply(extract_depth)
    out['pretrain_tag'] =names.apply(extract_pretrain)
    out['name_length']  =names.str.len()
    out['num_tokens']   =names.str.count('_')+1
    for kw in ARCH_KEYWORDS:
        out[f'kw_{kw}']=names.apply(lambda x,k=kw:int(k in x.lower()))
    for col in ['params_tinfo','macs_tinfo','params_ptflops','macs_ptflops','activations']:
        out[f'log_{col}']=np.log1p(out[col].astype(float))
    out['log_macs_per_param']=np.log1p(out['macs_tinfo']/(out['params_tinfo']+1))
    out['log_act_per_param'] =np.log1p(out['activations']/(out['params_tinfo']+1))
    out['log_macs_per_act']  =np.log1p(out['macs_tinfo']/(out['activations']+1))
    out['input_area']        =out['Input Size'].astype(float)**2
    out['log_input_area']    =np.log1p(out['input_area'])
    out['log_macs_x_area']   =np.log1p(out['macs_tinfo']*out['input_area'])
    out['log_optimal_bs']    =np.log1p(out['Optimal BS'])
    out['log_max_bs']        =np.log1p(out['Max BS'])
    return out

NUMERIC=['Input Size','patch_size','res_from_name','depth_hint','name_length','num_tokens','param_count',
         'log_params_tinfo','log_macs_tinfo','log_params_ptflops','log_macs_ptflops','log_activations',
         'log_macs_per_param','log_act_per_param','log_macs_per_act','log_input_area','log_macs_x_area',
         'log_optimal_bs','log_max_bs']
CATEG=['family_parsed','size_variant','pretrain_tag','Family','Type']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True); ap.add_argument('--platform',required=True,choices=['rtx','jetson'])
    a=ap.parse_args()
    df=pd.read_csv(a.csv); df['param_count']=pd.to_numeric(df['param_count'],errors='coerce')
    df=df.dropna(subset=TARGETS).reset_index(drop=True)
    feat=engineer_features(df)
    KW=[c for c in feat.columns if c.startswith('kw_')]
    FEATURES=NUMERIC+KW+CATEG
    X=feat[FEATURES].copy()
    for c in NUMERIC+KW:
        if X[c].isna().any(): X[c]=X[c].fillna(X[c].median())
    for c in CATEG: X[c]=X[c].fillna('missing').astype(str)
    cat_idx=[X.columns.get_loc(c) for c in CATEG]

    outdir=f'models/{a.platform}'; os.makedirs(outdir,exist_ok=True)
    for t in TARGETS:
        y=np.log1p(df[t].values) if LOG_TARGETS[t] else df[t].values
        pool=Pool(X,y,cat_features=cat_idx)
        for s in SEEDS:
            m=CatBoostRegressor(iterations=1500,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,
                                random_seed=s,loss_function='RMSE',verbose=False,cat_features=cat_idx)
            m.fit(pool)
            m.save_model(f'{outdir}/{t}_seed{s}.cbm')
        print(f'  ✓ {t}: trained 3 seeds (log={LOG_TARGETS[t]})')

    # Bảng tra cứu đặc trưng tĩnh theo base_name (để /predict lấy params/macs... từ tên)
    lookup={}
    for _,r in df.iterrows():
        bn=str(r['Model Name']).split('.')[0]
        lookup.setdefault(bn,{k:(float(r[k]) if pd.notna(r[k]) else None) for k in
            ['Input Size','Optimal BS','Max BS','param_count','params_tinfo','macs_tinfo',
             'params_ptflops','macs_ptflops','activations']} | {'Model Name':r['Model Name'],
             'Family':str(r['Family']),'Type':str(r['Type'])})
    json.dump({'features':FEATURES,'numeric':NUMERIC,'categ':CATEG,'cat_idx':cat_idx,
               'log_targets':LOG_TARGETS,'medians':{c:float(X[c].median()) for c in NUMERIC+KW},
               'lookup':lookup}, open(f'{outdir}/meta.json','w'))
    print(f'✓ Saved models + meta.json -> {outdir}/  ({len(lookup)} models in lookup)')

if __name__=='__main__': main()
