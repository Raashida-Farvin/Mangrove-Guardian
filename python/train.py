"""XGBoost grid-risk model, site-held-out validation, SHAP.
python train.py              -> SYNTHETIC data (pipeline demo only, not science)
python train.py data.csv     -> needs columns: site, <FEATS>, future_deterioration"""
import sys, joblib, numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
FEATS=["ndvi_change","ndre","ndmi","salinity_ppt","dissolved_oxygen_mg_l","soil_ph","inundation_hours","healthy_tree_pct","seedling_survival_pct","grazing_detected"]
def synthetic(n=800,seed=1):
    r=np.random.default_rng(seed);s=r.random(n);h=r.random(n);gz=(r.random(n)<.08+.3*s).astype(int);N=lambda x:r.normal(0,x,n)
    d=pd.DataFrame({"site":r.integers(0,8,n),"ndvi_change":-.22*s+N(.03),"ndre":.5-.2*s+N(.03),"ndmi":.4-.2*s+N(.03),
     "salinity_ppt":40+35*s+N(4),"dissolved_oxygen_mg_l":6-3*s+N(.4),"soil_ph":7.2+.6*s+N(.1),"inundation_hours":np.clip(7-5*h+N(.5),.5,None),
     "healthy_tree_pct":np.clip(92-45*s+N(5),0,99),"seedling_survival_pct":np.clip(92-40*s-10*h+N(5),0,99),"grazing_detected":gz})
    d["future_deterioration"]=(r.random(n)<1/(1+np.exp(-(5*(.55*s+.45*h-.5)+1.2*gz)))).astype(int);return d
mk=lambda:XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.08,subsample=.8,eval_metric="logloss")
if __name__=="__main__":
    df=pd.read_csv(sys.argv[1]) if len(sys.argv)>1 else synthetic()
    X,y,g=df[FEATS],df["future_deterioration"],df["site"];aucs=[]
    for tr,te in GroupKFold(n_splits=min(5,g.nunique())).split(X,y,g):   # whole sites held out
        aucs.append(roc_auc_score(y.iloc[te],mk().fit(X.iloc[tr],y.iloc[tr]).predict_proba(X.iloc[te])[:,1]))
    print("AUC per held-out site group:",np.round(aucs,3))
    m=mk().fit(X,y);joblib.dump(m,"model.joblib")
    sv=shap.TreeExplainer(m).shap_values(X);print(pd.Series(np.abs(sv).mean(0),FEATS).sort_values(ascending=False))
