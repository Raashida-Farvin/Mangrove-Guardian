"""pip install fastapi uvicorn xgboost shap scikit-learn pandas joblib ; python train.py ; uvicorn api:app --reload"""
import joblib, shap, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
FEATS=["ndvi_change","ndre","ndmi","salinity_ppt","dissolved_oxygen_mg_l","soil_ph","inundation_hours","healthy_tree_pct","seedling_survival_pct","grazing_detected"]
app=FastAPI();app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
model=joblib.load("model.joblib");ex=shap.TreeExplainer(model)
@app.post("/predict")
def predict(rows:list[dict]):
    df=pd.DataFrame(rows);X=df[FEATS];p=model.predict_proba(X)[:,1];sv=ex.shap_values(X);out=[]
    for i in range(len(df)):
        top=sorted(zip(FEATS,sv[i]),key=lambda t:-abs(t[1]))[:4]
        out.append({"grid_id":df.get("grid_id",pd.Series([None]*len(df)))[i],"risk_probability":round(float(p[i]),3),
         "risk_category":"Low" if p[i]<.36 else "Medium" if p[i]<.66 else "High","model_status":"Prototype",
         "contributing_factors":[{"feature":f,"log_odds":round(float(v),3)} for f,v in top]})
    return out
