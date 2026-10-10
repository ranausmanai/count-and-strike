"""P4 robustness (exploratory): own shot-share control, eras, windows, pinnacle-closing only."""
import os, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
ROOT=os.path.join(os.path.dirname(__file__),"..")
long=pd.read_parquet(os.path.join(ROOT,"data","proc","team_matches_fd.parquet"))
mk=pd.read_parquet(os.path.join(ROOT,"data","proc","market.parquet"))
und=long[(long.pwin<long.ploss)].dropna(subset=["tempo_dev","opp_tempo_dev","share_raw"]).copy()
def fit(d,f="resid ~ tempo_dev + opp_tempo_dev + gap + home + C(league)"):
    m=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d.ts)[0]})
    return dict(n=len(d),b=round(m.params["tempo_dev"],4),se=round(m.bse["tempo_dev"],4),p=round(m.pvalues["tempo_dev"],4))
out={}
out["base"]=fit(und)
out["+share_control"]=fit(und,"resid ~ tempo_dev + share_raw + opp_tempo_dev + opp_share + gap + home + C(league)")
out["2005-2014"]=fit(und[und.season<=2014]); out["2015-2025"]=fit(und[und.season>=2015])
for lg,d in und.groupby("league"): out[lg]=fit(d,"resid ~ tempo_dev + opp_tempo_dev + gap + home")
out["pwin 0.25-0.4"]=fit(und[(und.pwin>=0.25)])
# window sensitivity
for w in [5,20]:
    l=long.sort_values(["league","team","date"]).copy(); g=l.groupby(["league","team"])
    l["tr"]=g["tot"].transform(lambda s:s.shift(1).rolling(w,min_periods=w).mean())
    l["tempo_dev"]=l.tr-l.groupby(["league","season"])["tot"].transform("mean")
    o=l[["mid","team","tempo_dev"]].rename(columns={"team":"opp","tempo_dev":"opp_tempo_dev"})
    l=l.drop(columns=["opp_tempo_dev"]).merge(o,on=["mid","opp"],how="left")
    out[f"window{w}"]=fit(l[(l.pwin<l.ploss)].dropna(subset=["tempo_dev","opp_tempo_dev"]))
print("tempo_dev sd among underdogs:",und.tempo_dev.std())
for k,v in out.items(): print(k,v)
json.dump(out,open(os.path.join(ROOT,"results","p4_robust.json"),"w"),indent=1,default=float)
