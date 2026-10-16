import sys, numpy as np, importlib
import engine
def stats(n=30):
    sk=np.array([0.97,1.0,1.0,1.0,1.0]); t=engine.default_tactic()
    r=engine.run(sk,sk,t,t,n,7); m=r.mean(0); sh=m[2]+m[3]
    return f'goals {m[0]:.2f}/{m[1]:.2f} shots {m[2]:.1f} xG/shot {m[4]/max(m[2],1):.3f} completion {(m[8]+m[9])/max(m[18],1):.2f} passes {m[8]:.0f} tackles {m[19]/2:.0f} shots<10s {m[15]/max(sh,1):.2f}'
print(stats())
