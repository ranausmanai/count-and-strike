"""Build a self-contained HTML replay of simulated matches (engine4): underdog (red) vs strong team (blue).
usage: python make_replay.py [calibration.json] [out.html]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import engine4 as e

HERE = os.path.dirname(__file__)
calf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "calibration.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "site", "sim_replay.html")
CAL = np.array(json.load(open(calf))["cal"])
WEAK, STRONG = [0.955, .85, .85, .85, .95], [0.97, 1, 1, 1, 1]
opp = e.default_tactic(); opp[13] = 1.0
base = e.default_tactic(); base[13] = 1.0
sling = base.copy(); sling[10:13] = [0.8, 1.0, 1.0]
esc = base.copy(); esc[12] = 1.0
MIN = 6
scen = {}
for name, tac, desc in (("Normal play", base, "Underdog plays a standard 4-4-2."),
                        ("Escape", esc, "Underdog goes long only where it has +1 near the landing spot; two forwards move there."),
                        ("Slingshot", sling, "Lopsided shape with one isolated far winger, release to him after 3 s on one side, plus Escape.")):
    tr, st = e.replay(WEAK, STRONG, tac, opp, 11, CAL, minutes=MIN)
    scen[name] = dict(desc=desc, frames=np.round(tr[:, :43], 1).tolist(), score=[[int(a), int(b)] for a, b in tr[:, 43:45]])
html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simulator Replay</title>
<style>
:root{--bg:#f6f5f0;--ink:#17201b;--muted:#5b6660;--felt:#1e4a38;--line:#d7e6db;--red:#d9553a;--blue:#3b6fc0;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121614;--ink:#e8ede9;--muted:#9aa69f;--card:#1b211e}}
:root[data-theme="dark"]{--bg:#121614;--ink:#e8ede9;--muted:#9aa69f;--card:#1b211e}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:20px 16px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 14px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.tabs button,.ctl button{border:1px solid #c9cfca;background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--felt);color:#fff;border-color:var(--felt)}
canvas{width:100%;height:auto;border-radius:10px;display:block;background:var(--felt)}
.ctl{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
input[type=range]{flex:1;min-width:160px}
.info{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;margin-top:6px;flex-wrap:wrap;gap:8px}
.legend span{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
.note{background:var(--card);border-radius:10px;padding:10px 14px;margin-top:14px;color:var(--muted);font-size:13.5px}
</style></head><body><div class="wrap">
<h1>Inside the simulator</h1>
<p class="sub">The underdog (red) attacks right; the stronger team (blue) attacks left. Each dot is a player, and each run shows the first __MIN__ minutes of a simulated match.</p>
<div class="tabs" id="tabs"></div>
<canvas id="c" width="1050" height="680"></canvas>
<div class="ctl"><button id="play">Pause</button><button id="spd">Speed 2×</button><input id="t" type="range" min="0" value="0"></div>
<div class="info"><span id="clock"></span><span class="legend"><span style="background:var(--red)"></span>underdog<span style="background:var(--blue)"></span>strong team<span style="background:#fff"></span>ball</span><span id="score"></span></div>
<div class="note" id="desc"></div>
<div class="note">This is a research simulator, not a game engine. Keepers are implicit, and players move toward target spots chosen by each team's tactic. Results from it count only if they survive a realism check against real match data.</div>
</div>
<script>
const S=__DATA__;const names=Object.keys(S);let cur=names[0],f=0,playing=true,speed=2;
const c=document.getElementById('c'),x=c.getContext('2d'),T=document.getElementById('t');
const sx=c.width/105,sy=c.height/68;
function pitch(){x.fillStyle='#1e4a38';x.fillRect(0,0,c.width,c.height);for(let i=0;i<14;i+=2){x.fillStyle='#21503d';x.fillRect(i*7.5*sx,0,7.5*sx,c.height)}
x.strokeStyle='rgba(215,230,219,.9)';x.lineWidth=2;x.strokeRect(2,2,c.width-4,c.height-4);x.beginPath();x.moveTo(c.width/2,0);x.lineTo(c.width/2,c.height);x.stroke();
x.beginPath();x.arc(c.width/2,c.height/2,9.15*sx,0,7);x.stroke();
for(const g of [0,1]){const gx=g?105-16.5:0;x.strokeRect(gx*sx,(34-20.16)*sy,16.5*sx,40.32*sy);const gx2=g?105-5.5:0;x.strokeRect(gx2*sx,(34-9.16)*sy,5.5*sx,18.32*sy)}}
function ballAt(fr,i){const F=fr[i];if(F[42]<10)return[F[40],F[41]];let j=i;while(j<fr.length&&fr[j][42]>=10)j++;let k=i;while(k>0&&fr[k][42]>=10)k--;
 if(j>=fr.length)return[F[40],F[41]];const a=fr[k],b=fr[j],u=(i-k)/Math.max(1,j-k);return[a[40]+(b[40]-a[40])*u,a[41]+(b[41]-a[41])*u]}
function draw(){const sc=S[cur],F=sc.frames[f];pitch();
for(const tm of [0,1]){x.fillStyle=tm?'#3b6fc0':'#d9553a';for(let i=0;i<10;i++){const px=F[tm*20+i*2]*sx,py=(68-F[tm*20+i*2+1])*sy;x.beginPath();x.arc(px,py,9,0,7);x.fill();x.strokeStyle='#fff';x.lineWidth=1.5;x.stroke()}}
const b=ballAt(sc.frames,f);x.fillStyle='#fff';x.beginPath();x.arc(b[0]*sx,(68-b[1])*sy,5.5,0,7);x.fill();x.strokeStyle='#17201b';x.lineWidth=1;x.stroke();
const s=Math.floor(f*0.2);document.getElementById('clock').textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
const sc2=sc.score[f];document.getElementById('score').textContent=`Underdog ${sc2[0]} – ${sc2[1]} Strong`;T.value=f}
function setScen(n){cur=n;f=0;T.max=S[n].frames.length-1;document.getElementById('desc').textContent=n+': '+S[n].desc;
 [...document.querySelectorAll('#tabs button')].forEach(b=>b.classList.toggle('on',b.textContent===n));draw()}
names.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>setScen(n);document.getElementById('tabs').appendChild(b)});
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
document.getElementById('spd').onclick=e=>{speed=speed===1?2:speed===2?4:1;e.target.textContent='Speed '+speed+'×'};
T.oninput=()=>{f=+T.value;draw()};
let last=0;function loop(ts){if(playing&&ts-last>200/speed){f=(f+1)%S[cur].frames.length;draw();last=ts}requestAnimationFrame(loop)}
setScen(names[0]);requestAnimationFrame(loop);
</script></body></html>"""
html = html.replace("__DATA__", json.dumps(scen, separators=(",", ":"))).replace("__MIN__", str(MIN))
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(html)
print("wrote", out, round(len(html) / 1e6, 2), "MB")
