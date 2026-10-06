// Parity check: runs the site's own prediction functions (copied from app/index.html) against Python outputs.
function interpolate(x,a,b){if(x<a[0])return b[0];if(x>=a[a.length-1])return b[b.length-1];let i=0;while(i<a.length-1&&a[i+1]<=x)i++;return b[i]+(b[i+1]-b[i])*(x-a[i])/(a[i+1]-a[i]);}
function xgbPredict(x,m){x=x.map(Math.fround);let y=Math.fround(m.base);for(const t of m.trees){let n=0;while(t.left_children[n]!==-1){const v=x[t.split_indices[n]],left=Number.isNaN(v)?t.default_left[n]:v<Math.fround(t.split_conditions[n]);n=left?t.left_children[n]:t.right_children[n];}y=Math.fround(y+Math.fround(t.split_conditions[n]));}return y;}
function ensembleAmount(raw,rich,incumbent,region,lead){const m=ENSEMBLE.models.find(m=>m.lead===lead);if(raw===0)return {amount:0,zeroGate:true,qmxgb:0,incumbent,weight:m.weight};let map=m.qm[region]||m.qm.global,q=interpolate(raw,map.source,map.target),x=rich.concat([raw,q,Math.log1p(raw),Math.log1p(q)]),a=Math.max(0,q+xgbPredict(x,m)),flag=false;if(m.clf){const pr=1/(1+Math.exp(-xgbPredict(x,m.clf)));if(pr>=m.clf.threshold){flag=true;a=Math.max(a,64.5)}}return {amount:m.weight*a+(1-m.weight)*incumbent,zeroGate:false,qmxgb:a,incumbent,weight:m.weight,heavyFlag:flag};}
const fs=require('fs');const M=JSON.parse(fs.readFileSync(process.argv[2]));const ENSEMBLE={models:[M]};
const P=JSON.parse(fs.readFileSync(process.argv[3]));let mx=0;
for(const r of P){const o=ensembleAmount(r.raw,r.rich,0,r.region,1);const d=Math.abs(o.amount-r.py);if(d>mx)mx=d}
console.log('rows',P.length,'max_abs_diff',mx)
