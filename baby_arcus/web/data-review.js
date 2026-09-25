let loaded = null;
let queueOffset = 0;
const pageSize = 20;
const el = id => document.getElementById(id);
async function request(path, extra) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({credential:el('credential').value, ...extra})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function disable() { loaded=null; el('approve').disabled=true; el('reject').disabled=true; el('revoke').disabled=true; }
el('batch').oninput = disable;
el('credential').oninput = disable;
el('load').onclick = async () => {
  disable();
  try {
    const batch = await request('/batch', {batch_id:el('batch').value.trim()});
    loaded=batch.id; el('revoke').disabled=batch.decision!=='approved'||!!batch.revoked; el('record').textContent=JSON.stringify(batch,null,2);
    el('approve').disabled=!!batch.decision; el('reject').disabled=!!batch.decision;
    el('status').textContent=batch.fixture ? 'TEST FIXTURE — this cannot enter real training.' : 'Review all examples and provenance before approving.';
  } catch (error) { el('status').textContent=error.message; }
};
for (const [id,decision] of [['approve','approved'],['reject','rejected']]) {
  el(id).onclick=async () => {
    if (!loaded) return;
    const batch_id=loaded; disable();
    try {
      const result=await request('/review',{batch_id,decision,reviewer:el('reviewer').value});
      el('record').textContent=JSON.stringify(result,null,2);
      el('status').textContent='Decision recorded. Training remains separately controlled.';
    } catch (error) { el('status').textContent=error.message; }
  };
}

async function loadQueue() {
  try {
    const result = await request('/queue', {offset:queueOffset, limit:pageSize});
    el('queue-items').replaceChildren();
    el('queue-page').textContent=`Page ${Math.floor(queueOffset/pageSize)+1}`;
    el('queue-prev').disabled=queueOffset===0;
    el('queue-next').disabled=result.batches.length<pageSize;
    for (const item of result.batches) {
      const button = document.createElement('button');
      button.textContent = (item.decision || 'Pending') + ' - ' + item.id;
      button.onclick = () => { el('batch').value=item.id; disable(); el('load').click(); };
      el('queue-items').append(button);
    }
  } catch (error) { el('status').textContent=error.message; }
};

el('revoke').onclick=async () => {
  if (!loaded) return;
  const batch_id=loaded; disable();
  try {
    const result=await request('/revoke',{batch_id,reviewer:el('reviewer').value,reason:el('revoke-reason').value});
    el('record').textContent=JSON.stringify(result,null,2);
    el('status').textContent='Excluded from future training jobs. Existing learned weights are unchanged.';
  } catch(error) { el('status').textContent=error.message; }
};

el('queue').onclick=()=>{queueOffset=0;loadQueue();};
el('queue-prev').onclick=()=>{queueOffset=Math.max(0,queueOffset-pageSize);loadQueue();};
el('queue-next').onclick=()=>{queueOffset+=pageSize;loadQueue();};
