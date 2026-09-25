import {drawArcus} from '/arcus-renderer.js';
import {drawEnvironment,point} from '/environment-renderer.js';
const $=id=>document.getElementById(id), ctx=$('room').getContext('2d');
const sprite=new Image();sprite.src='/arcus-body.png';
let busy=false;
async function refreshMemory(){try{const response=await fetch('/api/test2/memory');if(!response.ok)throw Error(String(response.status));const data=await response.json();$('memory-status').textContent=data.busy?'Learner busy; memory sample deferred.':JSON.stringify(data,null,2);}catch(error){$('memory-status').textContent=`Memory unavailable: ${error.message}`;}}
setInterval(refreshMemory,10000);refreshMemory();
function human(w){const h=w.environment.human;if(h.present===false)return;const [x,y]=point(h.x,h.y,w.environment);ctx.beginPath();ctx.arc(x,y,18,0,2*Math.PI);ctx.fillStyle='#e36c3c';ctx.fill();ctx.font='14px Segoe UI';ctx.textAlign='center';ctx.fillStyle='#fff';ctx.fillText(h.name==='You'?'Y':'W',x,y+5);ctx.fillStyle='#405440';ctx.fillText(h.name,x,y+36);}
async function request(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await r.json();if(!r.ok)throw Error(data.error||String(r.status));if(path==='/api/test2')renderIdle(data);return data;}
function renderIdle(s){
  $('model-name').textContent=s.model_name||'Baby Arcus · Test 2'; document.title=$('model-name').textContent;
  const practice=s.practice||{state:'idle'};
  $('practice-status').textContent=JSON.stringify(practice);
  $('practice').disabled=practice.state==='running';
  const stages=s.three_stage;
  $('three-stage-panel').hidden=!stages;
  if(stages){
    $('three-stage-status').textContent=`${stages.enabled?'Configured for reviewed mixed learning':'Training disabled pending data review and agreed budget'} · ${stages.approved_batch_count} selected approved batches · eligible sources: ${stages.eligible_languages.join(', ')}`;
    $('three-stage-progress').textContent=JSON.stringify(stages.progress?.state||{additional_target_tokens:0,token_budget:stages.token_budget},null,2);
  }
  const idle=s.idle_learning; $('idle-panel').hidden=!idle;
  if(!idle)return;
  $('idle-resume').disabled=!!stages&&!stages.enabled;
  for(const id of ['learn','train','hear'])$(id).hidden=true;
  $('start').textContent='Explore · 10 actions';
  const active=stages?'Training reviewed three-stage mixture':'Training original mixed curriculum';
  $('idle-status').textContent=`${idle.error?'Stopped: '+idle.error:idle.training?active:idle.enabled?'Waiting for quiet time':'Quiet-time training paused'} · ${idle.updates_this_session}/${idle.config.session_updates} session updates · ${idle.idle_seconds_elapsed}s since interaction`;
}
$('idle-resume').onclick=()=>action('idle',{action:'resume'});
$('idle-pause').onclick=()=>request('/api/test2/idle',{action:'pause'}).then(refresh).catch(e=>{$('result').textContent=e.message;});
async function refresh(){const s=await request('/api/test2');const w=s.world; $('capacity').textContent=s.depth_capacity;drawEnvironment(ctx,w.environment);human(w);$('summary').textContent=`${s.state} · ${s.candidate.updates} learning updates · ${s.queued} queued experiences · ${s.remaining_cycles} exploration cycles remaining${s.error?' · '+s.error:''}`;const p=w.environment.placements[w.arcus.entity_id];const [x,y]=point(p.x,p.y,w.environment);drawArcus(ctx,sprite,w.arcus,{x,y});$('status').textContent=JSON.stringify({runtime:s.runtime,practice:s.practice,mode:s.mode,state:s.state,capacity:s.depth_capacity,generation:s.candidate.generation,updates:s.candidate.updates,queued:s.queued,learned:s.learned_experiences,dataset:s.dataset_available,last:s.last,error:s.error},null,2);$('messages').replaceChildren(...s.messages.map(m=>{const li=document.createElement('li');li.textContent=`${m.sender}: ${m.text} — ${m.observed?'observed; training tracked separately':'queued for hearing'}`;return li;}));$('expression').textContent=s.last?.decision?.expression?`Arcus: ${s.last.decision.expression}`:'';}
async function action(name,body={}){const priority=name==='message';if(busy&&!priority)return false;if(!priority)busy=true;$('result').textContent='Working…';try{const result=await request(`/api/test2/${name}`,{request_id:crypto.randomUUID(),...body});$('result').textContent=priority?'Queued for simulated hearing. Arcus will choose his response.':result.reason||result.outcome?.result||`Completed ${name}${result.updates_this_job!==undefined?`; ${result.updates_this_job} learning updates`:''}.`;await refresh();return true;}catch(e){$('result').textContent=e.message;return false;}finally{if(!priority)busy=false;}}
for(const id of ['step','hear','learn','train'])$(id).onclick=()=>action(id);
$('start').onclick=()=>action('control',{action:'start',cycles:10});
$('pause').onclick=()=>request('/api/test2/control',{action:'pause'}).then(refresh).catch(e=>{$('result').textContent=e.message;});
$('message').onsubmit=async e=>{e.preventDefault();if(await action('message',{sender:$('sender').value,text:$('text').value}))$('text').value='';};
$('call').onclick=()=>action('message',{sender:$('sender').value,text:'Come here, Arcus'});
$('objects').onclick=()=>action('action',{action:{kind:'color_lesson',colors:{floor:'#e2d2b8',rug:'#c5d1b4',wall:'#7d9472'},balls:['#ed3342','#3366ed']}});
$('room').onclick=async e=>{const s=await request('/api/test2');const rect=$('room').getBoundingClientRect();const px=(e.clientX-rect.left)*1100/rect.width,py=(e.clientY-rect.top)*740/rect.height;if(px<100||px>1000||py<170||py>660)return;await action('action',{action:{kind:'human',x:Math.max(.5,Math.min(s.world.environment.width-.5,(px-100)/900*s.world.environment.width)),y:Math.max(.5,Math.min(s.world.environment.height-.5,(py-170)/490*s.world.environment.height)),name:$('sender').value==='wife'?'Your wife':'You'}});};
setInterval(()=>refresh().catch(e=>{$('result').textContent=e.message;}),2000);refresh().catch(e=>{$('result').textContent=e.message;});



$('practice').onclick=()=>action('practice',{task:'positive_sum'});
$('practice-stop').onclick=()=>request('/api/test2/practice-stop',{}).then(refresh).catch(e=>{$('result').textContent=e.message;});
