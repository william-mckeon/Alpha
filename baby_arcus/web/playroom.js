import {drawArcus} from "/arcus-renderer.js";
import {drawEnvironment,floor,point} from "/environment-renderer.js";
import {initDesktop,desktopAvailable,pickup,returnToPen} from "/desktop-bridge.js";
import {initConversation} from "/conversation.js";
const $ = id => document.getElementById(id);
const canvas = $("room"), ctx = canvas.getContext("2d");
const sprite = new Image(); sprite.src = "/arcus-body.png";
let state = null, sending = 0, connected = false;
let observedConsequence = null, consequenceGeneration = null;
let commandQueue = Promise.resolve(), revision = 0;
function round(x,y,w,h,r,color,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.stroke();}}
function text(t,x,y,size=14,color="#697565",align="left"){ctx.font=`${size}px Segoe UI, sans-serif`;ctx.fillStyle=color;ctx.textAlign=align;ctx.fillText(t,x,y);}
function ellipse(x,y,rx,ry,color){ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
function draw(){
 drawEnvironment(ctx,state?.environment);
 if(!state)return;
 const env=state.environment, body=state.arcus, placement=env.placements[body.entity_id];
 const h=env.human,[hx,hy]=point(h.x,h.y,env);
 if(h.present!==false){ellipse(hx,hy+7,23,9,"#b7ac9566");ellipse(hx,hy,18,18,"#e36c3c");
 text(h.name==="You"?"Y":"W",hx,hy+5,14,"#ffffff","center");
 text(h.name,hx,hy+36,12,"#68775e","center");}
 if(placement && !state.view.held && state.view.region === "playpen"){
   const [x,y]=point(placement.x,placement.y,env);drawArcus(ctx,sprite,body,{x,y});
 }
 if(state.cue){round(355,581,390,38,19,"#f5f7edee");text(state.cue.from+": "+state.cue.kind.replaceAll("_"," "),550,605,13,"#6f8266","center");}
 if(state.paused){round(405,228,290,35,17,"#fcf8e7ee");text("SESSION PAUSED",550,251,12,"#9a8960","center");}
}
async function request(path,body){const response=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(data.error||`HTTP ${response.status}`);return data;}
function update(s){
 s={...state,...s};state=s;connected=true;
 const language=s.language||{status:'unavailable'};
 const visual=s.visual||{status:'unavailable'};
 const shared=s.shared||{status:'unavailable'};
 const measured=shared.qualification?.measurements;
 if(consequenceGeneration!==shared.generation){observedConsequence=null;consequenceGeneration=shared.generation;}
 if(shared.last_consequence?.forecast)observedConsequence=shared.last_consequence;
 const consequence=observedConsequence;
 const predicted=consequence?.forecast?.future_body,observed=consequence?.observed_body;
 const predictionError=predicted?.length===20&&observed?.length===20&&consequence.forecast.horizon_ticks===consequence.horizon_ticks?predicted.reduce((sum,value,i)=>sum+(value-observed[i])**2,0)/20:null;
 const exploration=shared.last_prediction?.experimentation?.selected;
 const continuity=shared.last_prediction?.continuity;
 if(continuity){
  const tracks=continuity.objects||[],visible=tracks.filter(track=>track.visible);
  const associations=visible.filter(track=>track.identity_status==='learned_association');
  const plan=continuity.plan||{};
  $('continuity-status').textContent=`${shared.enabled?'Current':'Last recorded'} object memory: ${visible.length} visible, ${tracks.length} remembered; ${associations.length} learned associations. ${continuity.survey_views!==undefined?`Prior visual survey: ${continuity.survey_views}/9 views; identity remains unknown. `:''}${plan.status==='reobserve_after_step'?`Looking for ${plan.target}; ${plan.remaining} steps left, then reassess.`:plan.reason||plan.status||'No search selected'} ${visible.length?`Visible references: ${visible.slice(0,8).map(track=>`${track.id} (${track.identity_status==='learned_association'?`match score ${track.association_confidence.toFixed(2)}`:'identity unconfirmed'})`).join(', ')}.`:''}`;
 }else $('continuity-status').textContent='Waiting for shared object-memory observations.';
 const comparison=predictionError!==null?`Last captured body prediction error: ${predictionError.toFixed(5)}`:consequence?`Forecast at ${consequence.forecast.horizon_ticks} ticks; observed at ${consequence.horizon_ticks} ticks (different intervals, error not scored)`:'Waiting for a measured consequence';
 $('shared-learning-detail').textContent=`Depth budget: ${Number.isFinite(shared.depth_capacity)?`${Math.round(100*shared.depth_capacity)}%`:'unverified'} · recalled experiences: ${shared.recalled_memories||0} · action sequences recorded: ${shared.sequence_outcomes||0} · ${comparison} · ${exploration?`Exploring a predicted new view; uncertainty ${exploration.uncertainty.toFixed(5)}`:'No exploration action selected'}. Training receipts and active checkpoint updates remain separate from observation.`;
 const score=value=>Number.isFinite(value)?`${(100*value).toFixed(1)}%`:'unmeasured';
 $('shared-status').textContent=`${shared.status} · observations: ${shared.observations||0} · generation: ${shared.generation||'none'} · candidate updates: ${shared.candidate?.updates??'unknown'} · qualified updates: ${shared.qualification?.qualified?(shared.qualification.candidate?.updates??'unknown'):'none'} · replay: ${shared.replay?.training||0} training / ${shared.replay?.evaluation||0} held out · delayed outcomes: ${shared.delayed_outcomes||0} · gates: ${['integration','retention','cross_modal','live','depth','curiosity','continuity'].map(k=>`${k} ${shared.qualification?.[k]===true?'passed':'not qualified'}`).join(', ')}${measured?` · held-out commands ${score(measured.commands)} / color references ${score(measured.color_reference)} / rest ${score(measured.rest)}`:''} · ${shared.reason||(shared.qualification?.qualified?'Qualified shared learner ready':'Shared candidate not promoted')}`;
 const perception=shared.enabled?shared.last_prediction?.visual_observation:visual.observation;
 const perceptionFrame=shared.enabled?shared.frame:visual.frame;
 $('color-status').textContent=perception?`${perception.objects.length} predicted visible regions · ${shared.enabled?shared.status:visual.status} · ${continuity?'identity associations and uncertainty shown in object memory':'no persistent identity observation'}`:'No learned observation yet.';
 if(perception && perceptionFrame?.image_base64){
  const frameId=perceptionFrame.sha256;
  if($('color-view').dataset.frame!==frameId){
   $('color-view').dataset.frame=frameId;const image=new Image();
   image.onload=()=>{if($('color-view').dataset.frame!==frameId)return;
    const c=$('color-view').getContext('2d');c.drawImage(image,0,0,400,280);c.strokeStyle='#ffffff';c.lineWidth=2;
    for(const obj of perception.objects){const [x,y,r,b]=obj.bbox;c.strokeRect(x/96*400,y/96*280,(r-x)/96*400,(b-y)/96*280);}
   };image.src='data:image/png;base64,'+perceptionFrame.image_base64;
  }
 }
 const rest=s.rest||{status:'unavailable'};
 const curiosity=s.curiosity||{status:'unavailable'};
 $('curiosity-status').textContent=`${curiosity.status} · discoveries: ${curiosity.discoveries||0} · ${curiosity.reason||''}`;
 $('rest-status').textContent=`${rest.status} · ${rest.last_choice||'no choice yet'} · ${rest.reason||''}`;
 $('rest-senses').textContent=`Posture: ${s.arcus.posture} · sleep: ${s.arcus.sleep_state} · eyes: ${s.arcus.eyelid_openness>0?'open':'closed'} · simulated rest need: ${Math.round(100*(s.arcus.rest_need||0))}% · alertness: ${Math.round(100*(s.arcus.alertness||0))}%`;
 $('visual-status').textContent=`${visual.status} · decisions: ${visual.decisions||0} · choice: ${visual.last_action||'none'} · ${visual.reason||''}`;
 $('visual-cost').textContent=visual.resources?`Inference: ${visual.resources.inference_ms.toFixed(1)} ms · tokens: ${visual.resources.input_tokens} · expert-routed tokens: ${(100*visual.resources.expert_routed_fraction).toFixed(1)}% · ${visual.resources.learned_budget?'Learned':'Fixed'} capacity: ${visual.resources.capacity} · decisions left: ${visual.resources.remaining_decisions??'—'}`:'';
 const listeningLabel=!language.enabled?'dataset held':language.listening?'dataset playing':'dataset paused';
 const hearingLabels=shared.enabled?{start:'Ask Arcus to listen',stop:'Stop shared mode',pause:'Ask him to pause listening',resume:'Ask him to resume listening',restart:'Ask him to restart listening'}:{start:'Enable language learning',stop:'Stop language learning',pause:'Pause the dataset',resume:'Resume the dataset',restart:'Restart the dataset'};
 for(const button of document.querySelectorAll('[data-language]')){button.textContent=hearingLabels[button.dataset.language]||button.textContent;button.disabled=!!(shared.shared_only&&!shared.enabled);}
 $('language-status').textContent=`${language.status} · ${listeningLabel} · last choice: ${language.last_choice||'none'}${language.reason?' · '+language.reason:''}`;
 $('language-counts').textContent=`Text input tokens: ${language.exposed_tokens||0} · accepted training targets: ${language.training_tokens||0} · remaining: ${language.remaining_training_tokens??4096} · accepted updates: ${language.accepted_updates||0} · declined updates: ${language.rejected_updates||0}`;
 if(shared.enabled){
  $('language-status').textContent=`Shared hearing · ${shared.hearing_cursor?.playing?'dataset playing':'dataset held'} · last choice: ${shared.last_prediction?.hearing_action||'none'}`;
  $('language-counts').textContent=`Dataset tokens heard: ${shared.hearing_cursor?.exposures||0} · Fresh words enter replay for candidate training.`;
 }
 const history=$("interaction-history");history.replaceChildren();
 for(const event of (s.interactions||[]).slice(-6).reverse()){
  const row=document.createElement("li");row.textContent=`${event.sequence} · ${event.payload?.hearing?'simulated hearing: '+event.payload.hearing.utterance:event.kind} · ${event.status}${event.reason?' · '+event.reason:''}`;history.append(row);
 }
 $("interaction-status").textContent="Model receiver: "+(s.policy?.receiver||"unavailable")+" · Delivery means exposure, not understanding.";
 $("call").disabled=(!shared.enabled&&!visual.navigation_ready&&!s.policy?.available_goals?.includes("approach"))||s.paused||s.view.held||s.view.region!=="playpen"||s.arcus.sleep_state!=="awake";
 const navigateButton=document.querySelector('[data-visual="navigate"]');
 navigateButton.disabled=shared.enabled||!visual.navigation_ready||visual.enabled||s.paused||s.view.held||s.view.region!=='playpen'||s.arcus.sleep_state!=='awake';
 for(const button of document.querySelectorAll('[data-rest], [data-curiosity], [data-visual]:not([data-visual="navigate"])'))button.disabled=!!(shared.enabled||shared.shared_only);
 navigateButton.title=visual.navigation_ready?'Stand using learned joint movements if needed, then approach a nearby visible marker':'Navigation qualification is not complete';

 if(s.policy){
  const p=s.policy,active=["loading","running"].includes(p.status);
  $("policy-status").textContent=`${p.status} · goal: ${p.goal||"standing"} · ${p.reason} · ${p.actions||0} actions · ${p.hold_seconds||0}/5 seconds balanced`;
  $("policy-checkpoint").textContent=`${p.policy||"Standing policy"} · ${p.checkpoint||""} · SHA-256 ${p.checkpoint_sha256||"available after start"}`;
  $("policy-start").disabled=active||p.status==="unavailable"||s.paused||s.view.held||s.view.region!=="playpen"||s.arcus.sleep_state!=="awake";
  $("policy-lie").disabled=$("policy-start").disabled||!p.available_goals?.includes("lying");
  $("policy-sit").disabled=$("policy-start").disabled||!p.available_goals?.includes("sitting");
  $("policy-stop").disabled=!active;
  document.querySelector(".notice").textContent=active?"Learned "+p.goal+" · "+p.status:"Human control · learned movement "+p.status;
 }
 if(visual.enabled)document.querySelector('.notice').textContent=`Learned ${visual.lesson||'visual'} choices · ${visual.status}`;
 if(shared.enabled){
  const unavailable=s.paused||s.view.held||s.view.region!=='playpen'||s.arcus.sleep_state!=='awake';
  for(const id of ['policy-start','policy-lie','policy-sit'])$(id).disabled=unavailable;
  $('policy-stop').disabled=false;
  $('interaction-status').textContent=`Shared model: ${shared.status} · Requests arrive through simulated hearing.`;
  $('policy-status').textContent=`Shared movement · ${shared.last_prediction?.intent?.goal||'waiting for a choice'}`;
  document.querySelector('.notice').textContent=`Arcus is choosing his actions · ${shared.status}`;
 }
 if(shared.shared_only&&!shared.enabled){
  for(const id of ['policy-start','policy-lie','policy-sit','call'])$(id).disabled=true;
  for(const button of document.querySelectorAll('[data-language], [data-visual]'))button.disabled=true;
  $('interaction-status').textContent='Start the qualified shared model to deliver requests at the fixed 25% depth budget.';
 }
 if(s.audit)$("audit-status").textContent=s.audit.healthy?"Activity logging active · "+s.audit.events+" events":s.audit.error?"Activity logging failed: "+s.audit.error+" · "+s.audit.dropped_events+" events missed":"Activity logging is not enabled on this host";
 const env=s.environment,a=s.arcus,p=env.placements[a.entity_id];
 $("connection").textContent="Session connected";$("connection-dot").className="online";
 $("posture").textContent=(a.sleep_state!=="awake"?a.sleep_state:a.posture).replaceAll("_"," ");$("result").textContent=s.last_result;
 $("pause").textContent=s.paused?"Resume session":"Pause session";
 $("person").value=env.human.name;
 $("body-status").textContent="Arcus "+a.entity_id+" · position "+p.x.toFixed(2)+", "+p.y.toFixed(2);
 $("environment-status").textContent="Play area "+env.environment_id+" · reset "+env.generation;
 $("view-status").textContent="Arcus sees: "+s.view.source+" · "+s.view.event.replaceAll("_"," ")+" · mouse pointer excluded";
 $("stand").disabled=s.view.held||a.sleep_state!=="awake";$("lie").disabled=s.view.held||a.sleep_state!=="awake";
 $("sleep").disabled=a.sleep_state==="sleeping";$("wake").disabled=a.sleep_state!=="sleeping";
 if(a.sleep_state!=="awake")$("view-status").textContent="Visual observations off · "+a.sleep_state.replaceAll("_"," ");
 if(a.eyelid_openness===0)$("view-status").textContent="Eyes closed · body sensations remain available when awake";
 $("senses-status").textContent="Height "+a.height.toFixed(2)+" · "+(s.senses?.stable?"balanced":"not balanced")+" · eyes "+(a.eyelid_openness>0?"open":"closed");
 $("telemetry").textContent=JSON.stringify({arcus:a,environment:env,cue:s.cue},null,2);
 document.querySelectorAll("[data-direction]").forEach(b=>b.disabled=a.sleep_state!=="awake"||s.view.held||s.view.region!=="playpen"||s.paused||a.height<.99||(a.motor_mode==="assisted"&&a.target_posture!=="standing"));draw();
}
function act(action){
 sending++; revision++;
 const command={request_id:crypto.randomUUID(),action};
 commandQueue=commandQueue.then(async()=>{
  try{
   const r=await request("/api/action",command);update(r.state);$("error").textContent="";
   const li=document.createElement("li");li.textContent=`${r.event.tick} · ${r.event.source} · ${r.event.result}`;$("events").prepend(li);
   while($("events").children.length>30)$("events").lastChild.remove();
  }catch(e){$("error").textContent=e.message;}finally{sending--;}
 });
 return commandQueue;
}

$('color-setup').onclick=()=>act({kind:'color_lesson',colors:{floor:$('color-floor').value,wall:$('color-wall').value,rug:$('color-rug').value},balls:[$('color-ball1').value,$('color-ball2').value]});
document.querySelectorAll('[data-shared]').forEach(button=>button.onclick=async()=>{
 try{await request('/api/shared/control',{request_id:crypto.randomUUID(),action:button.dataset.shared});update(await request('/api/state'));$('error').textContent='';}
 catch(e){$('error').textContent=e.message;}
});
$("stand").onclick=()=>act({kind:"stand"});$("lie").onclick=()=>act({kind:"lie"});
for(const [id,operation,goal] of [["start","start","standing"],["lie","start","lying"],["sit","start","sitting"],["stop","stop",null]])$("policy-"+id).onclick=async()=>{
 sending++;revision++;
 try{await commandQueue;await request("/api/policy/"+operation,{request_id:crypto.randomUUID(),...(goal?{goal}:{})});update(await request("/api/state"));$("error").textContent="";}
 catch(e){$("error").textContent=e.message;}finally{sending--;}
};
$("sleep").onclick=()=>act({kind:"sleep"});$("wake").onclick=()=>act({kind:"wake_up"});
$("eyes-open").onclick=()=>act({kind:"eyelids",openness:1});$("eyes-close").onclick=()=>act({kind:"eyelids",openness:0});
for(const leg of ["front_left","front_right","rear_left","rear_right"])for(const joint of ["hip","knee","ankle"]){
 const option=document.createElement("option");option.value=leg+"."+joint;option.textContent=(leg+" "+joint).replaceAll("_"," ");$("motor-joint").append(option);
}
$("motor-less").onclick=()=>act({kind:"joint",joint:$("motor-joint").value,delta:-.1});
$("motor-more").onclick=()=>act({kind:"joint",joint:$("motor-joint").value,delta:.1});
document.querySelectorAll("[data-look]").forEach(button=>button.onclick=()=>{const [yaw,pitch]=button.dataset.look.split(",").map(Number);act({kind:$("look-part").value,yaw,pitch});});
initConversation();
document.querySelectorAll('[data-curiosity]').forEach(button=>button.onclick=async()=>{
 try{await request('/api/curiosity/control',{request_id:crypto.randomUUID(),action:button.dataset.curiosity});update(await request('/api/state'));$('error').textContent='';}
 catch(e){$('error').textContent=e.message;}
});
document.querySelectorAll('[data-rest]').forEach(button=>button.onclick=async()=>{
 try{await request('/api/rest/control',{request_id:crypto.randomUUID(),action:button.dataset.rest});update(await request('/api/state'));$('error').textContent='';}
 catch(e){$('error').textContent=e.message;}
});
$("pause").onclick=()=>state&&act({kind:"pause",value:!state.paused});
$("reset").onclick=()=>act({kind:"reset"});
$("return").onclick=()=>desktopAvailable()?returnToPen():act({kind:"return"});
$("person").onchange=()=>state&&act({kind:"human",x:state.environment.human.x,y:state.environment.human.y,name:$("person").value});
$("call").onclick=()=>act({kind:"call"});$("encourage").onclick=()=>act({kind:"feedback",value:"encourage"});
document.querySelectorAll("[data-direction]").forEach(b=>b.onclick=()=>act({kind:"move",direction:b.dataset.direction}));
canvas.addEventListener("keydown",e=>{const d={ArrowUp:"up",ArrowDown:"down",ArrowLeft:"left",ArrowRight:"right"}[e.key];if(d){e.preventDefault();act({kind:"move",direction:d});}});
let pickedUp=false;
canvas.addEventListener("pointerdown",e=>{
 pickedUp=false;
 if(e.button!==0 || !state || !desktopAvailable() || state.view.held || state.view.region!=="playpen")return;
 const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1100,y=(e.clientY-r.top)/r.height*740;
 const p=state.environment.placements[state.arcus.entity_id];
 const [ax,ay]=point(p.x,p.y,state.environment);
 const pose=state.arcus.visual_pose||{width:160,height:129};
 if(Math.abs(x-ax)<pose.width/2 && y>ay-pose.height+8 && y<ay+8){
   pickedUp=true;e.preventDefault();pickup();
 }
});
canvas.addEventListener("click",e=>{
 if(pickedUp){pickedUp=false;return;}
 if(!connected)return;canvas.focus();
 const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*1100,py=(e.clientY-r.top)/r.height*740;
 const env=state.environment,x=(px-floor.x)/floor.w*env.width,y=(py-floor.y)/floor.h*env.height;
 if(x<.5||x>env.width-.5||y<.5||y>env.height-.5)return;
 act({kind:"human",x,y,name:$("person").value});
});
$("export").onclick=async()=>{try{const data=await request("/api/session");const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`arcus-playroom-${data.session}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){$("error").textContent=e.message;}};
async function poll(){try{if(!sending){const started=revision;const next=await request("/api/state");if(!sending&&revision===started)update(next);}}catch(e){connected=false;$("connection").textContent="Room disconnected";$("connection-dot").className="";$("error").textContent=e.message;}finally{setTimeout(poll,150);}}
document.querySelectorAll('[data-visual]').forEach(button=>button.addEventListener('click',async()=>{
 try{await request('/api/visual/control',{request_id:crypto.randomUUID(),action:button.dataset.visual});$('error').textContent='';}
 catch(e){$('error').textContent=e.message;}
}));
document.querySelectorAll('[data-language]').forEach(button=>button.addEventListener('click',async()=>{
 try{await request('/api/language/control',{request_id:crypto.randomUUID(),action:button.dataset.language});$('error').textContent='';}
 catch(error){$('error').textContent=error.message;}
}));
sprite.onload=draw;sprite.onerror=()=>{$("error").textContent="Arcus artwork could not be loaded.";};initDesktop(canvas);draw();poll();




