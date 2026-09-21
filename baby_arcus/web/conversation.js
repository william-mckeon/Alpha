export function initConversation(){
 const form=document.getElementById("conversation-form"), history=document.getElementById("conversation-history");
 let pending=null,busy=false;
 async function refresh(){
  const response=await fetch("/api/messages"); if(!response.ok)throw Error("Message history unavailable");
  const data=await response.json(); history.replaceChildren();
  for(const message of data.messages){
   const row=document.createElement("li");
   const receipt=message.language_receipt;
   const status=receipt?`${receipt.status.replaceAll('_',' ')} · ${receipt.trained_tokens} training targets`:(message.model_read?"Delivered to model; candidate training is separate":message.status==="queued"?"Queued until awake":"Waiting for model delivery");
   row.textContent=message.sender_name+": "+message.text+" — "+status;
   history.append(row);
  }
  const expressions=document.getElementById('arcus-expressions');expressions.replaceChildren();
  for(const expression of data.expressions||[]){const row=document.createElement('li');row.textContent='Arcus: '+expression.text+' — early model-generated text';expressions.append(row);}
 }
 form.addEventListener("submit",async event=>{
  event.preventDefault();if(busy)return;busy=true;
  const text=document.getElementById("message-text"),sender=document.getElementById("message-sender");
  const status=document.getElementById("message-status");
  const payload={sender:sender.value,text:text.value};
  if(!pending||pending.text!==payload.text||pending.sender!==payload.sender)pending={request_id:crypto.randomUUID(),...payload};
  try{
   const response=await fetch("/api/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(pending)});
   const data=await response.json();if(!response.ok)throw Error(data.error||"Message not sent");
   text.value="";pending=null;status.textContent="Message saved for model delivery. Exposure and completed training are separate; an immediate reply is not guaranteed.";
   await refresh();
  }catch(error){status.textContent=error.message;}finally{busy=false;}
 });
 setInterval(()=>refresh().catch(()=>{}),1500);refresh().catch(()=>{});
}

