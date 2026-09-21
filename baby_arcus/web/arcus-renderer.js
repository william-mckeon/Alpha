// Body rendering takes a pose and pixel placement, with no room dependency.
const lyingSprite=new Image();lyingSprite.src="/arcus-lying.png";
const sittingSprite=new Image();sittingSprite.src="/arcus-sitting.png";
export function drawArcus(ctx, sprite, body, placement) {
  const {x, y} = placement;
  ctx.save();
  ctx.fillStyle = "#59655425";
  ctx.beginPath();ctx.ellipse(x,y+3,50,12,0,0,Math.PI*2);ctx.fill();
  ctx.translate(x,y);ctx.scale(body.facing === "left" ? -1 : 1,1);
  const pose=body.visual_pose||{standing_weight:1,width:160,height:129,kind:"standing"};
  const weight=pose.standing_weight,height=pose.height,width=pose.width;
  for(const [art,alpha] of [[lyingSprite,pose.lying_weight??1-weight],[sprite,weight],[sittingSprite,pose.sitting_weight||0]]){
    if(art.complete && art.naturalWidth && alpha>0){ctx.globalAlpha=alpha;ctx.drawImage(art,-width/2,-height+8,width,height);}
  }
  ctx.restore();
  ctx.font="9px Segoe UI, sans-serif";ctx.fillStyle="#5c785f";ctx.textAlign="center";
  ctx.fillText("ARCUS · "+pose.kind.toUpperCase(),x,y+27);
  if(body.eyelid_openness===0){ctx.font="11px Segoe UI, sans-serif";ctx.fillText(body.sleep_state==="sleeping"?"Sleeping":"Eyes closed",x,y+42);}
  ctx.font="19px Segoe UI, sans-serif";
  ctx.fillText({up:"↑",down:"↓",left:"←",right:"→"}[body.facing],x+64,y-28);
}
