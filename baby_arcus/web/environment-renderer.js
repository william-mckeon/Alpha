export const floor = {x:100,y:170,w:900,h:490};
export const point = (x,y,environment) => [floor.x+x/environment.width*floor.w,floor.y+y/environment.height*floor.h];
export function drawEnvironment(ctx,environment){
function round(x,y,w,h,r,color,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.stroke();}}
function text(t,x,y,size=14,color="#697565",align="left"){ctx.font=`${size}px Segoe UI, sans-serif`;ctx.fillStyle=color;ctx.textAlign=align;ctx.fillText(t,x,y);}
function ellipse(x,y,rx,ry,color){ctx.beginPath();ctx.ellipse(x,y,rx,ry,0,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();}
 ctx.clearRect(0,0,1100,740);round(0,0,1100,740,0,"#e8eddf");
 // A dollhouse view: four perimeter walls, with low front/side walls for visibility.
 round(85,53,930,616,17,"#d2dcc9");round(100,65,900,130,5,"#eef0de");
 for(let x=117;x<1000;x+=40){ctx.strokeStyle="#e4e8d5";ctx.beginPath();ctx.moveTo(x,65);ctx.lineTo(x,188);ctx.stroke();}
 round(100,170,900,490,0,environment?.colors?.floor||"#dfcbae");
 for(let y=190;y<660;y+=45){ctx.strokeStyle="#d2bc9b";ctx.beginPath();ctx.moveTo(100,y);ctx.lineTo(1000,y);ctx.stroke();for(let x=180+(y%90)*3;x<1000;x+=180){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x,Math.min(y+45,660));ctx.stroke();}}
 round(93,164,914,12,3,"#b7c3aa");
 // Window, wall art and furniture remain outside the traversable floor.
 round(166,79,113,73,8,"#c6d4b9");round(173,85,99,60,3,"#e3ecdf");
 ellipse(247,101,10,10,"#f3d596");ctx.strokeStyle="#bdccad";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(220,85);ctx.lineTo(220,145);ctx.moveTo(173,116);ctx.lineTo(272,116);ctx.stroke();ctx.lineWidth=1;
 round(807,90,66,65,5,"#c7b496");round(813,96,54,53,2,"#f7edda");text("PLAY",840,129,13,"#9ba988","center");

 round(275,272,551,275,60,"#b5c5a3");round(290,287,521,245,50,environment?.colors?.rug||"#c5d1b4");
 ctx.setLineDash([5,9]);ctx.strokeStyle="#aabf98";ctx.lineWidth=2;ctx.beginPath();ctx.roundRect(304,301,493,217,42);ctx.stroke();ctx.setLineDash([]);ctx.lineWidth=1;
 text("a place to begin",550,420,22,"#adbfa0","center");
 round(85,163,16,510,4,environment?.colors?.wall||"#b6c4a9");round(999,163,16,510,4,environment?.colors?.wall||"#b6c4a9");round(85,660,930,15,5,environment?.colors?.wall||"#c4ceb5");
 if(environment)for(const object of Object.values(environment.objects||{})){
  const [x,y]=point(object.x,object.y,environment);
  ellipse(x,y,object.radius/environment.width*floor.w,object.radius/environment.height*floor.h,object.color);
  text(object.name+(object.discovered?' · '+object.discovered:''),x,y+30,12,'#405440','center');
 }
}
