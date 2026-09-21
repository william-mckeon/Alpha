// QWebChannel is injected only by the native host into our local page.
let host = null;
export function desktopAvailable(){return host !== null;}
export function initDesktop(canvas){
  if(!window.qt?.webChannelTransport || !window.QWebChannel)return;
  new QWebChannel(qt.webChannelTransport, channel=>{
    host=channel.objects.arcusDesktop;
    function geometry(){
      const r=canvas.getBoundingClientRect();
      host.geometry(JSON.stringify({x:r.x,y:r.y,width:r.width,height:r.height}));
    }
    geometry();
    new ResizeObserver(geometry).observe(canvas);
    window.addEventListener("scroll",geometry,true);
    window.addEventListener("resize",geometry);
    document.getElementById("desktop-status").textContent="Desktop host connected · drag Arcus to pick him up";
  });
}
export function pickup(){if(host)host.pickup();}
export function returnToPen(){if(host)host.returnToPen();}
