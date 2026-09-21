"""Observation rendering from simulation state only, never a desktop crop."""
from io import BytesIO
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=3)
def sprite_source(name):
    from PIL import Image
    with Image.open(Path(__file__).parent/'web'/name) as original:
        return original.convert('RGBA')


def capture_playpen(state,include_body=True):
    from PIL import Image, ImageDraw
    colors=state['environment'].get('colors',{'floor':'#e2d2b8','wall':'#7d9472','rug':'#c5d1b4'})
    image = Image.new("RGB", (800, 560), colors['floor'])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 799, 559), outline=colors['wall'], width=15)
    draw.rounded_rectangle((140, 130, 660, 440), radius=45, fill=colors['rug'])
    env, body = state["environment"], state["arcus"]
    def point(p):
        return p["x"]/env["width"]*770+15, p["y"]/env["height"]*530+15
    for obj in env.get('objects',{}).values():
        ox,oy=point(obj);rx=obj['radius']/env['width']*770;ry=obj['radius']/env['height']*530
        draw.ellipse((ox-rx,oy-ry,ox+rx,oy+ry),fill=obj['color'],outline='#405440')
    h = env["human"]
    hx, hy = point(h)
    if h.get('present',True):
        draw.ellipse((hx-18, hy-18, hx+18, hy+18), fill="#e36c3c")
        draw.text((hx-15, hy+22), h["name"], fill="#405440")
    view = state["view"]
    p = env["placements"].get(body["entity_id"])
    if include_body and p and not view["held"] and view["region"] == "playpen":
        pose=body.get("visual_pose",{"standing_weight":1,"width":160,"height":129})
        size=(round(pose["width"]*.75),round(pose["height"]*.75))
        layers=[]
        for name in ("arcus-lying.png","arcus-body.png","arcus-sitting.png"):
            layers.append(sprite_source(name).resize(size))
        seated=pose.get("sitting_weight",0)
        standing=pose["standing_weight"]/(1-seated) if seated<1 else 0
        sprite=Image.blend(Image.blend(layers[0],layers[1],max(0,min(1,standing))),layers[2],seated)
        if body["facing"] == "left":
            sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x, y = point(p)
        image.paste(sprite, (round(x-60), round(y-sprite.height+8)), sprite)
    output = BytesIO()
    image.save(output, "PNG")
    return {"bytes": output.getvalue(), "mime_type": "image/png", "width": 800, "height": 560,"gaze_enabled":True}
