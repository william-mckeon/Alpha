"""Explicit 2D gaze crop: normalized head plus paired-eye direction."""
from io import BytesIO
def crop_box(width,height,body):
    x=max(-1,min(1,body.get("head_yaw",0)+body.get("eye_yaw",0)))
    y=max(-1,min(1,body.get("head_pitch",0)+body.get("eye_pitch",0)))
    w,h=max(1,width//2),max(1,height//2)
    left=round((x+1)/2*(width-w)); top=round((y+1)/2*(height-h))
    return left,top,left+w,top+h
def crop_frame(frame,body):
    from PIL import Image
    with Image.open(BytesIO(frame["bytes"])) as source:
        box=crop_box(source.width,source.height,body)
        image=source.crop(box)
        output=BytesIO(); image.save(output,"PNG")
    return {"bytes":output.getvalue(),"mime_type":"image/png","width":image.width,"height":image.height,
            "camera":{"mapping":"source-relative-2d","crop":list(box),"source_size":[source.width,source.height]}}

