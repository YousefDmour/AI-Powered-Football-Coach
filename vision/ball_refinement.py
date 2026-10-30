"""Tiled small-ball search and conservative multi-view near-foot exceptions.

This module is separate from the frozen baseline. Coordinates are always mapped
back to the source frame. No annotations are used by detection or tracking.
"""

import math

from vision.ball_tracking import center, outside_people, valid_ball


def tile_windows(width, height, tile_width=960, tile_height=720):
    if min(width,height,tile_width,tile_height)<=0:
        raise ValueError('Image and tile dimensions must be positive')
    def positions(length,size):
        size=min(length,size)
        starts=list(range(0,length-size+1,max(1,size//2)))
        if starts[-1]!=length-size:starts.append(length-size)
        return starts,size
    xs,w=positions(width,tile_width);ys,h=positions(height,tile_height)
    return [(x,y,x+w,y+h) for y in ys for x in xs]


def overlap(a,b):
    left,top=max(a[0],b[0]),max(a[1],b[1])
    right,bottom=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,right-left)*max(0,bottom-top)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def merge_views(candidates):
    merged=[]
    for candidate in sorted(candidates,key=lambda row:row['confidence'],reverse=True):
        match=next((item for item in merged if overlap(item['bbox'],candidate['bbox'])>=.35
                    or math.dist(center(item['bbox']),center(candidate['bbox']))<=6),None)
        if match is None:
            merged.append({**candidate,'views':list(candidate['views'])})
        else:
            match['views']=sorted(set(match['views'])|set(candidate['views']))
    return merged


def detect_tiles(detector,frame):
    height,width=frame.shape[:2]
    raw=[]
    for number,(left,top,right,bottom) in enumerate(tile_windows(width,height)):
        result=detector.predict(frame[top:bottom,left:right],classes=[32],conf=.05,
                                imgsz=960,verbose=False)[0]
        for box,confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            mapped=[box[0]+left,box[1]+top,box[2]+left,box[3]+top]
            raw.append({'bbox':mapped,'confidence':confidence,'views':[number]})
    return merge_views(raw)


def near_feet(box,people):
    x,y=center(box)
    return any(left-5<=x<=right+5 and top+.65*(bottom-top)<=y<=bottom+5
               for left,top,right,bottom in people)


def filter_candidates(frame,candidates,people,allow_feet=False):
    accepted=[]
    for candidate in candidates:
        if not valid_ball(frame,candidate['bbox']):continue
        outside=outside_people(candidate['bbox'],people)
        exception=(allow_feet and candidate['confidence']>=.30
                   and len(set(candidate['views']))>=2 and near_feet(candidate['bbox'],people))
        if outside or exception:
            accepted.append({**candidate,'near_feet_exception':not outside})
    return accepted


def guard_foot_candidates(candidates,strict_tracker,frame_number):
    """An exception may extend a recently confirmed independent strict track.

    It cannot start a track or pull it to unrelated footwear elsewhere.
    """
    accepted=[]
    for candidate in candidates:
        if not candidate.get('near_feet_exception',False):
            accepted.append(candidate)
            continue
        if strict_tracker.last_frame is None or strict_tracker.last_position is None:
            continue
        gap=frame_number-strict_tracker.last_frame
        if (strict_tracker.hits>=2 and 1<=gap<=2
                and math.dist(center(candidate['bbox']),strict_tracker.last_position)<=strict_tracker.max_step*gap):
            accepted.append(candidate)
    return accepted
