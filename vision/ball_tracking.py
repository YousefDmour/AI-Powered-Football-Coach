"""Observed-only ball tracking: hidden or ambiguous means no reported position."""

import math
from collections import deque

import cv2
import numpy as np


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def valid_ball(frame, box):
    """Reject implausible sizes and candidates outside a green pitch neighborhood."""
    x1,y1,x2,y2 = map(float,box)
    width,height=x2-x1,y2-y1
    if not 3 <= width <= 36 or not 3 <= height <= 36 or not .4 <= width/height <= 2.5:
        return False
    x,y=center(box)
    radius=max(14,int(max(width,height)*2))
    left,right=max(0,int(x)-radius),min(frame.shape[1],int(x)+radius)
    top,bottom=max(0,int(y)-radius),min(frame.shape[0],int(y)+radius)
    crop=frame[top:bottom,left:right]
    if not crop.size:return False
    ball_patch=frame[max(0,int(y1)):min(frame.shape[0],int(y2)+1),
                     max(0,int(x1)):min(frame.shape[1],int(x2)+1)]
    if not ball_patch.size:return False
    ball_hsv=cv2.cvtColor(ball_patch,cv2.COLOR_BGR2HSV)
    # This clip has a predominantly white ball; reject colored boots/flags.
    white=(ball_hsv[:,:,1]<=100)&(ball_hsv[:,:,2]>=155)
    if float(np.mean(white))<.20:return False
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=35)
    return float(np.mean(green)) >= .35


def outside_people(box, people, margin=5):
    """Abstain on detections inside/adjacent to people, including ambiguous feet."""
    x,y=center(box)
    return not any(left-margin<=x<=right+margin and top-margin<=y<=bottom+margin
                   for left,top,right,bottom in people)


class BallTrack:
    def __init__(self, max_step=75, max_gap=4, minimum_confidence=.06):
        self.max_step=max_step
        self.max_gap=max_gap
        self.minimum_confidence=minimum_confidence
        self.reset()

    def reset(self):
        self.last_position=None
        self.last_frame=None
        self.last_update=None
        self.hits=0
        self.trail=deque(maxlen=20)

    def update(self, frame_number, candidates, global_search=False):
        """Candidates are measured boxes. Motion gates never create output points."""
        if self.last_update is not None and frame_number <= self.last_update:
            raise ValueError("Ball frames must increase")
        if self.last_frame is not None and frame_number-self.last_frame > self.max_gap:
            self.reset()
        # A strong independent global detection may recover a lost/wrong local
        # search. It still needs another observed frame before being displayed.
        strongest=max(candidates,key=lambda item:item['confidence'],default=None)
        if global_search and strongest and strongest['confidence']>=.35 and self.last_position:
            if math.dist(center(strongest['bbox']),self.last_position)>self.max_step*2:
                self.reset()
                candidates=[strongest]
        self.last_update=frame_number
        ranked=[]
        gap=frame_number-self.last_frame if self.last_frame is not None else 1
        for candidate in candidates:
            confidence=float(candidate['confidence'])
            if confidence < self.minimum_confidence:continue
            point=center(candidate['bbox'])
            distance=math.dist(point,self.last_position) if self.last_position else 0
            if self.last_position and distance > self.max_step*min(gap,2):continue
            score=confidence-(.35*distance/(self.max_step*min(gap,2)) if self.last_position else 0)
            ranked.append((score,candidate,point))
        ranked.sort(key=lambda row:row[0],reverse=True)
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.08 and math.dist(ranked[0][2],ranked[1][2])>20:
            ranked=[]
        if not ranked:
            self.hits=0
            self.trail.clear()
            return {'state':'unknown','x':None,'y':None,'confidence':None,'bbox':None}
        _,candidate,point=ranked[0]
        consecutive=self.last_frame is not None and frame_number==self.last_frame+1
        self.hits=self.hits+1 if consecutive else 1
        self.last_position,self.last_frame=point,frame_number
        if self.hits<2:
            self.trail.clear()
            return {'state':'pending','x':None,'y':None,'confidence':None,'bbox':None}
        self.trail.append((int(point[0]),int(point[1])))
        return {'state':'observed','x':point[0],'y':point[1],
                'confidence':candidate['confidence'],'bbox':candidate['bbox']}
