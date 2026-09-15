"""Independent vector architecture preview; perspective planes are schematic."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent.parent / 'reports/text_router_analysis/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'path'})
fig, ax = plt.subplots(figsize=(11.8,5.4))
fig.subplots_adjust(left=.015,right=.99,top=.98,bottom=.025)
ax.set(xlim=(0,12),ylim=(0,5.5)); ax.axis('off')
GRAY='#687786'; BLUE='#0072B2'; GOLD='#ad8234'
def text(x,y,s,size=10,**kw):
    ax.text(x,y,s,fontsize=size,ha='center',va='center',**kw)
def arrow(points,color=GRAY,dashed=False):
    for p,q in zip(points[:-2],points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.05,ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,lw=1.05,color=color,linestyle='--' if dashed else '-',shrinkA=0,shrinkB=0,zorder=2))
def block(x,y,w,h,label,blue=False):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor='#d8ebf6' if blue else '#eef0f3',edgecolor=BLUE if blue else GRAY,lw=1,zorder=3))
    text(x,y,label,10,color=BLUE if blue else '#263440',zorder=4)
def plane(x,y,w,level):
    # Parallel oblique planes: size, not perspective depth, encodes relative resolution.
    h=w*.32; dx=w*.23; dy=w*.18; thick=.065
    vertices=np.array([[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2+dx,y+h/2+dy],[x-w/2+dx,y+h/2+dy]])
    ax.add_patch(Polygon([vertices[0],vertices[1],vertices[1]+[0,-thick],vertices[0]+[0,-thick]],facecolor='#c5cdd6',edgecolor='none',zorder=3))
    ax.add_patch(Polygon([vertices[1],vertices[2],vertices[2]+[0,-thick],vertices[1]+[0,-thick]],facecolor='#b0bcc9',edgecolor='none',zorder=3))
    ax.add_patch(Polygon(vertices,facecolor='#e3e8ed',edgecolor='#8695a4',lw=.8,zorder=4))
    for t in [.25,.5,.75]:
        a=vertices[0]*(1-t)+vertices[1]*t;b=vertices[3]*(1-t)+vertices[2]*t
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#c1ccd6',lw=.45,zorder=5)
        a=vertices[0]*(1-t)+vertices[3]*t;b=vertices[1]*(1-t)+vertices[2]*t
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#c1ccd6',lw=.45,zorder=5)
    text(x+dx/2,y+dy/2,level,11,color=BLUE if level=='P5' else '#354554',zorder=6,bbox=dict(facecolor='#e3e8ed',edgecolor='none',pad=1))
    return x-w/2+dx/2,x+w/2+dx/2

photo=np.flipud(plt.imread(HERE/'method_input.png'))
ax.imshow(photo,extent=(.1,1.25,2.2,3.85),aspect='auto',zorder=3)
text(.67,4.1,'Image',11)
arrow([(1.25,3),(1.65,3)])
# Tapered extraction stage, compact fusion stage, explicit output feature pyramid.
ax.add_patch(Polygon([(1.65,1.95),(1.65,4.45),(3.05,4.05),(3.05,2.35)],facecolor='#eef0f3',edgecolor='#8695a4',lw=1,zorder=3))
text(2.33,3.2,'Backbone',11)
block(3.75,3.2,.72,2.25,'Neck')
text(3.75,4.65,'Multi-scale\nfusion',9,color=GRAY)
for y in [4,3.15,2.35]:arrow([(3.05,y),(3.39,y)])
for y,w,name in [(4,1.55,'P3'),(3.15,1.12,'P4'),(2.35,.76,'P5')]:
    left,right=plane(5.35,y,w,name)
    arrow([(4.11,y),(left-.06,y)])
    if name=='P5':
        arrow([(right+.04,y),(6.6,y)],BLUE)
        block(7.4,y,1.6,.66,'Two-expert\nadapter',True)
        arrow([(8.2,y),(8.65,y)],BLUE)
    else:arrow([(right+.04,y),(8.65,y)])
block(9.12,3.15,.94,2.45,'Detection\nhead')
arrow([(9.59,3.15),(10.12,3.15)])
ax.imshow(photo,extent=(10.15,11.55,2.12,4.13),aspect='auto',zorder=3)
text(10.85,4.45,'Detections',11)
for x1,y1,x2,y2,label,col in [(22,230,805,750,'bus 0.92',BLUE),(50,400,205,910,'person 0.88','#D55E00')]:
    x=10.15+x1/810*1.4;y=4.13-y2/1080*2.01;w=(x2-x1)/810*1.4;h=(y2-y1)/1080*2.01
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=col,lw=1,zorder=4))
    ax.text(x,y+h,label,ha='left',va='bottom',fontsize=6,color='white',bbox=dict(facecolor=col,edgecolor='none',pad=.7),zorder=5)
text(10.85,1.91,'Illustrative output',7.5,color=GRAY)
# The text encoder has its own lane, outside the visual feature extractor.
text(.82,.93,'Class names',10)
block(2.33,.93,1.35,.55,'Text encoder')
arrow([(1.48,.93),(1.655,.93)],GOLD)
arrow([(3.005,.93),(7.4,.93),(7.4,2.02)],GOLD,True)
text(5.2,1.15,'Routing text',8,color=GOLD)
arrow([(3.15,.93),(3.15,.4),(9.12,.4),(9.12,1.925)],GOLD,True)
text(6.2,.59,'Classification text',8,color=GOLD)
ax.plot(3.15,.93,'o',ms=2.5,color=GOLD,zorder=4)
for ext in ['svg']:
    fig.savefig(HERE/f'fig1_method.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)

# Normalize SVG whitespace for readable repository diffs.
for svg in HERE.glob('fig1_method.svg'):
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
