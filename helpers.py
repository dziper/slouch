# open cv HSV Color bounds: 0-360, 0-255, 0-255
def rgbToHsv(color):
    R=color[0]
    G=color[1]
    B=color[2]
    r= R/255
    g= G/255
    b= B/255
    mx=getMax([r,g,b])
    mn=getMin([r,g,b])
    div=mx-mn
    if div!=0:
        if mx==r:
            h=60*(((g-b)/div)%6)
        elif mx==g:
            h=60*(((b-r)/div)+2)
        elif mx==b:
            h=60*(((r-g)/div)+4)
        s=(div)/mx
        v=mx

    elif div ==0:
        if mx==0:
            h=0
            s=0
            v=0
        else:
            h=mx
            s=mx
            v=mx


    return (h,s*255,v*255)

def hsvToRgb(color):
    h = color[0]
    s = color[1]/255
    v = color[2]/255

    h=h%360

    C=v*s
    X=C*(1-(numpy.abs((h/60)%2-1)))
    m=v-C
    if 0<=h<60:
        (R,G,B)=(C,X,0)
    elif 60<=h<120:
        (R,G,B)=(X,C,0)
    elif 120<=h<180:
        (R,G,B)=(0,C,X)
    elif 180<=h<240:
        (R,G,B)=(0,X,C)
    elif 240<=h<300:
        (R,G,B)=(X,0,C)
    elif 300<=h<360:
        (R,G,B)=(C,0,X)
    (r,g,b)=((R+m)*255, (G+m)*255, (B+m)*255)

    ra=int(r)
    ga=int(g)
    da= int(b)
    return (ra,ga,da)
