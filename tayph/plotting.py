__all__ = [
    "plotting_scales_2D"
]


def plotting_scales_2D(rv,y,ccf,xrange,yrange,Nxticks=10.0,Nyticks=10.0,nsigma=3.0):
    """
    This is a primer for plot_ccf, which defines the plotting ranges and colour
    scale of a 2D CCF. It can probably be used generally for any 2D image, but it was
    meant for cross-correlations or specral-time-series-like Figures.
    """
    import numpy as np
    import tayph.functions as fun
    import math
    import pdb
    #Define the plotting range of the images.
    nrv=len(rv)
    ny=len(y)
    drv=rv[1]-rv[0]
    dy=y[1]-y[0]
    sel = ((rv >= xrange[0]) & (rv <=xrange[1]))
    ccf_sub = ccf[:,sel]
    sely = ((y >= yrange[0]) & (y <= yrange[1]))
    ccf_sub = ccf_sub[sely,:]
    vmin,vmax = fun.sigma_clip(ccf_sub,nsigma=nsigma)#Sigma clipping.
    rvmin=rv[sel].min()
    rvmax=rv[sel].max()
    ymin=y[sely].min()
    ymax=y[sely].max()
    #This initiates the meshgrid onto which the 2D image is defined.
    xmin = rvmin; xmax = rvmax; dx = drv#Generalization away from rv.
    x2,y2 = np.meshgrid(np.arange(xmin,xmax+dx+dx,dx)-dx/2.,np.arange(ymin,ymax+dy+dy,dy)-dy/2.)
    #Try the commented out line if suddenly there are problems with x2 and y2 not matching z... ?
    # x2,y2 = np.meshgrid(np.arange(xmin,xmax+dx,dx)-dx/2.,np.arange(ymin,ymax+dy,dy)-dy/2.)

    dxt = (xmax+dx - xmin) / Nxticks
    dyt = (ymax+dy - ymin) / Nyticks

    ndigits_dxt= -1.0*(min([math.floor(np.log(dxt)),0]))#Set the rounding number of digits to either 0 (ie integers)
    ndigits_dyt= -1.0*(min([math.floor(np.log(dyt)),0]))#or a power of 10 smaller than that, otherwise.

    xticks = np.arange(xmin,xmax+dx,round(dxt,int(ndigits_dxt)))
    yticks = np.arange(ymin,ymax+dy,round(dyt,int(ndigits_dyt)))
    return(x2,y2,ccf_sub,rv[sel],y[sely],xticks,yticks,vmin,vmax)
