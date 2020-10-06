__all__ = [
    "plotting_scales_2D",
    "zoom_factory",
    "adjust_title",
    "interactive_legend"
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



def interactive_legend(fig,ax,lines):
    leg = ax.legend(loc='upper left', fancybox=False, shadow=False)
    leg.get_frame().set_alpha(0.4)


# we will set up a dict mapping legend line to orig line, and enable
# picking on the legend line
    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline


    def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)

def adjust_title(s):
    """Change the title of an active plot to the string s."""
    import matplotlib.pyplot as plt
    plt.title(s, fontsize=16)
    plt.draw()


def zoom_factory(ax, max_xlim, max_ylim, base_scale = 2.):
    """https://stackoverflow.com/questions/29145821/can-python-matplotlib-ginput-be-independent-from-zoom-to-rectangle"""
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
            x_scale = scale_factor / 2
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
            x_scale = scale_factor * 2
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * x_scale
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        if xdata - new_width * (1 - relx) > max_xlim[0]:
            x_min = xdata - new_width * (1 - relx)
        else:
            x_min = max_xlim[0]
        if xdata + new_width * (relx) < max_xlim[1]:
            x_max = xdata + new_width * (relx)
        else:
            x_max = max_xlim[1]
        if ydata - new_height * (1 - rely) > max_ylim[0]:
            y_min = ydata - new_height * (1 - rely)
        else:
            y_min = max_ylim[0]
        if ydata + new_height * (rely) < max_ylim[1]:
            y_max = ydata + new_height * (rely)
        else:
            y_max = max_ylim[1]
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.figure.canvas.draw()

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)
    #return the function
    return zoom_fun


def plot_ccf(rv,ccf,dp,xrange=[-200,200],yrange=[0,0],Nxticks=10.0,Nyticks=10.0,title='',doppler_model = [],i_legend=True,show=True):
    """
    This is a routine that does all the plotting of the cleaned 2D CCF. It overplots
    the expected planet velocity, modified by the systemic velocity given in the config file.
    Optionally, a trace of the doppler model is added as a list (of y-values) in the
    doppler_model parameter. Set i_legend to False if you wish to remove the interactive legend.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    import tayph.drag_colour as dcb
    import tayph.functions as fun
    import pylab as pl
    import tayph.system_parameters as sp

    #Load necessary physics for overplotting planet velocity.
    vsys = sp.paramget('vsys',dp)
    RVp = sp.RV(dp)+vsys
    nexp = np.shape(ccf)[0]

    #Default to the entire y-axis if yrange = [0,0]
    if all(v == 0 for v in yrange):
        yrange=[0,nexp-1]

    x2,y2,z,rv_sel,y_sel,xticks,yticks,vmin,vmax = plotting_scales_2D(rv,fun.findgen(nexp),ccf,xrange,yrange,Nxticks=Nxticks,Nyticks=Nyticks,nsigma=3.0)
    #The plotting
    fig,ax = plt.subplots(figsize=(12,6))
    img=ax.pcolormesh(x2,y2,z,vmin=vmin,vmax=vmax,cmap='hot')
    ax.axis([x2.min(),x2.max(),y2.min(),y2.max()])
    line1, = ax.plot(RVp,fun.findgen(nexp),'--',color='black',label='Planet rest-frame')
    if len(doppler_model) > 0:
        line2, = ax.plot(doppler_model+vsys,fun.findgen(nexp),'--',color='black',label='Doppler shadow')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_title(title)
    ax.set_xlabel('Radial velocity (km/s)')
    ax.set_ylabel('Exposure')
    #The colourbar
    cbar = plt.colorbar(img,format='%05.4f',aspect = 15)
    # cbar.set_norm(dcb.MyNormalize(vmin=vmin,vmax=vmax,stretch='linear'))
    cbar = dcb.DraggableColorbar_fits(cbar,img,'hot')
    cbar.connect()

    #The clickable legend.
    if len(doppler_model) > 0:
        lines = [line1, line2]
    else:
        lines = [line1]

    if i_legend == True:
        interactive_legend(fig,ax,lines)
    if show == True:
        plt.show()
    return(fig,ax,cbar)
