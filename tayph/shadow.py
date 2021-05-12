__all__ = [
    "xcor",
    "clean_ccf",
    "filter_ccf",
    "shift_ccf",
    "construct_KpVsys",
]



def prime_doppler_model(fig,ax,cbar):
    import matplotlib.pyplot as plt
    import tayph.plotting as fancyplots
    import numpy as np
    import sys
    import pdb
    import time
    from matplotlib.widgets import Button
    import tayph.drag_colour as dcb
    """This uses the active plot_ccf figure to let the user manually select where the
    #doppler shadow feature is located by placing a line on it."""
    fancyplots.adjust_title('Please define a line that traces the shadow by clicking on its top and bottom.')
    plt.subplots_adjust(bottom=0.2)#Make room for a set of buttons.

    class Index(object):
        #&&&&&&&&This class contains the behaviour of the buttons and the "loop" when pressing
        #reset.
        #First we define a function to let the user click on the start and end of the
        #shadow feature.
        def measure_line(self):
            from matplotlib.widgets import Button
            pts = []#This contais the startpoint and endpoint.
            while len(pts) < 2:
                pts = np.asarray(plt.ginput(2, timeout=-1,mouse_pop=3,mouse_add=1))#This waits for the user
                #to click on the plot twice. Mouse_add=1 means that its using the left mouse button.
                #Mouse_pop would be used to remove points, but in this case the right mouse is also used
                #for drag-adjusting the colourbar. I effectively had to disable it in ginput and this was the
                #way to do it.

            #These 3 lines create class-wide variables (attributes) top/bottom point and line.
            #these can be called from the outside (see below).
            self.top_point = [pts[0,0],pts[0,1]]#X and Y coordinates of the first point
            self.bottom_point = [pts[1,0],pts[1,1]]#and the second point.
            #The following plots the line onto the active figure:
            self.line = ax.plot([self.top_point[0],self.bottom_point[0]],[self.top_point[1],self.bottom_point[1]],'-',color='black')#Put the input points
            plt.draw()
            print('Selected points:')#and output some chatter to the terminal.
            print(f'   {self.top_point}')
            print(f'   {self.bottom_point}')
            #Ok, so now this function is defined in the class.


        def __init__(self):
            #This is what is immediately exectued when the class is initialised somewhere.
            #We start by calling the above function. I.e. this prime_doppler_model function
            #starts by letting the user do the clicking. (now look at the initialisation
            #line below *****)
            self.measure_line()

        def reset(self,event):
            #When pressing the reset button, the user indicates that he/she is not
            #happy with the line. This function removes it from the plot.
            for p in self.line:#all lines, in case self.line is a list.
                p.remove()#Remove it (ive never seen this before but,, awesome...)
            plt.draw()
            self.top_point = [0,0]#We set the points chosen by the user to zero.
            self.bottom_point = [0,0]#This is to let the ok button (see below) know
            #that reset has been pressed, so it can be deactivated while the user is
            #defining a new line.
            self.measure_line()#Measure line again.

        def okay(self, event):
            #First check that the points are not both zero, i.e. that the reset button
            #has not been pressed.
            if all(v == 0 for v in self.top_point) and all(w == 0 for w in self.bottom_point):
                pass
            else:
                plt.close('all')#This is again interesting, in order to accept the chosen
                #line, all we need to do is close the figure; which breaks out of
                #plt.show() below.


    #***** here:
    callback = Index()
    #Now we create 2 small axis objects in which we are gonna put two buttons.
    axreset = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axokay = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axreset.set_title('')
    axokay.set_title('')
    breset = Button(axreset, 'Reset')#This fills in a button object in the above axes.
    bokay = Button(axokay, 'OK')
    bokay.on_clicked(callback.okay)#Now this goes back into the class object.
    breset.on_clicked(callback.reset)
    #Actually, the whole reason I needed to create a class is because the button event
    #handling is done using such a class object. I have two buttons that can be clicked,
    #and each has its own function inside the object.

    plt.show()#This line is super interesting. plt.show() holds the program in a certain state,
    #waiting for the user to close the plot. During this time, the buttons can (apparently)
    #be clicked. So now go back up to &&&&&&

    return([callback.top_point,callback.bottom_point])
    #Then, when we have broken out of the figure, we return the chosen points as a list.





def evaluate_poly(phase,params,S,D):
    A_poly = 1.0
    W_poly = 1.0
    if D > 0:#The polynomial correction is applied. Setting D to zero is a way to switch this off.
        if phase>0.5:
            phase-=1.0#Make symmetric around zero in case of transits.
        if S == 0:
            A_poly += phase**D * params[7]
            W_poly += phase**D * params[8]
        if S == 1:#If it is symmetric, the number of parameters per polynomial is half.
        #E.g. with a 6th, 4th and 2nd components if D==6.
            for ii in range(int(D/2)):
                A_poly += phase**(ii*2+2) * params[7+ii]
                W_poly += phase**(ii*2+2) * params[7+int(D/2)+ii]
        if S == 2:
            for ii in range(int(D)):
                A_poly += phase**(ii+1) * params[7+ii]
                W_poly += phase**(ii+1) * params[7+int(D)+ii]
    return(A_poly,W_poly)

def evaluate_shadow(params,rv,ccf,transit,phase,aRstar,vsys,inclination,n_components,offset_second_component,S,D,leastsq=True):
    import numpy as np
    import tayph.functions as fun
    import pdb
    import matplotlib.pyplot as plt
    """This evaluates the doppler shadow model. Primary usage is in leastsq and in evaluation
    of its output. Still need to write good documentation and tests. If leastsq is true, it
    returns the flattened difference between the input CCF and the model. If it is set to false,
    it just returns the model; and the ccf object is ignored.

    Offset is the velocity offset of the second component."""
    modelled_ccf = ccf*0.0
    modelled_ccf[np.isnan(modelled_ccf)] = 0.0
    nexp = np.shape(ccf)[0]
    #From https://www.aanda.org/articles/aa/pdf/2016/04/aa27794-15.pdf
    A = params[0]#Amplitude
    l = params[1]#Spin-orbit angle
    vsini = params[2]#Stellar vsini
    W = params[3]
    C = params[4]
    A2 = params[5]
    W2 = params[6]


    v_star = fun.local_v_star(phase,aRstar,inclination,vsini,l)+vsys
    for i in range(nexp):
        A_poly,W_poly = evaluate_poly(phase[i],params,S,D)
        modelled_ccf[i,:] = transit[i]*A * A_poly * np.exp(-(rv-v_star[i])**2 / (2.0*(W*W_poly)**2)) + C
        if n_components == 2:
            modelled_ccf[i,:] += transit[i]*A2 * np.exp(-(rv-v_star[i]-offset_second_component)**2 / (2.0*W2**2))
    if leastsq == True:
        diffs = modelled_ccf - ccf
        diffs[np.isnan(diffs)] = 0.0
        return diffs.flatten() # it expects a 1D array out.
    else:
        return modelled_ccf
               # it doesn't matter that it's conceptually 2D, provided that I flattened it consistently

def read_shadow(dp,shadowname,rv,ccf):
    """
    This reads  shadow parameters from a pickle file generated by the fit_doppler_
    model class below, and evaluates it on the user-supplied rv and ccf (the latter of which is
    just for shape-purposes and the mask defined during the creation of the shadow model.
    """
    import pickle
    import os.path
    import sys
    from pathlib import Path
    import tayph.util as ut
    from tayph.vartests import typetest,dimtest
    import tayph.system_parameters as sp
    import numpy as np
    typetest(shadowname,str,'shadowname in read_shadow().')
    typetest(rv,np.ndarray,'rv in read_shadow()')
    typetest(ccf,np.ndarray,'ccf in read_shadow()')


    dp = ut.check_path(dp,exists=True)
    inpath = dp/(shadowname+'.pkl')
    ut.check_path(inpath,exists=True)
    RVp = sp.RV(dp)
    pickle_in = open(inpath,"rb")
    d = pickle.load(pickle_in)
    params = d["fit_params"]
    T = d["Transit"]
    p = d["Phases"]
    aRstar = d["aRstar"]
    vsys = d["vsys"]
    i = d["inclination"]
    n_c = d["n_components"]
    maskHW = d["maskHW"]
    offset = d["offset"]#Offset of second component.
    S=d["S"]
    D=d["D"]
    RVp = sp.RV(dp)+vsys
    mask = mask_ccf_near_RV(rv,ccf,RVp,maskHW)
    return(evaluate_shadow(params,rv,ccf,T,p,aRstar,vsys,i,n_c,offset,S,D,leastsq=False),mask)


def mask_ccf_near_RV(rv,ccf,RVp,hw):
    """
    This mask a region in the ccf near a the velocity RVp, typically for the
    purpose of masking velocities that are close to the expected velocity of the planet
    (hence RVp). RVp should be a list or array-like object with the same number of elements
    as there are rows in ccf; which is a 2D array-like object with x-size equal to the
    number of values in rv (its x-axis). The width of the masked area is twice the
    hw parameter (half-width).

    The output mask has values 1 everywhere, apart from the masked area which is
    set to NaN. The mask is therefore typically used in a multiplicative sense.
    """
    import numpy as np
    nexp = np.shape(ccf)[0]
    ccf_mask = ccf*0.0+1.0
    for i in range(nexp):
        sel = (rv >= RVp[i] - hw) & (rv <= RVp[i]+hw)
        ccf_mask[i,sel] = np.nan
    return(ccf_mask)

def match_shadow(rv,ccf,mask,dp,doppler_model):
    #THIS NEEDS TO BE DEBUGGED OR SIMPLIFIED (FOLDED INTO THE CLASS BELOW) ALTOGETHER
    import scipy.optimize
    # import pdb
    import numpy as np
    # import tayph.system_parameters as sp
    import tayph.util as ut
    import math

    # nexp = np.shape(ccf)[0]
    # # RVp = sp.RV(dp)
    # transit = sp.transit(dp)
    # mask = mask_ccf_near_RV(rv,ccf,RVp,maskHW)
    # masked_ccf = ccf*mask
    # ut.writefits('test2.fits',ccf*mask)
    # print(np.sum(np.isnan(masked_ccf))/nexp)
    # pdb.set_trace()
    #This is to fit the primer and provide starting parameters for vsini and lambda.
    def scale_shadow(scale,doppler_model,masked_ccf):
        import numpy as np
        diff = scale[0] * doppler_model - masked_ccf+scale[1]
        diff[np.isnan(diff)] = 0.0
        return(diff.flatten())
    result = scipy.optimize.leastsq(scale_shadow,[0.5,0.0],args = (doppler_model,ccf*mask))
    print(f'------Obtained scaling and offset: {np.round(result[0][0],3)}, '+format(result[0][1],'.2e'))
    print('------If the scaling is not nearly exactly 1 for the species for which the model was made, there is a problem.')

    matched_model = result[0][0]*doppler_model+result[0][1]
    return(ccf - matched_model,matched_model)







class fit_doppler_model(object):
    #This is my second home-made class: A doppler model.
    #The reason why I make it into a class is so that it can interact with a
    #GUI.

    def __init__(self,fig,ax,rv,ccf,primer,dp,outname):
        """We initialize with a figure object, three axis objects (in a list)
        an rv axis, the CCF, the user-generated prior on the doppler shadow (the two)
        points, and the pointer to the data in order to load physics."""
        import tayph.system_parameters as sp
        import numpy as np
        import pdb
        import scipy.interpolate as interpol
        import tayph.functions as fun
        import tayph.util as ut
        import sys
        #Upon initialization, we pass the keywords onto self.
        nexp = np.shape(ccf)[0]
        nrv = np.shape(ccf)[1]
        self.rv = rv
        self.ccf  = ccf
        self.p = sp.phase(dp)
        if len(self.p) != nexp:
            print('ERROR IN FIT_DOPPLER_MODEL __INIT__:')
            print('The height of the CCF does not match nexp.')
            sys.exit()
        transit = sp.transit(dp)-1.0
        self.T = abs(transit/max(abs(transit)))
        self.ax = ax
        self.aRstar = sp.paramget('aRstar',dp)
        self.vsys = sp.paramget('vsys',dp)
        self.RVp = sp.RV(dp) + self.vsys
        self.inclination = sp.paramget('inclination',dp)
        self.n_components = 1
        self.maskHW = 10.0 #Default masking half-width
        self.offset = 0.0
        self.D = 0#Initial degree of polynomial fitting.
        self.S = 0
        self.dp=ut.check_path(dp,exists=True)
        self.outpath=self.dp/(outname+'.pkl')
        #Translate the pivot to RV-phase points.
        #Need to interpolate on phase only, because the primer was already defined in RV space.
        p_i = interpol.interp1d(fun.findgen(nexp),self.p)
        p1 = float(p_i(primer[0][1]))
        p2 = float(p_i(primer[1][1]))
        v1 = primer[0][0]
        v2 = primer[1][0]

        self.primer = [[v1,p1],[v2,p2]]
        self.fit_spinorbit()
        # self.primer = a*self.rv+b
        self.v_star_primer = fun.local_v_star(self.p,self.aRstar,self.inclination,self.vsini_fit,self.l_fit)+self.vsys
        # ax[0].plot(v_star_primer,fun.findgen(nexp),'--',color='black',label='Spin-orbit fit to primer')
        self.mask_ccf()
        self.fit_model()


    def fit_spinorbit(self):
        """This is fit on a set (in this case 2) RV-phase pairs derived from the
        user-defined primer of the doppler shadow. These are later used by fit_model
        as starting parameters for the fit of the doppler-shadow feature."""
        #Why?
        import scipy.optimize
        import pdb

        #This is to fit the primer and provide starting parameters for vsini and lambda.
        def spinorbit(params,primer,vsys,aRstar,inclination):
            import numpy as np
            import tayph.functions as fun
            phases = np.array([primer[0][1],primer[1][1]])
            RVs = np.array([primer[0][0],primer[1][0]])
            l = params[0]
            vsini = params[1]
            v_star = fun.local_v_star(phases,aRstar,inclination,vsini,l)+vsys
            diff = v_star - RVs
            return(diff)
        result = scipy.optimize.leastsq(spinorbit,[0.7,10.0],args = (self.primer,self.vsys,self.aRstar,self.inclination))
        self.l_fit = result[0][0]
        self.vsini_fit = result[0][1]
        print(f'------Fitted lambda and v sin(i) from primer: {self.l_fit}, {self.vsini_fit}')



    def fit_model(self):#Only use class-wide variables.
            import scipy.optimize
            import numpy as np
            import tayph.functions as fun
            import tayph.util as ut
            import pdb
            A_start = np.max(np.abs(self.ccf))
            C_start = np.nanmedian(self.ccf)
            W_start = 5.0#km/s.
            A2_start = A_start * (-0.2)
            W2_start = 12.0#km/s.
            startparams = [A_start,self.l_fit,self.vsini_fit,W_start,C_start,A2_start,W2_start]

            if self.S == 0 and self.D > 0: #Only one high-order component.
                startparams+=[0,0]
            if self.S == 1 and self.D > 0:
                startparams+=self.D*[0]#Half-D for A and half-D for W, = 1*D.
            if self.S == 2 and self.D > 0:
                startparams+=2*self.D*[0]#D parameters for A and another D for w, = 2*D.

            result = scipy.optimize.leastsq(evaluate_shadow,startparams,args = (self.rv,self.ccf*self.ccf_mask,self.T,self.p,self.aRstar,self.vsys,self.inclination,self.n_components,self.offset,self.S,self.D)) # alternatively you can do this with closure variables in f if you like
            self.model = evaluate_shadow(result[0],self.rv,self.ccf*0.0,self.T,self.p,self.aRstar,self.vsys,self.inclination,self.n_components,self.offset,self.S,self.D,leastsq=False)
            # void,matched_ds_model=match_shadow(self.rv,self.ccf,self.ccf_mask,self.dp,self.model,self.maskHW)
            # ut.save_stack('test1.fits',[self.model,matched_ds_model])
            #This appears to be correct now to within a percent?


            self.out_result = result[0]
            self.l_final = result[0][1]
            self.vsini_final = result[0][2]
            print(f'------Fitted lambda and v sin(i) from entire feature: {self.l_final}, {self.vsini_final}')
            self.v_star_fit = fun.local_v_star(self.p,self.aRstar,self.inclination,self.vsini_final,self.l_final)+self.vsys

    def mask_ccf(self):
        self.ccf_mask = mask_ccf_near_RV(self.rv,self.ccf,self.RVp,self.maskHW)

    def save(self, event):
        """
        This is a dictionary that will hold the output (i.e. the fitting parameters
        and all the input values (including phases, vsys, and everything, needed
        to call evaluate_model exactly the way it was saved. The reason that I
        save all that information is because the number of RV points of the ccf to
        which the model is fit may be different from the ccf with which the model
        was initially constructed, but I don't want the removal of the doppler shadow
        to be dependent on that.
        """
        import matplotlib.pyplot as plt
        import pickle
        out_dict = {
            "fit_params": self.out_result,
            "Transit": self.T,
            "Phases": self.p,
            "aRstar": self.aRstar,
            "vsys": self.vsys,
            "inclination": self.inclination,
            "n_components": self.n_components,
            "maskHW": self.maskHW,
            "offset": self.offset,
            "S": self.S,
            "D": self.D}
        with open(self.outpath, 'wb') as f: pickle.dump(out_dict, f)
        plt.close('all')

    def cancel(self,event):
        import sys
        import matplotlib.pyplot as plt
        plt.close('all')
        print("Exiting")
        sys.exit()

























class fit_pulsation_model(object):
    #This is my second home-made class: A doppler model.
    #The reason why I make it into a class is so that it can interact with a
    #GUI.

    def __init__(self,fig,ax,rv,ccf,dp,outname):
        """We initialize with a figure object, three axis objects (in a list)
        an rv axis, the CCF, the user-generated prior on the doppler shadow (the two)
        points, and the pointer to the data in order to load physics."""
        import tayph.system_parameters as sp
        import numpy as np
        import pdb
        import scipy.interpolate as interpol
        import tayph.functions as fun
        import tayph.util as ut
        import sys
        #Upon initialization, we pass the keywords onto self.
        nexp = np.shape(ccf)[0]
        nrv = np.shape(ccf)[1]
        self.rv = rv
        self.ccf  = ccf
        self.p = sp.phase(dp)
        if len(self.p) != nexp:
            print('ERROR IN FIT_DOPPLER_MODEL __INIT__:')
            print('The height of the CCF does not match nexp.')
            sys.exit()
        # transit = sp.transit(dp)-1.0
        # self.T = abs(transit/max(abs(transit)))
        self.ax = ax
        # self.aRstar = sp.paramget('aRstar',dp)
        self.vsys = sp.paramget('vsys',dp)
        self.RVp = sp.RV(dp) + self.vsys
        # self.inclination = sp.paramget('inclination',dp)
        self.n_components = 0
        self.maskHW = 10.0 #Default masking half-width
        # self.offset = 0.0
        self.dp=ut.check_path(dp,exists=True)
        self.outpath=self.dp/(outname+'_pulsations.pkl')
        #Translate the pivot to RV-phase points.
        #Need to interpolate on phase only, because the primer was already defined in RV space.
        p_i = interpol.interp1d(fun.findgen(nexp),self.p)
        # p1 = float(p_i(primer[0][1]))
        # p2 = float(p_i(primer[1][1]))
        # v1 = primer[0][0]
        # v2 = primer[1][0]
        #
        # self.primer = [[v1,p1],[v2,p2]]
        # self.fit_spinorbit()
        # self.primer = a*self.rv+b
        # self.v_star_primer = fun.local_v_star(self.p,self.aRstar,self.inclination,self.vsini_fit,self.l_fit)+self.vsys
        # ax[0].plot(v_star_primer,fun.findgen(nexp),'--',color='black',label='Spin-orbit fit to primer')
        self.mask_ccf()
        # self.fit_model()



    def fit_model(self):#Only use class-wide variables.
            import scipy.optimize
            import numpy as np
            import tayph.functions as fun
            import tayph.util as ut
            A_start = np.max(np.abs(self.ccf))
            C_start = np.nanmedian(self.ccf)
            W_start = 5.0#km/s.
            A2_start = A_start * (-0.2)
            W2_start = 12.0#km/s.
            startparams = [A_start,self.l_fit,self.vsini_fit,W_start,C_start,A2_start,W2_start]

            result = scipy.optimize.leastsq(evaluate_shadow,startparams,args = (self.rv,self.ccf*self.ccf_mask,self.T,self.p,self.aRstar,self.vsys,self.inclination,self.n_components,self.offset)) # alternatively you can do this with closure variables in f if you like
            self.model = evaluate_shadow(result[0],self.rv,self.ccf*0.0,self.T,self.p,self.aRstar,self.vsys,self.inclination,self.n_components,self.offset,leastsq=False)
            # void,matched_ds_model=match_shadow(self.rv,self.ccf,self.ccf_mask,self.dp,self.model,self.maskHW)
            # ut.save_stack('test1.fits',[self.model,matched_ds_model])
            #This appears to be correct now to within a percent?


            self.out_result = result[0]
            self.l_final = result[0][1]
            self.vsini_final = result[0][2]
            print(f'------Fitted lambda and v sin(i) from entire feature: {self.l_final}, {self.vsini_final}')
            self.v_star_fit = fun.local_v_star(self.p,self.aRstar,self.inclination,self.vsini_final,self.l_final)+self.vsys

    def mask_ccf(self):
        self.ccf_mask = mask_ccf_near_RV(self.rv,self.ccf,self.RVp,self.maskHW)

    def save(self, event):
        """
        This is a dictionary that will hold the output (i.e. the fitting parameters
        and all the input values (including phases, vsys, and everything, needed
        to call evaluate_model exactly the way it was saved. The reason that I
        save all that information is because the number of RV points of the ccf to
        which the model is fit may be different from the ccf with which the model
        was initially constructed, but I don't want the removal of the doppler shadow
        to be dependent on that.
        """
        import matplotlib.pyplot as plt
        import pickle
        out_dict = {
            "fit_params": self.out_result,
            # "Transit": self.T,
            # "Phases": self.p,
            # "aRstar": self.aRstar,
            # "vsys": self.vsys,
            # "inclination": self.inclination,
            "n_components": self.n_components,
            "maskHW": self.maskHW}
            # "offset": self.offset}
        with open(self.outpath, 'wb') as f: pickle.dump(out_dict, f)
        plt.close('all')

    def cancel(self,event):
        import sys
        import matplotlib.pyplot as plt
        plt.close('all')
        print("Exiting")
        sys.exit()











def construct_doppler_model(rv,ccf,dp,shadowname,xrange=[-200,200],Nxticks=20.0,Nyticks=10.0):
    """This is the the main function to construct a doppler model. The above are mostly dependencies."""
    import numpy as np
    import matplotlib.pyplot as plt
    import tayph.drag_colour as dcb
    import tayph.functions as fun
    import tayph.system_parameters as sp
    import tayph.plotting as fancyplots
    import sys
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

    #This is for setting plot axes in the call to plotting_scales_2D below.
    nexp = np.shape(ccf)[0]
    yrange=[0,nexp-1]
    y_axis = fun.findgen(nexp)
    #And for adding the planet line:
    vsys = sp.paramget('vsys',dp)
    vsini = sp.paramget('vsini',dp)
    RVp = sp.RV(dp)+vsys
    transit = sp.transit(dp)
    sel_transit = y_axis[(transit < 1.0)]
    transit_start = min(sel_transit)
    transit_end = max(sel_transit)
    fig,ax,cbar = fancyplots.plot_ccf(rv,ccf,dp,xrange = xrange,Nxticks = Nxticks,Nyticks = Nyticks,i_legend=False,show=False)#Initiates the plot for the primer.
    primer = prime_doppler_model(fig,ax,cbar)#We use the active Figure
    #to let the user indicate where the doppler shadow is located (the primer). This calls the plt.show()
    #which was not called by plot_ccf. To proceed, we would like to model the shadow using the
    #primer, along with fancy visualization of the ccf.
    #We first re-instate a plot. This plot is a bit more complex than the primer so we don't
    #use plot_ccf anymore; though the philosophy is similar.

    x2,y2,z,rv_sel,y_sel,xticks,yticks,vmin,vmax = fancyplots.plotting_scales_2D(rv,y_axis,ccf,xrange,yrange,Nxticks=Nxticks,Nyticks=Nyticks,nsigma=3.0)

    #Create empty plots.
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(13,6))
    plt.subplots_adjust(right=0.75)

    #Here we initiate the model instance that does the fitting and handles the GUI.
    #This does an initial fit based on the primer.
    model_callback = fit_doppler_model(fig,ax,rv_sel,z,primer,dp,shadowname)
    s_init = model_callback.maskHW#Masking half-width around the planet RV to start the slider with.

    #We continue by overplotting the velocty traces and transit markers onto the 3 subplots, saving the references to the lines.
    #These lines can be set to be visible/invisible using the checkbuttons below, and are set to be invisible
    #when the plot opens.
    l_planet = []
    t_start = []
    t_end = []
    l_primer = []
    l_vfit = []
    for sub_ax in ax:
        sub_ax.axis([x2.min(),x2.max(),y2.min(),y2.max()])
        sub_ax.set_xticks(xticks)
        sub_ax.set_yticks(yticks)
        sub_ax.set_ylabel('Exposure')
        l1 = sub_ax.plot(RVp,y_axis,'--',color='black',label='Planet rest-frame',visible=False)[0]
        l2 = sub_ax.plot(rv,rv*0.0+transit_start,'--',color='white',label='Transit start',visible=False)[0]
        l3 = sub_ax.plot(rv,rv*0.0+transit_end,'--',color='white',label='Transit end',visible=False)[0]
        l4 = sub_ax.plot(model_callback.v_star_primer,y_axis,'--',color='black',label='Local velocity (primer)',visible=False)[0]
        l5 = sub_ax.plot(model_callback.v_star_fit,y_axis,'--',color='black',label='Local velocity (fit)',visible=False)[0]
        l_planet.append(l1)
        t_start.append(l2)
        t_end.append(l3)
        l_primer.append(l4)
        l_vfit.append(l5)

    ax[0].set_title('Data')
    ax[1].set_title('Model shadow')
    ax[2].set_title('Residual')
    ax[2].set_xlabel('Radial velocity (km/s)')


    #Here we actually plot the initial fit, which will be modified each time the parameters are changed
    #using the GUI buttons/sliders.
    img1=ax[0].pcolormesh(x2,y2,z,vmin=vmin,vmax=vmax,cmap='hot')
    img2=ax[1].pcolormesh(x2,y2,model_callback.model,vmin=vmin,vmax=vmax,cmap='hot')
    img3=ax[2].pcolormesh(x2,y2,z-model_callback.model,vmax=vmax,cmap='hot')
    #This trick to associate a single CB to multiple axes comes from
    #https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    cbar = fig.colorbar(img1, ax=ax.ravel().tolist(),format='%05.4f',aspect = 15)
    cbar = dcb.DraggableColorbar_fits(cbar,[img1,img2,img3],'hot')
    cbar.connect()


    #We define the interface and the bahaviour of button/slider callbacks.
    #First the check buttons for showing the lines defined above.
    rax_top = plt.axes([0.8, 0.65, 0.15, 0.25])
    rax_top.set_title('Plot:')
    labels = ['Planet velocity','Transit start/end','Shadow v$_c$ primer','Shadow $v_c$ fit','Masked area']
    start = [False,False,False,False,False,False]#Start with none of these actually visible.
    check = CheckButtons(rax_top, labels, start)
    def func(label):
        index = labels.index(label)
        lines = [l_planet,np.append(t_end,t_start),l_primer,l_vfit]
        if index < len(lines):
            for l in lines[index]:
                l.set_visible(not l.get_visible())
        if index == len(lines):#I.e. if we are on the last element, which is not a line an option for SHOWING the masked area:
            status = check.get_status()[-1]
            if status == True:#If true, mask the image.
                data = z*model_callback.ccf_mask
                data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot. Makes them white, too.
                img1.set_array((data).ravel())
                img3.set_array((data-model_callback.model).ravel())
            if status == False:#If false (unclicked), then just the data w/o mask.
                img1.set_array(z.ravel())
                img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    check.on_clicked(func)

    #Then the choice for 1 or 2 Gaussian fitting components:
    rax_middle = plt.axes([0.8, 0.53, 0.15, 0.10])
    clabels = ['1 component', '2 components']
    radio = RadioButtons(rax_middle,clabels)
    def cfunc(label):
        index = clabels.index(label)
        model_callback.n_components = index+1
        model_callback.fit_model()#Each time we change the choice, refit.
        status = check.get_status()[-1]
        if status == True:#If true, mask the image.
            data = z*model_callback.ccf_mask
            data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((data-model_callback.model).ravel())
        if status == False:#If false (unclicked), then just the data w/o mask.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    radio.on_clicked(cfunc)


    rax_deg = plt.axes([0.8,0.35,0.07,0.13])
    plt.title('Polynomial degree',fontsize=8)
    rax_poly = plt.axes([0.88,0.35,0.07,0.13])
    dlabels = ['0 (off)','2','4','6']
    plabels = ['Single','Even','Full']
    dradio = RadioButtons(rax_deg,dlabels)
    pradio = RadioButtons(rax_poly,plabels)

    def update_degree(label):
        model_callback.D = dlabels.index(label) * 2
        model_callback.fit_model()#Each time we change the choice, refit.
        status = check.get_status()[-1]
        if status == True:#If true, mask the image.
            data = z*model_callback.ccf_mask
            data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((data-model_callback.model).ravel())
        if status == False:#If false (unclicked), then just the data w/o mask.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        plt.draw()

    def update_poly(label):
        model_callback.S = plabels.index(label)
        model_callback.fit_model()#Each time we change the choice, refit.
        status = check.get_status()[-1]
        if status == True:#If true, mask the image.
            data = z*model_callback.ccf_mask
            data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((data-model_callback.model).ravel())
        if status == False:#If false (unclicked), then just the data w/o mask.
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        plt.draw()

    dradio.on_clicked(update_degree)
    pradio.on_clicked(update_poly)






    #Then the offset slider:
    rax_slider2 = plt.axes([0.8, 0.25, 0.15, 0.02])
    rax_slider2.set_title('Offset 2nd component')
    offset_slider = Slider(rax_slider2,'',-1.0*np.ceil(vsini),np.ceil(vsini),valinit=0.0,valstep=1.0)
    def update_offset(val):
        model_callback.offset = offset_slider.val
        status = radio.value_selected
        if status == clabels[1]:#Only update the fit if we are actually asked to do 2 components.
            model_callback.fit_model()
            # data = z*model_callback.ccf_mask
            # data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs...
            # img1.set_array((data).ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        # if status == False:#If false (unclicked), then just the data w/o mask.
        #     img1.set_array(z.ravel())
        #     img2.set_array(model_callback.model.ravel())
        #     img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    offset_slider.on_changed(update_offset)

    #Then the slider:
    rax_slider = plt.axes([0.8, 0.18, 0.15, 0.02])
    rax_slider.set_title('Mask width (km/s)')
    mask_slider = Slider(rax_slider,'', 0.0,30.0,valinit=s_init,valstep=1.0)
    def update(val):
        model_callback.maskHW = mask_slider.val
        model_callback.mask_ccf()
        model_callback.fit_model()

        status = check.get_status()[-1]
        if status == True:#If true, mask the image.
            data = z*model_callback.ccf_mask
            data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs...
            img1.set_array((data).ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((data-model_callback.model).ravel())
        if status == False:#If false (unclicked), then just the data w/o mask.
            img1.set_array(z.ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    mask_slider.on_changed(update)

    #And finally the save button.
    rax_save = plt.axes([0.875, 0.1, 0.06, 0.05])
    bsave = Button(rax_save, 'Save')
    bsave.on_clicked(model_callback.save)

    rax_cancel = plt.axes([0.8, 0.1, 0.06, 0.05])
    bcancel = Button(rax_cancel, 'Cancel')
    bcancel.on_clicked(model_callback.cancel)
    plt.show()#All fitting is done before this line through event handling.



def construct_gaussian_model(rv,ccf,dp,shadowname,xrange=[-200,200],Nxticks=20.0,Nyticks=10.0):
    #Now we do the whole thing again for pulsations
    # nexp = np.shape(ccf)[0]
    # yrange=[0,nexp-1]
    # y_axis = fun.findgen(nexp)
    # #And for adding the planet line:
    # vsys = sp.paramget('vsys',dp)
    # vsini = sp.paramget('vsini',dp)
    # RVp = sp.RV(dp)+vsys
    # transit = sp.transit(dp)
    # sel_transit = y_axis[[transit < 1.0]]
    # transit_start = min(sel_transit)
    # transit_end = max(sel_transit)



    fig,ax,cbar = fancyplots.plot_ccf(rv,ccf,dp,xrange = xrange,Nxticks = Nxticks,Nyticks = Nyticks,i_legend=False,show=False)#Initiates the plot for the primer.
    #primer = prime_doppler_model(fig,ax,cbar)#We use the active Figure
    #to let the user indicate where the doppler shadow is located (the primer). This calls the plt.show()
    #which was not called by plot_ccf. To proceed, we would like to model the shadow using the
    #primer, along with fancy visualization of the ccf.
    #We first re-instate a plot. This plot is a bit more complex than the primer so we don't
    #use plot_ccf anymore; though the philosophy is similar.

    x2,y2,z,rv_sel,y_sel,xticks,yticks,vmin,vmax = fancyplots.plotting_scales_2D(rv,y_axis,ccf,xrange,yrange,Nxticks=Nxticks,Nyticks=Nyticks,nsigma=3.0)

    #Create empty plots.
    fig,ax = plt.subplots(3,1,sharex=True,figsize=(13,6))
    plt.subplots_adjust(right=0.75)

    #Here we initiate the model instance that does the fitting and handles the GUI.
    #This does an initial fit based on the primer.
    model_callback = fit_pulsations(fig,ax,rv_sel,z,dp,shadowname)
    s_init = model_callback.maskHW#Masking half-width around the planet RV to start the slider with.

    #We continue by overplotting the velocty traces and transit markers onto the 3 subplots, saving the references to the lines.
    #These lines can be set to be visible/invisible using the checkbuttons below, and are set to be invisible
    #when the plot opens.
    l_planet = []
    t_start = []
    t_end = []
    l_primer = []
    l_vfit = []
    for sub_ax in ax:
        sub_ax.axis([x2.min(),x2.max(),y2.min(),y2.max()])
        sub_ax.set_xticks(xticks)
        sub_ax.set_yticks(yticks)
        sub_ax.set_ylabel('Exposure')
        l1 = sub_ax.plot(RVp,y_axis,'--',color='black',label='Planet rest-frame',visible=False)[0]
        l2 = sub_ax.plot(rv,rv*0.0+transit_start,'--',color='white',label='Transit start',visible=False)[0]
        l3 = sub_ax.plot(rv,rv*0.0+transit_end,'--',color='white',label='Transit end',visible=False)[0]
        l4 = sub_ax.plot(model_callback.v_star_primer,y_axis,'--',color='black',label='Local velocity (primer)',visible=False)[0]
        l5 = sub_ax.plot(model_callback.v_star_fit,y_axis,'--',color='black',label='Local velocity (fit)',visible=False)[0]
        l_planet.append(l1)
        t_start.append(l2)
        t_end.append(l3)
        l_primer.append(l4)
        l_vfit.append(l5)

    ax[0].set_title('Data')
    ax[1].set_title('Model shadow')
    ax[2].set_title('Residual')
    ax[2].set_xlabel('Radial velocity (km/s)')


    #Here we actually plot the initial fit, which will be modified each time the parameters are changed
    #using the GUI buttons/sliders.
    img1=ax[0].pcolormesh(x2,y2,z,vmin=vmin,vmax=vmax,cmap='hot')
    img2=ax[1].pcolormesh(x2,y2,model_callback.model,vmin=vmin,vmax=vmax,cmap='hot')
    img3=ax[2].pcolormesh(x2,y2,z-model_callback.model,vmax=vmax,cmap='hot')
    #This trick to associate a single CB to multiple axes comes from
    #https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    cbar = fig.colorbar(img1, ax=ax.ravel().tolist(),format='%05.4f',aspect = 15)
    cbar = dcb.DraggableColorbar_fits(cbar,[img1,img2,img3],'hot')
    cbar.connect()


    #We define the interface and the bahaviour of button/slider callbacks.
    #First the check buttons for showing the lines defined above.
    rax_top = plt.axes([0.8, 0.65, 0.15, 0.25])
    rax_top.set_title('Plot:')
    labels = ['Planet velocity','Transit start/end','Shadow v$_c$ primer','Shadow $v_c$ fit','Masked area']
    start = [False,False,False,False,False,False]#Start with none of these actually visible.
    check = CheckButtons(rax_top, labels, start)
    def func(label):
        index = labels.index(label)
        lines = [l_planet,np.append(t_end,t_start),l_primer,l_vfit]
        if index < len(lines):
            for l in lines[index]:
                l.set_visible(not l.get_visible())
        if index == len(lines):#I.e. if we are on the last element, which is not a line an option for SHOWING the masked area:
            status = check.get_status()[-1]
            if status == True:#If true, mask the image.
                data = z*model_callback.ccf_mask
                data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot. Makes them white, too.
                img1.set_array((data).ravel())
                img3.set_array((data-model_callback.model).ravel())
            if status == False:#If false (unclicked), then just the data w/o mask.
                img1.set_array(z.ravel())
                img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    check.on_clicked(func)

    #Then the choice for 1 or 2 Gaussian fitting components:
    rax_middle = plt.axes([0.8, 0.45, 0.15, 0.15])
    clabels = ['1 component', '2 components']
    radio = RadioButtons(rax_middle,clabels)
    def cfunc(label):
            index = clabels.index(label)
            model_callback.n_components = index+1
            model_callback.fit_model()#Each time we change the choice, refit.
            status = check.get_status()[-1]
            if status == True:#If true, mask the image.
                data = z*model_callback.ccf_mask
                data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs, set them to inf instead for the plot.
                img2.set_array(model_callback.model.ravel())
                img3.set_array((data-model_callback.model).ravel())
            if status == False:#If false (unclicked), then just the data w/o mask.
                img2.set_array(model_callback.model.ravel())
                img3.set_array((z-model_callback.model).ravel())
            plt.draw()
    radio.on_clicked(cfunc)

    #Then the offset slider:
    rax_slider2 = plt.axes([0.8, 0.35, 0.15, 0.02])
    rax_slider2.set_title('Offset 2nd component')
    offset_slider = Slider(rax_slider2,'',-1.0*np.ceil(vsini),np.ceil(vsini),valinit=0.0,valstep=1.0)
    def update_offset(val):
        model_callback.offset = offset_slider.val
        status = radio.value_selected
        if status == clabels[1]:#Only update the fit if we are actually asked to do 2 components.
            model_callback.fit_model()
            # data = z*model_callback.ccf_mask
            # data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs...
            # img1.set_array((data).ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        # if status == False:#If false (unclicked), then just the data w/o mask.
        #     img1.set_array(z.ravel())
        #     img2.set_array(model_callback.model.ravel())
        #     img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    offset_slider.on_changed(update_offset)

    #Then the slider:
    rax_slider = plt.axes([0.8, 0.28, 0.15, 0.02])
    rax_slider.set_title('Mask width (km/s)')
    mask_slider = Slider(rax_slider,'', 0.0,30.0,valinit=s_init,valstep=1.0)
    def update(val):
        model_callback.maskHW = mask_slider.val
        model_callback.mask_ccf()
        model_callback.fit_model()

        status = check.get_status()[-1]
        if status == True:#If true, mask the image.
            data = z*model_callback.ccf_mask
            data[np.isnan(data)] = np.inf#The colobar doesn't eat NaNs...
            img1.set_array((data).ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((data-model_callback.model).ravel())
        if status == False:#If false (unclicked), then just the data w/o mask.
            img1.set_array(z.ravel())
            img2.set_array(model_callback.model.ravel())
            img3.set_array((z-model_callback.model).ravel())
        plt.draw()
    mask_slider.on_changed(update)

    #And finally the save button.
    rax_save = plt.axes([0.875, 0.1, 0.06, 0.05])
    bsave = Button(rax_save, 'Save')
    bsave.on_clicked(model_callback.save)

    rax_cancel = plt.axes([0.8, 0.1, 0.06, 0.05])
    bcancel = Button(rax_cancel, 'Cancel')
    bcancel.on_clicked(model_callback.cancel)
    plt.show()#All fitting is done before this line through event handling.
