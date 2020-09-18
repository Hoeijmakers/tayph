#This package contains all the routines that allow the user to mask pixels and columns
#out of the list of spectral orders. The main functionality is wrapped into mask_orders()
#which is defined at the bottom. Mask_orders() is the highest-level thing that's called.
#It does masking of all spectral orders in two steps (depending on which functionality was
#requested by the user when calling it):
#-An automatic sigma clipping.
#-Manual selection of columns using a GUI.

#The masked areas from these two operations are saved separately in the data folder.
#and can be loaded/altered upon new calls/passes through run.py.
#Most of the routines below are related to making the GUI work.

__all__ = [
    "manual_masking",
    "apply_mask_from_file",
    "mask_orders",
    "load_columns_from_file",
    "write_columns_to_file",
    "interpolate_over_NaNs",
    "mask_maker"
]



def interpolate_over_NaNs(list_of_orders,cutoff=0.2):
    #This is a helper function I had to dump here that is mostly unrelated to the GUI,
    #but with healing NaNs. If there are too many NaNs in a column, instead of
    #interpolating, just set the entire column to NaN. If an entire column is set to NaN,
    #it doesn't need to be healed because the cross-correlation never sees it, and the pixel
    #never contributes. It becomes like the column is beyond the edge of the wavelength range of the data.
    import numpy as np
    import tayph.functions as fun
    from tayph.vartests import typetest
    import astropy.io.fits as fits
    """
    This function loops through a list of orders, over the individual
    spectra in each order, and interpolates over the NaNs. It uses the manual provided at
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    which I duplicated in tayph.functions.

    Parameters
    ----------
    list_of_orders : list
        The list of 2D orders for which NaNs need to be removed.

    cutoff : float
        If a column contains more NaNs than this value times its length, instead
        of interpolating over those NaNs, the entire column is set to NaN.

    Returns
    -------
    list_of_healed_orders : list
        The corrected 2D orders.
    """

    typetest(cutoff,float,'cutoff in masking.interpolate_over_NaNs()',)
    if cutoff <= 0  or cutoff > 1:
        raise Exception('Runtime Error in interpolate_over_NaNs: cutoff should be between 0 or 1 (not including 0).')

    N = len(list_of_orders)
    if N == 0:
        raise Exception('Runtime Error in interpolate_over_NaNs: List of orders is empty.')

    # N_nans_total = 0
    N_nans_columns = 0
    N_nans_isolated = 0
    N_healed = 0
    N_pixels = 0
    list_of_healed_orders = []
    for i in range(len(list_of_orders)):
        order = list_of_orders[i]*1.0#x1 to copy it, otherwise the input is altered backwardly.
        shape  = np.shape(order)
        nexp = shape[0]
        npx = shape[1]
        N_pixels += nexp*npx
        list_of_masked_columns = []#This will contain the column numbers to mask completely at the end.
        if np.sum(np.isnan(order)) > 0:
            # N_nans_total+=np.sum(np.isnan(order))
            #So this order contains NaNs.
            #First we loop over all columns to try to find columns where the number
            #of NaNs is greater than CUTOFF.
            for j in range(npx):
                column = order[:,j]
                N_nans = np.sum(np.isnan(column))
                if N_nans > cutoff*nexp:
                    list_of_masked_columns.append(j)
                    N_nans_columns+=nexp
                else:
                    N_nans_isolated+=N_nans
            # fits.writeto('test.fits',order,overwrite=True)
            for k in range(nexp):
                spectrum = order[k,:]
                nans,x= fun.nan_helper(spectrum)
                if np.sum(nans) > 0:
                    spectrum_healed = spectrum*1.0
                    #There are nans in this spectrum.
                    N_healed += np.sum(nans)
                    if len(x(~nans)) > 0:
                        spectrum_healed[nans]= np.interp(x(nans), x(~nans), spectrum[~nans])
                        #This heals all the NaNs, including the ones in all-NaN columns.
                        #These will be set back to NaN below.
                    else:#This happens if an entire order is masked.
                        spectrum_healed[nans]=0
                    order[k,:] = spectrum_healed

        if len(list_of_masked_columns) > 0:
            for l in list_of_masked_columns:
                order[:,l]+=np.nan#Set the ones that were erroneously healed back to nan.
        list_of_healed_orders.append(order)
    print(f'------Total number of pixels in {N} orders: {N_pixels}')
    print(f'------Number of NaNs in columns identified as bad (or previously masked): {N_nans_columns} ({np.round(N_nans_columns/N_pixels*100,2)}% of total)')
    print(f'------Number of NaNs in isolated pixels: {N_nans_isolated} ({np.round(N_nans_isolated/N_pixels*100,2)}% of total)')
    print(f'------Number of bad pixels identified: {N_nans_isolated+N_nans_columns} ({np.round((N_nans_isolated+N_nans_columns)/N_pixels*100,2)}% of total)')
    return(list_of_healed_orders)



class mask_maker(object):
    #This is my third home-made class: A GUI for masking pixels in the spectrum.
    def __init__(self,list_of_wls,list_of_orders,list_of_saved_selected_columns,Nxticks,Nyticks,nsigma=3.0):
        """
        We initialize with a figure object, three axis objects (in a list)
        the wls, the orders, the masks already made; and we do the first plot.

        NOTICE: Anything that is potted in these things as INF actually used to be
        a NaN that was masked out before.
        """
        import numpy as np
        import pdb
        import tayph.functions as fun
        import tayph.plotting as plotting
        import tayph.drag_colour as dcb
        import matplotlib.pyplot as plt
        import itertools
        from matplotlib.widgets import MultiCursor
        import tayph.util as ut
        from tayph.vartests import typetest,postest
        import copy
        #Upon initialization, we raise the keywords onto self.
        self.N_orders = len(list_of_wls)
        if len(list_of_wls) < 1 or len(list_of_orders) < 1:# or len(list_of_masks) <1:
            raise Exception('Runtime Error in mask_maker init: lists of WLs, orders and/or masks have less than 1 element.')
        if len(list_of_wls) != len(list_of_orders):# or len(list_of_wls) != len(list_of_masks):
            raise Exception('Runtime Error in mask_maker init: List of wls and list of orders have different length (%s & %s).' % (len(list_of_wls),len(list_of_orders)))
        typetest(Nxticks,int,'Nxticks in mask_maker init',)
        typetest(Nyticks,int,'Nyticks in mask_maker init',)
        typetest(nsigma,float,'Nsigma in mask_maker init',)
        postest(Nxticks,varname='Nxticks in mask_maker init')
        postest(Nyticks,varname='Nyticks in mask_maker init')
        postest(nsigma,varname='Nsigma in mask_maker init')

        self.list_of_wls = list_of_wls
        self.list_of_orders = list_of_orders
        self.list_of_selected_columns = list(list_of_saved_selected_columns)
        #Normally, if there are no saved columns to load, list_of_saved_selected_columns is an empty list. However if
        #it is set, then its automatically loaded into self.list_of_selected_columns upon init.
        #Below there is a check to determine whether it was empty or not, and whether the list of columns
        #has the same length as the list of orders.
        if len(self.list_of_selected_columns) == 0:
            for i in range(self.N_orders):
                self.list_of_selected_columns.append([])#Make a list of empty lists.
                    #This will contain all columns masked by the user, on top of the things
                    #that are already masked by the program.
        else:
            if len(self.list_of_selected_columns) != self.N_orders:
                raise Exception('Runtime Error in mask_maker init: Trying to restore previously saved columns but the number of orders in the saved column file does not match the number of orders provided.')
            print('------Restoring previously saved columns in mask-maker')


        #All checks are now complete. Lets prepare to do the masking.
        # self.N = min([56,self.N_orders-1])#We start on order 56, or the last order if order 56 doesn't exist.
        self.N=0
        #Set the current active order to order , and calculate the meanspec
        #and residuals to be plotted, which are saved in self.
        self.set_order(self.N)


        #Sorry for the big self.spaghetti of code. This initializes the plot.
        #Functions and vars further down in the class will deal with updating the plots
        #as buttons are pressed. Some of this is copied from the construct_doppler_model
        #function; but this time I made it part of the class.
        #First define plotting and axis parameters for the colormesh below.
        self.Nxticks = Nxticks
        self.Nyticks = Nyticks
        self.nsigma = nsigma
        self.xrange = [0,self.npx-1]
        self.yrange=[0,self.nexp-1]
        self.x_axis=fun.findgen(self.npx).astype(int)
        self.y_axis = fun.findgen(self.nexp).astype(int)
        self.x2,self.y2,self.z,self.wl_sel,self.y_axis_sel,self.xticks,self.yticks,void1,void2= plotting.plotting_scales_2D(self.x_axis,self.y_axis,self.residual,self.xrange,self.yrange,Nxticks=self.Nxticks,Nyticks=self.Nyticks,nsigma=self.nsigma)
        self.fig,self.ax = plt.subplots(3,1,sharex=True,figsize=(14,6))#Initialize the figure and 3 axes.
        plt.subplots_adjust(left=0.05)#Make them more tight, we need all the space we can get.
        plt.subplots_adjust(right=0.85)

        self.ax[0].set_title('Spectral order %s  (%s - %s nm)' % (self.N,round(np.min(self.wl),1),round(np.max(self.wl),1)))
        self.ax[1].set_title('Residual of time-average')
        self.ax[2].set_title('Time average 1D spectrum')

        array1 = copy.deepcopy(self.order)
        array2 = copy.deepcopy(self.residual)
        array1[np.isnan(array1)] = np.inf#The colobar doesn't eat NaNs, so now set them to inf just for the plot.
        array2[np.isnan(array2)] = np.inf#And here too.
        #The previous three lines are repeated in self.update_plots()
        self.img1=self.ax[0].pcolormesh(self.x2,self.y2,array1,vmin=0,vmax=self.img_max,cmap='hot')
        self.img2=self.ax[1].pcolormesh(self.x2,self.y2,array2,vmin=self.vmin,vmax=self.vmax,cmap='hot')
        self.img3=self.ax[2].plot(self.x_axis,self.meanspec)
        self.ax[2].set_xlim((min(self.x_axis),max(self.x_axis)))
        self.ax[2].set_ylim(0,self.img_max)
        #This trick to associate a single CB to multiple axes comes from
        #https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
        self.cbar = self.fig.colorbar(self.img2, ax=self.ax.ravel().tolist(),aspect = 20)
        self.cbar = dcb.DraggableColorbar_fits(self.cbar,[self.img2],'hot')
        self.cbar.connect()

        #The rest is for dealing with the masking itself; the behaviour of the
        #add/subtact buttons, the cursor and the saving of the masked columns.
        self.col_active = ['coral','mistyrose']#The colours for the ADD and SUBTRACT buttons that
        #can be activated.
        self.col_passive = ['lightgrey','whitesmoke']#Colours when they are not active.
        self.MW = 50#The default masking width.
        self.addstatus = 0#The status for adding-to-mask mode starts as zero; i.e. it starts inactive.
        self.substatus = 0#Same for subtraction.
        self.list_of_polygons = []#This stores the polygons that are currently plotted. When initializing, none are plotted.
        #However when proceeding through draw_masked_areas below, this variable could become populated if previously selected
        #column were loaded from file.
        self.multi = MultiCursor(self.fig.canvas, (self.ax[0],self.ax[1],self.ax[2]), color='g', lw=1, horizOn=False, vertOn=True)
        self.multi.set_active(False)#The selection cursor starts deactivated as well, and is activated and deactivated
        #further down as the buttons are pressed.
        self.apply_to_all = False
        self.apply_to_all_future = False

        #The following show the z value of the plotted arrays in the statusbar,
        #taken from https://matplotlib.org/examples/api/image_zcoord.html
        numrows, numcols = self.order.shape
        def format_coord_order(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = self.order[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        def format_coord_res(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = self.residual[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        self.ax[0].format_coord = format_coord_order
        self.ax[1].format_coord = format_coord_res
        self.draw_masked_areas()
        #This is the end of init.


    def draw_masked_areas(self):
        """
        This function draws green boxes onto the three plots corresponding to
        which columns where masked.
        """
        import matplotlib.pyplot as plt

        def plot_span(min,max):#This is a shorthand for drawing the polygons in the same style on all subplots.
            for subax in self.ax:#There are 3 ax objects in this list.
                self.list_of_polygons.append(subax.axvspan(min,max,color='green',alpha=0.5))

        #Start by removing any polygons that were saved by earlier calls to this
        #function after switching orders. Everything needs to be redrawn each time
        #a new order is selected, i.e. each time this function is called.
        if len(self.list_of_polygons) > 0:#If blocks are already plotted...
            for i in self.list_of_polygons:
                i.remove()#emtpy the list.
            self.list_of_polygons = []

        #Select the columns defined by the select-columns events in the add and subtract subroutines.
        columns = self.list_of_selected_columns[self.N]
        if len(columns) > 0:
            columns.sort()
            min = columns[0]#start by opening a block
            for i in range(1,len(columns)-1):
                dx = columns[i] - columns[i-1]
                if dx > 1:#As long as dx=1, we are passing through adjacently selected columns.
                #Only do something if dx>1, in which case we need to end the block and start a new one.
                    max=columns[i-1]#then the previous column was the last element of the block
                    plot_span(min,max)
                    min=columns[i]#Begin a new block
            #at the end, finish the last block:
            max = columns[-1]
            plot_span(min,max)




    def set_order(self,i):
        """
        This modifies the currently active order to be plotted.
        """
        import numpy as np
        import tayph.functions as fun
        import warnings
        import tayph.util as ut
        import copy
        from tayph.vartests import typetest
        typetest(i,int,'i in mask_maker.set_order()',)

        self.wl = self.list_of_wls[i]
        self.order = self.list_of_orders[i]

        #Measure the shape of the current order
        self.nexp = np.shape(self.order)[0]
        self.npx = np.shape(self.order)[1]

        #Compute the meanspec and the residuals, ignoring runtime warnings related to NaNs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.meanspec = np.nanmean(self.list_of_orders[i],axis=0)
            self.residual = self.order / self.meanspec
        self.img_max = np.nanmean(self.meanspec[fun.selmax(self.meanspec,0.02,s=0.02)])*1.3
        self.vmin = np.nanmedian(self.residual)-3.0*np.nanstd(self.residual)
        self.vmax = np.nanmedian(self.residual)+3.0*np.nanstd(self.residual)


    def exit_add_mode(self):
        """
        This exits column-addition mode of the interface.
        This is a separate function that can be called on 3 occasions:
        When pressing the Mask button for the second time, when pressing the
        subtract button while in Mask mode, and when exiting the GUI.
        """
        self.multi.set_active(False)#This is the green vertical line. Its not shown when this mode is off.
        self.fig.canvas.mpl_disconnect(self.click_connector)
        self.addstatus = 0
        self.badd.color=self.col_passive[0]
        self.badd.hovercolor=self.col_passive[1]
        self.fig.canvas.draw()
        print('---------Exited add mode')

    def exit_sub_mode(self):
        """
        This exits column-subtraction mode of the interface.
        This is a separate function that can be called on 3 occasions:
        When pressing the Mask button for the second time, when pressing the
        subtract button while in Mask mode, and when exiting the GUI.
        """
        self.multi.set_active(False)#This is the green vertical line. Its not shown when this mode is off.
        self.fig.canvas.mpl_disconnect(self.click_connector)
        self.substatus = 0
        self.bsub.color=self.col_passive[0]
        self.bsub.hovercolor=self.col_passive[1]
        self.fig.canvas.draw()
        print('---------Exited sub mode')

    def add(self,event):
        """
        This is an event handler for pressing the Mask button in the GUI.
        It has 2 behaviours depending on whether the button was pressed before or not.
        If it was not pressed, addstatus == 0 and it will enter mask mode.
        If it was pressed, the GUI is in mask mode, addstatus == 1 and instead it will
        leave mask mode upon pressing it. Addstatus then becomes zero and the thing starts over.

        When in mask mode, the user can click on any of the three subplots to
        select columns that he/she wants to be masked. Exiting mask mode deactivates
        this behaviour.
        """

        def add_columns(event):
            #This handles with a mouseclick in either of the three plots while in add mode.
            if event.inaxes in [self.ax[0],self.ax[1],self.ax[2]]:#Check that it occurs in one of the subplots.
                ci = event.xdata*1.0#xdata is the column that is selected.
                selmin = max([int(ci-0.5*self.MW),0])#Select all columns in a range of self.HW from that click.
                selmax = min([int(ci+0.5*self.MW),self.npx-1])
                sel = self.x_axis[selmin:selmax]
                if self.apply_to_all == True:
                    for o in range(self.N_orders):#Loop through all orders.
                        for i in sel:#Add the selected range to the list of this order.
                            self.list_of_selected_columns[o].append(i)
                        self.list_of_selected_columns[o]=list(set(self.list_of_selected_columns[o]))#Remove duplicates
                elif self.apply_to_all_future == True:
                    for o in range(self.N,self.N_orders):#Loop through all orders greater than this one.
                        for i in sel:#Add the selected range to the list of this order.
                            self.list_of_selected_columns[o].append(i)
                        self.list_of_selected_columns[o]=list(set(self.list_of_selected_columns[o]))#Remove duplicates
                else:#If apply to all (future) is false:
                    for i in sel:#Add the selected range to the list of this order.
                        self.list_of_selected_columns[self.N].append(i)
                    self.list_of_selected_columns[self.N]=list(set(self.list_of_selected_columns[self.N]))#Remove duplicates

                self.draw_masked_areas()#Update the green areas.
                self.fig.canvas.draw_idle()

        if self.addstatus == 0:
            if self.substatus == 1:
                self.exit_sub_mode()
            print('---------Entered adding mode')
            self.addstatus=1
            self.badd.color=self.col_active[0]
            self.badd.hovercolor=self.col_active[1]
            self.fig.canvas.draw()
            self.multi.set_active(True)#This is the green vertical line. Its shown only when this mode is on.
            self.click_connector = self.fig.canvas.mpl_connect('button_press_event', add_columns)#This is the connector that registers clicks
            #on the plot. Not clicks on the buttons!
        else:
            self.exit_add_mode()

    def subtract(self,event):
        """
        Similar to add(), this is an event handler for pressing the Unmask button in the GUI.
        It has 2 behaviours depending on whether the button was pressed before or not.
        If it was not pressed, substatus == 0 and it will enter unmask mode.
        If it was pressed, the GUI is in unmask mode, substatus == 1 and instead it will
        leave unmask mode upon pressing it. Substatus then becomes zero and the thing starts over.

        When in unmask mode, the user can click on any of the three subplots to
        select columns that he/she wants to be removed from the list of masked columns.
        Exiting mask mode deactivates this behaviour.
        """
        def remove_columns(event):
            if event.inaxes in [self.ax[0],self.ax[1],self.ax[2]]:#Check that it occurs in one of the subplots.
                ci = event.xdata*1.0
                selmin = max([int(ci-0.5*self.MW),0])
                selmax = min([int(ci+0.5*self.MW),self.npx-1])
                sel = self.x_axis[selmin:selmax]
                if self.apply_to_all == True:
                    for o in range(self.N_orders):#Loop through all orders.
                        for i in sel:
                            if i in self.list_of_selected_columns[o]:
                                self.list_of_selected_columns[o].remove(i)
                elif self.apply_to_all_future == True:
                    for o in range(self.N,self.N_orders):#Loop through all orders.
                        for i in sel:
                            if i in self.list_of_selected_columns[o]:
                                self.list_of_selected_columns[o].remove(i)
                else:#If apply_to_all is false
                    for i in sel:
                        if i in self.list_of_selected_columns[self.N]:
                            self.list_of_selected_columns[self.N].remove(i)

                self.draw_masked_areas()
                self.fig.canvas.draw_idle()
                #End of remove_columns subroutine.


        if self.substatus == 0:
            if self.addstatus == 1:
                self.exit_add_mode()
            print('---------Entered subtraction mode')
            self.substatus=1
            self.bsub.color=self.col_active[0]
            self.bsub.hovercolor=self.col_active[1]
            self.fig.canvas.draw()
            self.multi.set_active(True)
            self.click_connector = self.fig.canvas.mpl_connect('button_press_event', remove_columns)
        else:
            self.exit_sub_mode()

    def applyall(self,event):
        """
        This is an event handler for pressing the Apply All button in the GUI.
        It has 2 behaviours depending on whether the button was pressed before or not.
        If it was not pressed, apply_to_all == 0 and it will enter apply-to-all mode.
        If it was pressed, the GUI is in apply-to-all mode, apply_to_all == 1 and instead it will
        switch it to zero upon pressing it. Apply_to_all then becomes zero and the thing starts over.
        """
        if self.apply_to_all == False:
            print('---------Entering Apply-to-all mode')
            self.apply_to_all = True
            self.ball.color=self.col_active[0]
            self.ball.hovercolor=self.col_active[1]
        else:
            print('---------Exiting Apply-to-all mode')
            self.apply_to_all = False
            self.ball.color=self.col_passive[0]
            self.ball.hovercolor=self.col_passive[1]
        if self.apply_to_all_future == True:
            print('---------Exiting Apply-to-all-future mode')
            self.apply_to_all_future = False
            self.ballf.color=self.col_passive[0]
            self.ballf.hovercolor=self.col_passive[1]
        self.fig.canvas.draw()
        self.fig.canvas.draw()

    def applyallfuture(self,event):
        """
        This is an event handler for pressing the Apply All Future button in the GUI.
        It has 2 behaviours depending on whether the button was pressed before or not.
        If it was not pressed, apply_to_all == 0 and it will enter apply-to-all mode.
        If it was pressed, the GUI is in apply-to-all mode, apply_to_all == 1 and instead it will
        switch it to zero upon pressing it. Apply_to_all then becomes zero and the thing starts over.
        """
        if self.apply_to_all_future == False:
            print('---------Entering Apply-to-all-future mode')
            self.apply_to_all_future = True
            self.ballf.color=self.col_active[0]
            self.ballf.hovercolor=self.col_active[1]
        else:
            print('---------Exiting Apply-to-all-future mode')
            self.apply_to_all_future = False
            self.ballf.color=self.col_passive[0]
            self.ballf.hovercolor=self.col_passive[1]
        if self.apply_to_all == True:
            print('---------Exiting Apply-to-all mode')
            self.apply_to_all = False
            self.ball.color=self.col_passive[0]
            self.ball.hovercolor=self.col_passive[1]
        self.fig.canvas.draw()
        self.fig.canvas.draw()


    def applytotal(self,event):
        """
        This is an event handler for pressing the Mask entire order button in the GUI.
        It has 2 behaviours depending on whether anything is masked or not.
        If nothing was masked, the entire order will be masked.
        If something was, it will all be unmasked.
        """
        import tayph.functions as fun
        ncol = len(self.list_of_orders[self.N][0])
        self.list_of_selected_columns[self.N] = list(fun.findgen(ncol))
        # print(self.list_of_selected_columns[self.N])
        self.draw_masked_areas()#Update the green areas.
        self.fig.canvas.draw()
        self.fig.canvas.draw()

    def cleartotal(self,event):
        """This is an event handler for pressing the Unmask entire order button in the GUI.
        It has 2 behaviours depending on whether anything is masked or not.
        If nothing was masked, the entire order will be masked.
        If something was, it will all be unmasked.
        """
        self.list_of_selected_columns[self.N] = []
        # print(self.list_of_selected_columns[self.N])
        self.draw_masked_areas()#Update the green areas.
        self.fig.canvas.draw()
        self.fig.canvas.draw()

    def previous(self,event):
        """
        The button to go to the previous order.
        """
        self.N -= 1
        if self.N <0:#If order tries to become less than zero, loop to the highest order.
            self.N = len(self.list_of_orders)-1
        self.set_order(self.N)
        self.mask_slider.set_val(self.N)#Update the slider value.
        self.update_plots()#Redraw everything.

    def next(self,event):
        """
        The button to go to the next order. Similar to previous().
        """
        self.N += 1
        if self.N > len(self.list_of_orders)-1:
            self.N = 0
        self.set_order(self.N)
        self.mask_slider.set_val(self.N)
        self.update_plots()

    def cancel(self,event):
        """
        This is a way to crash hard out of the python interpreter.
        """
        import sys
        print('------Canceled by user')
        sys.exit()

    def save(self,event):
        """
        The "save" button is actually only closing the plot. Actual saving of things
        happens below plt.show() below.
        """
        import matplotlib.pyplot as plt
        plt.close(self.fig)

    def update_plots(self):
        """
        This redraws the plot planels, taking care to reselect axis ranges and such.
        """
        import numpy as np
        import tayph.drag_colour as dcb
        import tayph.functions as fun
        import copy
        array1 = copy.deepcopy(self.order.ravel())
        array2 = copy.deepcopy(self.residual.ravel())
        array1[np.isnan(array1)] = np.inf#The colobar doesn't eat NaNs, so now set them to inf just for the plot.
        array2[np.isnan(array2)] = np.inf#And here too.

        self.img1.set_array(array1)
        self.img1.set_clim(vmin=0,vmax=self.img_max)
        self.img2.set_array(array2)
        self.img2.set_clim(vmin=self.vmin,vmax=self.vmax)
        self.img3[0].set_ydata(self.meanspec)
        self.ax[0].set_title('Spectral order %s  (%s - %s nm)' % (self.N,round(np.min(self.wl),1),round(np.max(self.wl),1)))
        self.ax[2].set_ylim(0,self.img_max)
        self.draw_masked_areas()
        self.fig.canvas.draw_idle()

    def slide_order(self,event):
        """
        This handles the order-selection slide bar.
        """
        self.N = int(self.mask_slider.val)
        self.set_order(self.N)
        self.update_plots()

    def slide_maskwidth(self,event):
        """
        This handles the mask-width selection slide bar.
        """
        self.MW = int(self.MW_slider.val)




def manual_masking(list_of_wls,list_of_orders,list_of_masks,Nxticks = 20,Nyticks = 10,saved = []):
    import numpy as np
    import matplotlib.pyplot as plt
    import pdb
    import tayph.drag_colour as dcb
    import tayph.util as ut
    from tayph.vartests import typetest,postest,dimtest
    import tayph.functions as fun
    import sys
    from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
    """
    This brings the user into a GUI in which he/she/they can both visualize
    spectral orders and mask regions of bad data; such as occur at edges
    of orders, inside deep spectral lines (stellar or telluric) or other places.

    In the standard workflow, this routine succeeds an automatic masking step that has
    performed a sigma clipping using a rolling standard deviation. This procedure has
    added a certain number of pixels to the mask. If this routine is subsequently
    called, the user can mask out bad *columns* by selecting them in the GUI. These
    are then added to that mask as well.
    """


    typetest(Nxticks,int,'Nxticks in manual_masking()')
    typetest(Nyticks,int,'Nyticks in manual_masking()')
    postest(Nxticks,'Nxticks in manual_masking()')
    postest(Nyticks,'Nyticks in manual_masking()')
    typetest(list_of_wls,list,'list_of_wls in manual_masking()')
    typetest(list_of_orders,list,'list_of_orders in manual_masking()')
    typetest(list_of_masks,list,'list_of_masks in manual_masking()')
    typetest(saved,list,'saved list_of_masks in manual_masking()')
    dimtest(list_of_wls,[0,0],'list_of_wls in manual_masking()')
    dimtest(list_of_orders,[len(list_of_wls),0,0],'list_of_orders in manual_masking()')
    dimtest(list_of_masks,[len(list_of_wls),0,0],'list_of_masks in manual_masking()')


    print('------Entered manual masking mode')

    M = mask_maker(list_of_wls,list_of_orders,saved,Nxticks=Nxticks,Nyticks=Nyticks,nsigma=3.0) #Mask callback
    #This initializes all the parameters of the plot. Which order it is plotting, what
    #dimensions these have, the actual data arrays to plot; etc. Initialises on order 56.
    #I also dumped most of the buttons and callbacks into there; so this thing
    #is probably a bit of a sphaghetti to read.

    #Here we only need to define the buttons and add them to the plot. In fact,
    #I could probably have shoved most of this into the class as well, (see
    #me defining all these buttons as class attributes? But, I
    #suppose that the structure is a bit more readable this way.


    #The button to add a region to the mask:
    rax_sub = plt.axes([0.8, 0.4, 0.14, 0.05])
    M.bsub = Button(rax_sub, ' Unmask columns ')
    M.bsub.on_clicked(M.subtract)

    #The button to add a region to the mask:
    rax_add = plt.axes([0.8, 0.48, 0.14, 0.05])
    M.badd = Button(rax_add, ' Mask columns ')
    M.badd.color=M.col_passive[0]
    M.badd.hovercolor=M.col_passive[1]
    M.add_connector = M.badd.on_clicked(M.add)

    #The button to add a region to the mask:
    rax_all = plt.axes([0.8, 0.56, 0.14, 0.05])
    M.ball = Button(rax_all, ' Apply to all ')
    M.ball.on_clicked(M.applyall)

    rax_future = plt.axes([0.8, 0.64, 0.14, 0.05])
    M.ballf = Button(rax_future, ' Apply to all above')
    M.ballf.on_clicked(M.applyallfuture)

    rax_clearentire = plt.axes([0.8, 0.72, 0.14, 0.05])
    M.bcent = Button(rax_clearentire, ' Unmask order ')
    M.bcent.on_clicked(M.cleartotal)

    #The button to add a region to the mask:
    rax_entire = plt.axes([0.8, 0.8, 0.14, 0.05])
    M.bent = Button(rax_entire, ' Mask order ')
    M.bent.on_clicked(M.applytotal)



    # rax_atv = plt.axes([0.9, 0.45, 0.1, 0.15])
    # clabels = ['To air', 'No conversion','To vaccuum']
    # radio = RadioButtons(rax_atv,clabels)
    # radio.on_clicked(M.atv)

    #The mask width:
    rax_slider = plt.axes([0.8, 0.3, 0.14, 0.02])
    rax_slider.set_title('Mask width')
    M.MW_slider = Slider(rax_slider,'', 1,200,valinit=M.MW,valstep=1)#Store the slider in the model class
    M.MW_slider.on_changed(M.slide_maskwidth)

    #The slider to cycle through orders:
    rax_slider = plt.axes([0.8, 0.2, 0.14, 0.02])
    rax_slider.set_title('Order')
    M.mask_slider = Slider(rax_slider,'', 0,M.N_orders-1,valinit=M.N,valstep=1)#Store the slider in the model class
    M.mask_slider.on_changed(M.slide_order)

    #The Previous order button:
    rax_prev = plt.axes([0.8, 0.1, 0.04, 0.05])
    bprev = Button(rax_prev, ' <<< ')
    bprev.on_clicked(M.previous)

    #The Next order button:
    rax_next = plt.axes([0.86, 0.1, 0.04, 0.05])
    bnext = Button(rax_next, ' >>> ')
    bnext.on_clicked(M.next)

    #The save button:
    rax_save = plt.axes([0.92, 0.1, 0.07, 0.05])
    bsave = Button(rax_save, 'Save')
    bsave.on_clicked(M.save)

    #The cancel button:
    rax_cancel = plt.axes([0.92, 0.025, 0.07, 0.05])
    bcancel = Button(rax_cancel, 'Cancel')
    bcancel.on_clicked(M.cancel)


    plt.show()
    #When pressing "save", the figure is closed, the suspesion caused by plt.show() is
    #lifted and we continue to exit this GUI function, as its job has been done:
    #return all the columns that were selected by the user.
    return(M.list_of_selected_columns)


#The following functions deal with saving and loading the clipped and selected pixels
#and columns to/from file.
def load_columns_from_file(dp,maskname,mode='strict'):
    """
    This loads the list of lists of columns back into memory after having been
    saved by a call of write_columns_to_file() below.
    """
    import tayph.util as ut
    from tayph.vartests import typetest
    import pickle
    import os
    from pathlib import Path
    ut.check_path(dp)
    typetest(maskname,str,'maskname in load_columns_from_file()')
    typetest(mode,str,'mode in load_columns_from_file()')
    outpath=Path(dp)/(maskname+'_columns.pkl')

    if os.path.isfile(outpath) ==  False:
         if mode == 'strict':
             raise Exception('FileNotFoundError in reading columns from file: Column file named %s do not exist at %s.' % (maskname,dp))
         else:
             print('---No previously saved manual mask exists. User will start a new mask.')
             return([])
    else:
        print(f'------Loading previously saved manual mask {str(outpath)}.')
        pickle_in = open(outpath,"rb")
        return(pickle.load(pickle_in))

def write_columns_to_file(dp,maskname,list_of_selected_columns):
    """
    This dumps the list of list of columns that are manually selected by the
    user to a pkl file for loading at a later time. This is done to allow the user
    to resume work on a saved mask.
    """
    import pickle
    from pathlib import Path
    import tayph.util as ut
    from tayph.vartests import typetest
    ut.check_path(dp)
    typetest(maskname,str,'maskname in write_columns_to_file()')
    typetest(list_of_selected_columns,list,'list_of_selected_columns in write_columns_to_file()')
    outpath=Path(dp)/(maskname+'_columns.pkl')
    print(f'---Saving list of masked columns to {str(outpath)}')
    with open(outpath, 'wb') as f: pickle.dump(list_of_selected_columns, f)

def write_mask_to_file(dp,maskname,list_of_masks_auto,list_of_masks_manual=[]):
    import sys
    from pathlib import Path
    import tayph.util as ut
    from tayph.vartests import typetest,dimtest
    import pdb
    ut.check_path(dp)
    typetest(maskname,str,'maskname in write_mask_to_file()')
    typetest(list_of_masks_auto,list,'list_of_masks_auto in write_mask_to_file()')
    typetest(list_of_masks_manual,list,'list_of_masks_manual in write_mask_to_file()')
    dimtest(list_of_masks_auto,[len(list_of_masks_manual),0,0],'list_of_masks_auto in write_mask_to_file()')
    outpath=Path(dp)/maskname
    if len(list_of_masks_auto) == 0 and len(list_of_masks_manual) == 0:
        print('RuntimeError in write_mask_to_file: Both lists of masks are emtpy!')
        sys.exit()

    print(f'---Saving lists of auto and manual masks to {str(outpath)}')
    if len(list_of_masks_auto) > 0:
        ut.save_stack(str(outpath)+'_auto.fits',list_of_masks_auto)
    if len(list_of_masks_manual) > 0:
        ut.save_stack(str(outpath)+'_manual.fits',list_of_masks_manual)

def apply_mask_from_file(dp,maskname,list_of_orders):
    import astropy.io.fits as fits
    import numpy as np
    import os.path
    import sys
    import tayph.util as ut
    from tayph.vartests import typetest,dimtest
    from pathlib import Path
    ut.check_path(dp)
    typetest(maskname,str,'maskname in write_mask_to_file()')
    typetest(list_of_orders,list,'list_of_orders in apply_mask_from_file()')

    N = len(list_of_orders)

    inpath_auto = Path(dp)/(maskname+'_auto.fits')
    inpath_man = Path(dp)/(maskname+'_manual.fits')

    if os.path.isfile(inpath_auto) ==  False and os.path.isfile(inpath_man) == False:
        raise Exception(f'FileNotFoundError in reading mask from file: Both mask files named {maskname} do not exist at {str(dp)}.')

    #At this point either of the mask files is determined to exist.
    #Apply the masks to the orders, by adding. This works because the mask is zero
    #everywhere, apart from the NaNs, and x+0=x, while x+NaN = NaN.


    if os.path.isfile(inpath_auto) ==  True:
        print('------Applying sigma_clipped mask from %s' % inpath_auto)
        cube_of_masks_auto = fits.getdata(inpath_auto)
        Nm = len(cube_of_masks_auto)
        err = f'ERROR in apply_mask_from_file: List_of_orders and list_of_masks_auto do not have the same length ({N} vs {Nm}), meaning that the number of orders provided and the number of orders onto which the masks were created are not the same. This could have happened if you copy-pased mask_auto from one dataset to another. This is not recommended anyway, as bad pixels / outliers are expected to be in different locations in different datasets.'
        if Nm != N:
            raise Exception(err)
        #Checks have passed. Add the mask to the list of orders.
        for i in range(N):
            list_of_orders[i]+=cube_of_masks_auto[i,:,:]

    if os.path.isfile(inpath_man) ==  True:
        print('------Applying manually defined mask from %s' % inpath_man)
        cube_of_masks_man = fits.getdata(inpath_man)
        Nm = len(cube_of_masks_man)
        err = f'ERROR in apply_mask_from_file: List_of_orders and list_of_masks_auto do not have the same length ({N} vs {Nm}), meaning that the number of orders provided and the number of orders onto which the masks were created are not the same. This could have happened if you copy-pased mask_auto from one dataset to another. This is not recommended anyway, as bad pixels / outliers are expected to be in different locations in different datasets.'
        if Nm != N:
            raise Exception(err)
        for i in range(N):
            list_of_orders[i]+=cube_of_masks_man[i,:,:]
    return(list_of_orders)



def mask_orders(list_of_wls,list_of_orders,dp,maskname,w,c_thresh,manual=False):
    """
    This code takes the list of orders and masks out bad pixels.
    It combines two steps, a simple sigma clipping step and a manual step, where
    the user can interactively identify bad pixels in each order. The sigma
    clipping is done on a threshold of c_thresh, using a rolling standard dev.
    with a width of w pixels. Manual masking is a big routine needed to support
    a nice GUI to do that.

    If c_thresh is set to zero, sigma clipping is skipped. If manual=False, the
    manual selection of masking regions (which is manual labour) is turned off.
    If both are turned off, the list_of_orders is returned unchanged.

    If either or both are active, the routine will output 1 or 2 FITS files that
    contain a stack (cube) of the masks for each order. The first file is the mask
    that was computed automatically, the second is the mask that was constructed
    manually. This is done so that the manual mask can be transplanted onto another
    dataset, or saved under a different file-name, to limit repetition of work.

    At the end of the routine, the two masks are merged into a single list, and
    applied to the list of orders.
    """
    import tayph.operations as ops
    import numpy as np
    import tayph.functions as fun
    import tayph.plotting as plotting
    import sys
    import matplotlib.pyplot as plt
    import tayph.util as ut
    import warnings
    from tayph.vartests import typetest,dimtest,postest
    ut.check_path(dp)
    typetest(maskname,str,'maskname in mask_orders()')
    typetest(w,[int,float],'w in mask_orders()')
    typetest(c_thresh,[int,float],'c_thresh in mask_orders()')
    postest(w,'w in mask_orders()')
    postest(c_thresh,'c_thresh in mask_orders()')
    typetest(list_of_wls,list,'list_of_wls in mask_orders()')
    typetest(list_of_orders,list,'list_of_orders in mask_orders()')
    typetest(manual,bool,'manual keyword in mask_orders()')
    dimtest(list_of_wls,[0,0],'list_of_wls in mask_orders()')
    dimtest(list_of_orders,[len(list_of_wls),0,0],'list_of_orders in mask_orders()')

    if c_thresh <= 0 and manual == False:
        print('---WARNING in mask_orders: c_thresh is set to zero and manual masking is turned off.')
        print('---Returning orders unmasked.')
        return(list_of_orders)

    N = len(list_of_orders)
    void = fun.findgen(N)

    list_of_orders = ops.normalize_orders(list_of_orders,list_of_orders)[0]#first normalize. Dont want outliers to
    #affect the colour correction later on, so colour correction cant be done before masking, meaning
    #that this needs to be done twice; as colour correction is also needed for proper maskng. The second variable is
    #a dummy to replace the expected list_of_sigmas input.
    N_NaN = 0
    list_of_masked_orders = []

    for i in range(N):
        list_of_masked_orders.append(list_of_orders[i])

    list_of_masks = []

    if c_thresh > 0:#Check that c_thresh is positive. If not, skip sigma clipping.
        print('------Sigma-clipping mask')
        for i in range(N):
            order = list_of_orders[i]
            N_exp = np.shape(order)[0]
            N_px = np.shape(order)[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                meanspec = np.nanmean(order,axis = 0)
            meanblock = fun.rebinreform(meanspec,N_exp)
            res = order / meanblock - 1.0
            sigma = fun.running_MAD_2D(res,w)
            with np.errstate(invalid='ignore'):#https://stackoverflow.com/questions/25345843/inequality-comparison-of-numpy-array-with-nan-to-a-scalar
                sel = np.abs(res) >= c_thresh*sigma
                N_NaN += np.sum(sel)#This is interesting because True values count as 1, and False as zero.
                order[sel] = np.nan
            list_of_masks.append(order*0.0)
            ut.statusbar(i,void)

        print(f'%s outliers identified and set to NaN ({N_NaN}/{round(N_NaN/np.size(list_of_masks)*100.0,3)}).')
    else:
        print('------Skipping sigma-clipping (c_thres <= 0)')
        #Do nothing to list_of_masks. It is now an empty list.
        #We now automatically proceed to manual masking, because at this point
        #it has already been established that it must have been turned on.


    list_of_masks_manual = []
    if manual == True:


        previous_list_of_masked_columns = load_columns_from_file(dp,maskname,mode='relaxed')
        list_of_masked_columns = manual_masking(list_of_wls,list_of_orders,list_of_masks,saved = previous_list_of_masked_columns)
        print('------Successfully concluded manual mask.')
        write_columns_to_file(dp,maskname,list_of_masked_columns)

        print('------Building manual mask from selected columns')
        for i in range(N):
            order = list_of_orders[i]
            N_exp = np.shape(order)[0]
            N_px = np.shape(order)[1]
            list_of_masks_manual.append(np.zeros((N_exp,N_px)))
            for j in list_of_masked_columns[i]:
                list_of_masks_manual[i][:,int(j)] = np.nan

    #We write 1 or 2 mask files here. The list of manual masks
    #and list_of_masks (auto) are either filled, or either is an emtpy list if
    #c_thresh was set to zero or manual was set to False (because they were defined
    #as empty lists initially, and then not filled with anything).
    write_mask_to_file(dp,maskname,list_of_masks,list_of_masks_manual)
    return(0)
