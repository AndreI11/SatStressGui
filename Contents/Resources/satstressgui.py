#!/usr/bin/python
# -*- coding: utf-8 -*-

# for GUI
#    wx = python extention module that acts as python lang wrapper for wxWidgets
#         (cross platform GUI API written in C++)t
import wx
import wx.scrolledpanel as scrolled
# for manipulating tabluar data
#     wx.grid = allows displaying, editing, customizing tabular data'''
import wx.grid, csv

# for command line parsing?
#     copy/Users/andreismailyan = allows shallow & deep copying operations
#     sys = access syst specific parameters & fcts, variables held by interpreter
#     os = allows more direct interaction with OS
import copy, sys, os

# for trying to print stack traces under program control (i.e. wrapper), probs used for wx?
#     traceback = allows printing/extracting/formatting Python program stack traces
import traceback

# for plotting capabilities
#    matplotlib = 2D
#    scipy.ndimage = multi-D
import matplotlib, scipy.ndimage

import time
#For saving images in unique folders.

# fig state updated every plot command, but only redraw on explicit calls to draw()
matplotlib.interactive(False)
# sets matplotlib backend to 'WXAgg'
matplotlib.use('WXAgg')

# for plotting
#    basemap allows cartographic
from mpl_toolkits import basemap
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas, \
                                              NavigationToolbar2Wx as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# for manipulating netCDF files
import netCDF3

# for math functions
import numpy

# satstress library
from satstress.satstress import *
from satstress.gridcalc import *
from satstress.lineament import plotlinmap, Lineament, lingen_nsr, shp2lins, lins2shp  
from satstress.cycloid import Cycloid, plotcoordsonbasemap
from satstress.stressplot import scalar_grid, vector_points
import satstress.physcon

import re

from Model import DataModel


# constants set as global variables
seconds_in_year = 31556926.0  ## 365.24 days
vector_mult = 4000
display_dpi = 72
scale_left = 0.25
scale_bar_length = 0.38


def vector_points1(stresscalc=None, lons=None, lats=None, time_t=0.0,\
                   plot_tens=True, plot_comp=True, plot_greater=True, plot_lesser=True,\
                   basemap_ax=None, lonshift=0, w_stress=False,\
                   scale=1e8, scale_arr=None, arrow_width=0.008):
    """
        Display the principal components of the tidal stresses defined by the input
        stresscalc object at the points defined by lons and lats, which are one
        dimensional arrays of equal length, and a time defined by time_t, in
        seconds after periapse.
        
        The stress vectors are plotted on the map axes defined by basemap_ax.
        
        By default all the principal components are plotted, but if you wish to see
        only the more or less tensile (less or more compressive) or only those
        principal components which are absolutely compressive or tensile, you may
        exclude some subset of the vectors using the following flags:
        
        plot_tens  --  if True, plot all tensile stresses.
        plot_comp  --  if True, plot all compressive stresses.
        plot_greater  --  if True, plot the greater (more tensile) principal component
        plot_lesser  --  if True, plot the lesser (more compressive) principal component
        
        lonshift is a longitudinal displacement added to lons when the stresses are
        calculated, useful in creating plots of lineaments at their current
        location, compared to stresses that they would have experienced at their
        apparent location of formation (i.e. those stresses which best match the
        feature).  For instance, if you wished to show only those stresses which are
        the more tensile, and which are actually tensile, you would need to set
        the flags: plot_comp=False, plot_lesser=False.
        
        If w_stress is true, the lengths of the arrows which are used to represent
        the stresses are scaled according to how significant their location within
        the stress field is, i.e. large stresses and anisotropic stresses will be
        more prominent than small stresses and isotropic stresses.
        
        scale determines the overall size of the arrows representing the stresses.
        A smaller scale means bigger arrows.
        
        scale_arr is an array of the same length as lons and lats, which is used to
        scale the lengths of the vectors.  Useful in showing the relative
        importance of different segments of a feature having non-uniform lengths.
        
        arrow_width is passed in to numpy.quiver(), and is the width of the arrow
        shaft, as a proportion of the width of the overall plot.
        """
    
    calc_phis   = lons
    calc_thetas = (numpy.pi / 2.0) - lats
    
    Ttt, Tpt, Tpp = stresscalc.tensor(calc_thetas, calc_phis + lonshift, time_t)
    
    Tau = numpy.array([[Ttt, Tpt], [Tpt, Tpp]])
    eigensystems = [ numpy.linalg.eig(Tau[:,:,N]) for N in range(len(Tau[0,0,:])) ]
    evals = numpy.array([ e[0] for e in eigensystems ])
    evecs = numpy.array([ e[1] for e in eigensystems ])
    
    eigval_A = evals[:,0]
    ex_A     = evecs[:,0,1]
    ey_A     = evecs[:,0,0]
    
    eigval_B = evals[:,1]
    ex_B     = evecs[:,1,1]
    ey_B     = evecs[:,1,0]
    
    mag1 = numpy.where(eigval_A >  eigval_B, eigval_A, eigval_B)
    ex1  = numpy.where(eigval_A >  eigval_B, ex_A, ex_B)
    ey1  = numpy.where(eigval_A >  eigval_B, ey_A, ey_B)
    
    mag2 = numpy.where(eigval_A <= eigval_B, eigval_A, eigval_B)
    ex2  = numpy.where(eigval_A <= eigval_B, ex_A, ex_B)
    ey2  = numpy.where(eigval_A <= eigval_B, ey_A, ey_B)
    
    if numpy.shape(scale_arr) != numpy.shape(mag1):
        scale_arr = numpy.ones(numpy.shape(mag1))
    if numpy.shape(w_stress) == numpy.shape(mag1):
        scale_arr = scale_arr*(mag1 - mag2)/stresscalc.mean_global_stressdiff()
    
    mag1_comp = numpy.ma.masked_where(mag1 > 0, mag1)
    mag1_tens = numpy.ma.masked_where(mag1 < 0, mag1)
    
    mag2_comp = numpy.ma.masked_where(mag2 > 0, mag2)
    mag2_tens = numpy.ma.masked_where(mag2 < 0, mag2)
    
    scaled = {}
    scaled[1, 'comp'] = mag1_comp*scale_arr
    scaled[2, 'comp'] = mag2_comp*scale_arr
    scaled[1, 'tens'] = mag1_tens*scale_arr
    scaled[2, 'tens'] = mag2_tens*scale_arr
    
    ex = { 1: ex1, 2: ex2 }
    ey = { 1: ey1, 2: ey2 }
    # map coordinates
    dlons, dlats = numpy.degrees(lons), numpy.degrees(lats)
    x,y = basemap_ax(dlons, dlats)
    # new basis
    exx,exy = basemap_ax.rotate_vector(numpy.ones(numpy.shape(lons)), numpy.zeros(numpy.shape(lons)), dlons, dlats)
    eyx,eyy = basemap_ax.rotate_vector(numpy.zeros(numpy.shape(lats)), numpy.ones(numpy.shape(lats)), dlons, dlats)
    
    rotated = {}
    for i in range(1,3):
        for s in ['comp', 'tens']:
            x1 = scaled[i,s] * ex[i]
            y1 = scaled[i,s] * ey[i]
            rotated[i,s,'x'], rotated[i,s,'y'] = x1*exx + y1*eyx, x1*exy + y1*eyy
    
    # where is the exclusion done?
    for i in range(1,3):
        for k in range(2):
            basemap_ax.quiver(x, y, (-1)**k*rotated[i,'comp','x'], (-1)**k*rotated[i,'comp','y'],
            lw=0., width=arrow_width, scale=scale, color='blue', pivot='tip', units='inches')
            basemap_ax.quiver(x, y, (-1)**k*rotated[i,'tens','x'], (-1)**k*rotated[i,'tens','y'],
            lw=0., width=arrow_width, scale=scale, color='red', pivot='tail', units='inches')

# ===============================================================================
# Global Function for shear and normal components
# ===============================================================================
def vector_points2(stresscalc=None, lons=None, lats=None, time_t=0.0,\
                   plot_norm_lon=True, plot_norm_lat=True, plot_shear=True, \
                   basemap_ax=None, lonshift=0, \
                   scale=1e8, arrow_width=0.008):
    """
        Display the normal and shear components of the tidal stresses defined by the input
        stresscalc object at the points defined by lons and lats, which are one
        dimensional arrays of equal length, and a time defined by time_t, in
        seconds after periapse.
        
        The stress vectors are plotted on the map axes defined by basemap_ax.
        
        lonshift is a longitudinal displacement added to lons when the stresses are
        calculated, useful in creating plots of lineaments at their current
        location, compared to stresses that they would have experienced at their
        apparent location of formation (i.e. those stresses which best match the
        feature).  For instance, if you wished to show only those stresses which are
        the more tensile, and which are actually tensile, you would need to set
        the flags: plot_comp=False, plot_lesser=False.
        
        scale determines the overall size of the arrows representing the stresses.
        A smaller scale means bigger arrows.
        
        arrow_width is passed in to numpy.quiver(), and is the width of the arrow
        shaft, as a proportion of the width of the overall plot.
        
        """
    
    calc_phis   = lons
    # because should be co-latitudal (south-positive, [0, pi])
    calc_thetas = (numpy.pi/2.0)-lats
    # plot coordinates
    dlons, dlats = numpy.degrees(lons), numpy.degrees(lats)
    px,py = basemap_ax(dlons, dlats)
    # new basis
    exx,exy = basemap_ax.rotate_vector(numpy.ones(numpy.shape(lons)), numpy.zeros(numpy.shape(lons)), dlons, dlats)
    eyx,eyy = basemap_ax.rotate_vector(numpy.zeros(numpy.shape(lats)), numpy.ones(numpy.shape(lats)), dlons, dlats)
    
    Ttt, Tpt, Tpp = stresscalc.tensor(calc_thetas, calc_phis+lonshift, time_t)
    
    def plot_vector(v, x, y, scale, color, pivot):
        vx, vy = v*x*exx + v*y*eyx, v*x*exy + v*y*eyy
        basemap_ax.quiver(px, py,  vx,  vy,\
        lw=0., width=arrow_width, scale=scale, color=color, pivot=pivot, units='inches')
        basemap_ax.quiver(px, py, -vx, -vy,\
        lw=0., width=arrow_width, scale=scale, color=color, pivot=pivot, units='inches')
    
    def plot_vectors(vs, x, y):
        plot_vector(numpy.ma.masked_where(vs > 0, vs), x, y, scale, 'blue', 'tip')
        plot_vector(numpy.ma.masked_where(vs < 0, vs), x, y, scale, 'red', 'tail')
    
    if plot_norm_lat:
        plot_vectors(Ttt, 0, 1)
    if plot_norm_lon:
        plot_vectors(Tpp, 1, 0)
    if plot_shear:
        for diag_angle in [numpy.pi/4, 3*numpy.pi/4]:
            plot_vectors(-Tpt, numpy.cos(diag_angle), 1 - numpy.sin(diag_angle))


# ===============================================================================
# Global Functions showing dialog boxes given event
# ===============================================================================
def file_dir_dialog(parent, dialog_class, message="", style=wx.OPEN, action=None, **kw):
    fd = dialog_class(parent, message=message, style=style, **kw)
    wx.Yield()
    if (fd.ShowModal() == wx.ID_OK):
        action(fd.GetPath())
    fd.Destroy()

def file_dialog(parent, message="", style=wx.OPEN, action=None, **kw):
    file_dir_dialog(parent, wx.FileDialog, message, style, action, **kw)

def dir_dialog(parent, message="", style=wx.OPEN, action=None, **kw):
    file_dir_dialog(parent, wx.DirDialog, message, style, action, **kw)

def error_dialog(parent, e, title=u'Error'):
    d = wx.MessageDialog(parent, e, title, style=wx.ICON_ERROR|wx.OK)
    d.ShowModal()

class LocalError(Exception):
    def __init__(self, e, title):
        self.msg = str(e)
        self.title = title
    
    def __str__(self):
        return self.msg


# ===============================================================================
# Exception class, simple error handling
# ===============================================================================

class ComboBox2(wx.ComboBox):
    """
    Custom implementation of wx.ComboBox
    """
    def __init__(self, parent, id=-1, value='',
        pos=wx.DefaultPosition, size=wx.DefaultSize,
        choices=[], style=0, validator=wx.DefaultValidator,
        name=wx.ChoiceNameStr):
        self.__choice_map = dict([ (c,d) for c,d in choices ])
        self.__reverse_map = dict([ (d,c) for c,d in choices ])
        super(ComboBox2, self).__init__(parent, id=id, value=self.__choice_map[value],
            pos=pos, size=size, choices=[ d for c,d in choices ],
            style=style, validator=validator, name=name)
    
    def GetValue(self):
        return self.__reverse_map[super(ComboBox2, self).GetValue()]
    
    def SetValue(self, val):
        super(ComboBox2, self).SetValue(self.__choice_map[val])



class Config:
    """
    Class the holds application settings --> specifically?
    """
    #default step for plots <-- what units?
    default_step = 30

    def __init__(self, configfile='config'):
        self.configfile = configfile
        self.conf = {}

    # a is optional arg
    def load(self, *a):
        try:
            c = open(self.configfile)
            self.conf = nvf2dict(c)
            c.close()
            ret = filter(lambda x: x, map(self.conf.get, a))
            if len(a) == 1 and len(ret) == 1:
                return ret[0]
            else:
                return ret
        except:
            self.conf = {}
            return []

    # **kw unpacks the extra dictionary args
    def save(self, **kw):
        for k, v in kw.items():
            self.conf[k] = v   # conf is dictionary
        try:
            c = open(self.configfile, 'w')
            c.writelines([ "%s = %s\n" % (k,v) for k,v in self.conf.items() ])
            c.close()
        except:
            pass

    def load_step(self, step_field='STEP'):
        self.load()
        try:
            return float(self.conf.get(step_field, self.default_step))
        except:
            return self.default_step

    def save_step(self, step, step_field='STEP'):
        self.conf[step_field] = step
        self.save()

#creates a global instance of config
config = Config()


class WrapStaticText(wx.StaticText):
    """
    Constrains number of characters in a line in the GUI to less than 58
    """
    def __init__(self, *args, **kw):
        super(WrapStaticText, self).__init__(*args, **kw)
        self._label = self.GetLabel()
        self._rewrap()
        wx.EVT_SIZE(self, self.change_size)
    def _rewrap(self):
        w = self.GetSize().width
        self.SetLabel(self._label)
        if w > 50:
            self.Wrap(w)
    def change_size(self, evt):
        self.Unbind(wx.EVT_SIZE)
        self._rewrap()
        self.Bind(wx.EVT_SIZE, self.change_size)

class SatStressPanel(wx.Panel):
    """
        Defines the panel that contains all GUI pages
        """
    def __init__(self, *args, **kw):
        wx.Panel.__init__(self,*args)

        self.SetMinSize((1024, 640))
        sz = wx.BoxSizer(orient=wx.VERTICAL)
        
        self.nb = wx.Notebook(self)
        self.slp = SatelliteLayersPanel(self.nb, model=kw['model'],controller = kw['controller'], view_parameters=kw['view_parameters'])
        self.stp = StressListPanel(self.nb, model=kw['model'], controller = kw['controller'], view_parameters=kw['view_parameters'])
        self.gp = GridCalcPanel(self.nb, model=kw['model'],controller = kw['controller'], view_parameters=kw['view_parameters'])
        self.pp=  PointPanel(self.nb, model=kw['model'],controller = kw['controller'], view_parameters=kw['view_parameters'])
        self.cp=  CycloidsPanel(self.nb, model=kw['model'],controller = kw['controller'], view_parameters=kw['view_parameters'])

        self.spp = ScalarPlotPanel(self.nb, model=kw['model'],controller = kw['controller'], view_parameters=kw['view_parameters'])

        self.nb.AddPage(self.slp, u"Satellite")
        self.nb.AddPage(self.stp, u"Stresses")
        self.nb.AddPage(self.pp,  u"Points")
        self.nb.AddPage(self.gp,  u"Grid")
        self.nb.AddPage(self.cp,  u"Cycloids")
        self.nb.AddPage(self.spp,  u"Plot")


        # Assign each panel to a page and give it a name
        
        # self.nb.AddPage(dummy, u'Test')
        
        sz.Add(self.nb, 1, wx.ALL|wx.EXPAND)
        #sz.Add(bp, 0, wx.ALIGN_BOTTOM | wx.EXPAND)
        
        self.SetSizer(sz)
        self.Fit()
            
   
        
        #if isinstance(p, PlotPanel):
         #   p.plot()

class SatPanel(wx.Panel):
    """
    Class that serves as the superclass of all panels of GUI.
    Having all tabs share a SatelliteCalculation object under superclass SatPanel
    allows information to be passed from one panel to another. 
    """
    def __init__(self, *args, **kw):
        super(SatPanel, self).__init__(*args)
        self.model = kw['model']
        self.controller = kw['controller']
        self.view_parameters = kw['view_parameters']

    def make_text_controls_with_labels(self, parent, sz, parameters_d):
        for param_name, labl in parameters_d:
            sz.Add(wx.StaticText(parent, label=labl), flag=wx.ALIGN_CENTER_VERTICAL)
            self.view_parameters[param_name] = wx.TextCtrl(parent, style=wx.TE_PROCESS_ENTER, name=param_name)
            sz.Add(self.view_parameters[param_name], flag=wx.EXPAND|wx.ALL)
    
    def add_static_texts(self,parent,sz, parameters_d):
        sts = [ wx.StaticText(parent, label=d, style=wx.TE_PROCESS_ENTER) for p, d in parameters_d ]
        for st in sts:
            sz.Add(st, flag=wx.ALIGN_CENTER)
        return sts

    def make_text_controls(self,parent,sz,parameters_d):
        params = {}
        for param_name, labl in parameters_d:
            self.view_parameters[param_name] = wx.TextCtrl(parent, style = wx.TE_PROCESS_ENTER, name = param_name)
            sz.Add(self.view_parameters[param_name], flag=wx.ALL|wx.EXPAND)
            params.update({param_name:self.view_parameters[param_name]})
        return params

    def make_checkbox_controls(self,parent,sz,parameters_d):
        for param_name, d in parameters_d:
            self.view_parameters[param_name] = wx.CheckBox(parent, label=d, name=param_name)
            sz.Add(self.view_parameters[param_name], flag=wx.ALIGN_CENTER_VERTICAL)
        
    def make_text_control(self, parent, sz, param_name):
        text = wx.TextCtrl(parent, style = wx.TE_PROCESS_ENTER, name = param_name)
        self.view_parameters[param_name] = text
        return text

    def make_combobox2_controls(self,parent, sz, parameter, description, choices):
        sz.Add(wx.StaticText(parent, label=description), flag=wx.ALIGN_CENTER_VERTICAL)
        self.view_parameters[parameter] = ComboBox2(parent, value=self.model.get_parameter(parameter).get_value(), choices=choices, style=wx.CB_DROPDOWN | wx.CB_READONLY, name= parameter)
        sz.Add(self.view_parameters[parameter])
        


class SatelliteLayersPanel(SatPanel):
    """
    Defines the satellite layers panel of the GUI
    """
    def __init__(self, *args, **kw):
        super(SatelliteLayersPanel, self).__init__(*args, **kw)

        satellite_vars = [
        ("SYSTEM_ID", u"System ID"),
        ("PLANET_MASS", u"Planet Mass [kg]"),
        ("ORBIT_ECCENTRICITY", u"Orbit Eccentricity"),
        ("ORBIT_SEMIMAJOR_AXIS", u"Orbit Semimajor Axis [m]"),
        ("NSR_PERIOD", u"NSR Period [yrs]")]


        layer_vars_d = [
        ("LAYER_ID", u"Layer ID"),
        ("DENSITY", u"Density [kg/m3]"),
        ("YOUNGS_MODULUS", u"Young's Modulus [Pa]"),
        ("POISSONS_RATIO", u"Poisson's Ratio"),
        ("THICKNESS", u"Thickness [m]"),
        ("VISCOSITY", u"Viscosity [Pa s]")]


        satlayers_d = [
        (3, "ICE_UPPER"),
        (2, "ICE_LOWER"),
        (1, "OCEAN"),
        (0, "CORE")]

        sz = wx.BoxSizer(orient=wx.VERTICAL)
        filler = wx.BoxSizer(wx.HORIZONTAL)
        filler.AddSpacer(15)
        top = wx.BoxSizer(orient=wx.VERTICAL)

        bp = wx.BoxSizer(orient=wx.HORIZONTAL)
        load_b = wx.Button(self, label=u'Load from file')
        save_b = wx.Button(self, label=u'Save to file')
        bp.Add(load_b, 1, wx.ALL|wx.EXPAND, 3)
        bp.Add(save_b, 1, wx.ALL|wx.EXPAND, 3)
        # satellite parameters
        # FlexGridSizer organizes visual elements into grid layout
        sp = wx.FlexGridSizer(1,2)
        self.make_text_controls_with_labels(self, sp, satellite_vars)        

        lp = wx.FlexGridSizer(1, len(layer_vars_d))
        self.add_static_texts(self, lp, layer_vars_d)
        lv = []

        for l, v in satlayers_d:
            for p, d in layer_vars_d:
                #if p == 'YOUNG':
                #    lv.append(('LAME_MU_%d' % l, ''))
                #if p == 'POISSON':
                #    lv.append(('LAME_LAMBDA_%d' % l, ''))
                lv.append(("%s_%d" % (p, l), ''))

        self.make_text_controls(self, lp, lv)



        top.Add(bp, 0, wx.ALL|wx.EXPAND)
        top.Add(filler)
        top.Add(sp)


        sz.Add(top)
        sz.Add(filler)
        sz.Add(lp)


        sz.Add(wx.StaticText(self, label=u'ASSUMPTIONS: '))
        sz.Add(wx.StaticText(self, label=u'This model makes several assumptions when calculating stresses.'))
        sz.Add(wx.StaticText(self, label=u'The body is assumed to be composed of four layers, with the third layer being a liquid ocean.'))
        sz.Add(wx.StaticText(self, label=u'It is assumed to behave in a viscoelastic manner.'))
        sz.Add(wx.StaticText(self, label=u'Each layer is considered to be homogenous throughout, with no differences in density or thickness based on location, but decreasing in mass out from the core.'))
        sz.Add(wx.StaticText(self, label=u'The Polar Wander stress assumes that the body is in a circular, zero-inclination, synchronous orbit.'))
        sz.Add(wx.StaticText(self, label=u'Polar Wander stress is calculated using an elastic model.'))
        sz.Add(wx.StaticText(self, label=u'The orbit is assumed to have an eccentricity of <0.25, and the primary\'s mass be at least 10 times the satellite\'s mass.'))
        
        self.SetSizer(sz)
        wx.EVT_BUTTON(self, load_b.GetId(), self.load)
        wx.EVT_BUTTON(self, save_b.GetId(), self.save)
    def load(self, evt):
        
        file_dialog(self,
            message=u"Load from satellite file",
            style=wx.OPEN,
            wildcard='Satellite files (*.satellite;*.sat)|*.satellite;*.sat',
            action=self.controller.load_file)
        
    
    def save(self, evt):
        file_dialog(self,
            message=u"Save to satellite file",
            style=wx.SAVE | wx.OVERWRITE_PROMPT,
            wildcard='Satellite files (*.satellite;*.sat)|*.satellite;*.sat',
            defaultFile='satellite.satellite',
            action=self.controller.save_file)


class StressListPanel(SatPanel):
    """
    Defines the stresses panel of the GUI. Contains stress type selections and allows
    user to input their own love numbers for further calculations
    """
    def __init__(self, *args, **kw):
        super(StressListPanel, self).__init__(*args, **kw)

        filler = wx.BoxSizer(wx.HORIZONTAL)
        filler.AddSpacer(15)

        topsizer = wx.BoxSizer(wx.VERTICAL)
        othersz = wx.BoxSizer(wx.HORIZONTAL) #Does this have a purpose?
        
        sz = wx.BoxSizer(orient=wx.VERTICAL)

        topsizer.Add(WrapStaticText(self, label=
            u'Select the stress types to use in further computation, such as Love numbers, stress tensors, plotting of stress trajectories.'),
            0, wx.ALL|wx.EXPAND)
        topsizer.AddSpacer(10)

        sz.AddSpacer(23)
        
        # for Diurnal
        self.make_checkbox_controls(self, sz, [ ('Diurnal', 'Diurnal') ])
        sz.AddSpacer(8)
        
        # for NSR
        self.make_checkbox_controls(self, sz, 
            [ ('Nonsynchronous Rotation', 'Nonsynchronous Rotation') ])

        sz.AddSpacer(8)

        sz.Add(wx.StaticText(self, label=u'To input custom Love numbers, use the format <Re> +/- <Im>j.'))
        sz.Add(wx.StaticText(self, label=u'Do not use scientific notation when inputting custom Love numbers.'))
        sz.Add(wx.StaticText(self, label=u'"3.0-1.0e-03j" should be written as "3.0-0.001j".'))

        sz.AddSpacer(8)
        
        # for Diurnal w/ Obliquity
        self.make_checkbox_controls(self, sz,
            [ ('Obliquity', 'Obliquity') ])
        DiObliq_sz = wx.BoxSizer(wx.VERTICAL)
        # include arg of periapsis parameter for Diurnal w/ Obliquity
        peri_sz = wx.BoxSizer(orient=wx.HORIZONTAL)
        peri_sz.AddSpacer(28) 
        self.periapsis_label = wx.StaticText(self,
           label=u'Argument of Periapsis [°]  ')
        peri_sz.Add(self.periapsis_label, flag=wx.ALIGN_CENTER_VERTICAL)
        self.make_text_controls(self, peri_sz,
           [ ('periapsis_arg', 'periapsis_arg') ])
        DiObliq_sz.Add(peri_sz)
        DiObliq_sz.AddSpacer(5)
        # inclue degree of obliquity for Diurnal w/ Obliquity
        obliq_sz = wx.BoxSizer(orient=wx.HORIZONTAL)
        obliq_sz.AddSpacer(30)
        self.obliq_label = wx.StaticText(self, label=u'Degree of Obliquity [°]  ')
        obliq_sz.Add(self.obliq_label, flag=wx.ALIGN_CENTER_HORIZONTAL)
        obliq_sz.Add(filler)
        self.make_text_controls(self, obliq_sz,
            [ ('obliquity', 'obliquity') ])
        obliq_sz.AddSpacer(5)
        DiObliq_sz.Add(obliq_sz)

        sz.Add(DiObliq_sz)

        self.make_checkbox_controls(self, sz,
            [ ('Ice Shell Thickening', 'Ice Shell Thickening') ])
        ISTParams_sz = wx.BoxSizer(wx.VERTICAL)
        # include ice thickness parameter for IST aka Ice Shell Thickening
        delta_tc_sz = wx.BoxSizer(orient=wx.HORIZONTAL)
        delta_tc_sz.AddSpacer(28)
        self.delta_label = wx.StaticText(self, label=u'Change in Thickness [km] ')
        delta_tc_sz.Add(self.delta_label, flag=wx.ALIGN_CENTER_VERTICAL)
        self.make_text_controls(self, delta_tc_sz, [ ('delta_tc', 'delta_tc') ])
        ISTParams_sz.Add(delta_tc_sz)
        ISTParams_sz.AddSpacer(5)
        # include thermal diffusivity parameter for IST
        diff_sz = wx.BoxSizer(orient=wx.HORIZONTAL)
        diff_sz.AddSpacer(28)
        self.diffusivity_label = wx.StaticText(self, label=u'Thermal Diffusivity [m\u00b2/s]  ')
        diff_sz.Add(self.diffusivity_label, flag=wx.ALIGN_CENTER_VERTICAL)
        self.make_text_controls(self, diff_sz, [ ('diffusivity', 'diffusivity') ])
        ISTParams_sz.Add(diff_sz)
        sz.Add(ISTParams_sz)
        
        
        
           
        self.make_checkbox_controls(self, sz, [ ('Polar Wander', 'Polar Wander') ])
        

        Polargrid = wx.FlexGridSizer(rows=5, cols=3, hgap=3, vgap=5)
        self.Latitude_label = wx.StaticText(self, label=u'Latitude [°]')
        self.Longitude_label = wx.StaticText(self, label=u'Longitude [°]')
        self.Blank_label = wx.StaticText(self, label=u' ')
        self.PoleInitial = wx.StaticText(self, label=u'Initial Pole Location')
        self.PoleFinal = wx.StaticText(self, label=u'Final Pole Location')
        self.TidalInitial = wx.StaticText(self, label=u'Initial Tidal Bulge Location')
        self.TidalFinal = wx.StaticText(self, label=u'Final Tidal Bulge Location')


        def set_thetaRf(self, evt):
            print 'balloony'
            #self.sc.stresses_changed = True
            self.model.stress_d['Polar Wander'].UserCoordinates.update_thetaRf(float(evt.GetString()))

       

        self.PWthetaRi = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.PWphiRi = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_phiRi, self.PWphiRi)
        self.PWthetaRf = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_thetaRf, self.PWthetaRf)
        self.PWphiRf = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_phiRf, self.PWphiRf)
        self.PWthetaTi = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_thetaTi, self.PWthetaTi)
        self.PWphiTi = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_phiTi, self.PWphiTi)
        self.PWthetaTf = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_thetaTf, self.PWthetaTf)
        self.PWphiTf = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        #self.Bind(wx.EVT_TEXT, self.set_phiTf, self.PWphiTf)


        Polargrid.AddMany([
            (self.Blank_label, 0, wx.ALL|wx.EXPAND), (self.Latitude_label, 0, wx.ALL|wx.EXPAND), (self.Longitude_label, 0, wx.ALL|wx.EXPAND),
            (self.PoleInitial, 0, wx.ALL|wx.EXPAND), (self.PWthetaRi, 0, wx.ALL|wx.EXPAND), (self.PWphiRi, 0, wx.ALL|wx.EXPAND),
            (self.PoleFinal, 0, wx.ALL|wx.EXPAND), (self.PWthetaRf, 0, wx.ALL|wx.EXPAND), (self.PWphiRf, 0, wx.ALL|wx.EXPAND),
            (self.TidalInitial, 0, wx.ALL|wx.EXPAND), (self.PWthetaTi, 0, wx.ALL|wx.EXPAND), (self.PWphiTi, 0, wx.ALL|wx.EXPAND),
            (self.TidalFinal, 0, wx.ALL|wx.EXPAND), (self.PWthetaTf, 0, wx.ALL|wx.EXPAND), (self.PWphiTf, 0, wx.ALL|wx.EXPAND)
            ])


        sz.Add(Polargrid)

        sz.AddSpacer(15)
        save_love_bt = wx.Button(self, label='Save Love numbers')
        wx.EVT_BUTTON(self, save_love_bt.GetId(), self.on_save_love)
        sz.Add(save_love_bt)

        ##### Create boxes for inputting love number #####
        ## todo: display calcuated love numbers in these boxes also ##
        grid = wx.FlexGridSizer(rows=4, cols=3, hgap=0, vgap=5)
        
        self.h2 = wx.StaticText(self, label=u'h\u2082')
        self.k2 = wx.StaticText(self, label=u'k\u2082')
        self.l2 = wx.StaticText(self, label=u'l\u2082')

        self.h2Diurn = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.k2Diurn = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.l2Diurn = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.userDiurn = wx.CheckBox(self, wx.ID_ANY, label='Input Love Numbers')
        self.h2NSR = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.k2NSR = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.l2NSR = wx.TextCtrl(self, wx.ID_ANY, '', style=wx.TE_PROCESS_ENTER)
        self.userNSR = wx.CheckBox(self, wx.ID_ANY, label='Input Love Numbers')
        
        grid.AddMany([
            (self.h2, 0, wx.ALL|wx.EXPAND), (self.k2, 0, wx.ALL|wx.EXPAND), (self.l2, 0, wx.ALL|wx.EXPAND), 
            (self.h2Diurn, 0, wx.ALL|wx.EXPAND), (self.k2Diurn, 0, wx.ALL|wx.EXPAND), (self.l2Diurn, 0, wx.ALL|wx.EXPAND),
            (self.h2NSR, 0, wx.ALL|wx.EXPAND), (self.k2NSR, 0, wx.ALL|wx.EXPAND), (self.l2NSR, 0, wx.ALL|wx.EXPAND)
            ])


        ck = wx.BoxSizer(wx.VERTICAL)
        ck.AddSpacer(23)
        
        ck.Add(self.userDiurn)
        ck.AddSpacer(8)
        ck.Add(self.userNSR)
        
        othersz.Add(sz, 5, wx.ALL|wx.EXPAND)
        othersz.Add(grid, 5, wx.ALL|wx.EXPAND)
        othersz.Add(ck, 10, wx.ALL|wx.EXPAND)

        topsizer.Add(othersz, wx.ALL|wx.EXPAND)
        self.SetSizer(topsizer)


    def on_save_love(self, evt):
        try:
            file_dialog(self,
                message=u"Save Love Numbers",
                style=wx.SAVE | wx.OVERWRITE_PROMPT,
                wildcard='Text files (*.txt)|*.txt',
                defaultFile='love.txt',
                action=self.sc.save_love)
        except LocalError, e:
            error_dialog(self, str(e), e.title)

def add_table_header(parent, sz, parameters_d):
        sts = [ wx.StaticText(parent, label=d[0], style=wx.TE_PROCESS_ENTER) for p, d in parameters_d ]
        for st in sts:
            sz.Add(st, flag=wx.ALIGN_CENTER)
        return sts 
class PointPanel(SatPanel):
  
    
    def params_grid(self, panel, params_d, defval, width=3, row = 1):
        pp = wx.FlexGridSizer(row, width)
        add_table_header(panel, pp, params_d)
        return pp
    
    
    def __init__(self, *args, **kw):
        super(PointPanel, self).__init__(*args, **kw)
        self.rows = 10

        self.header1 = [('theta', [u'θ [°]']), ('phi', [u'φ [°]']), ('t', [u't [yrs]']), ('orbit', [u'orbital pos [°]'])]
        self.header2 = [("Ttt", [u'Stt [kPa]']), ("Tpt", [u'Spt [kPa]']), ("Tpp", [u'Spp [kPa]'])]
        self.header3 = [("s1", [u'σ1 [kPa]']), ("s3", [u'σ3 [kPa]']), ("a", [u'α [°]'])]
        self.headers = self.header1+self.header2+self.header3
        
        sz = wx.BoxSizer(orient=wx.VERTICAL)
        
        sz.Add(WrapStaticText(self, label=
        u'This tab is for calculating the stress tensor at a location at the surface ' +\
        u'at a point in the orbit. It uses the Stresses tab to determine which ' +\
        u'stresses are being calculated.'), flag=wx.ALL|wx.EXPAND)

        sz.AddSpacer(20)
        self.fieldPanel = wx.scrolledpanel.ScrolledPanel(self,-1, size=(1000,400), style=wx.SIMPLE_BORDER)
        self.fieldPanel.SetupScrolling()

        rsz = wx.BoxSizer(orient=wx.HORIZONTAL)

        p2 = wx.BoxSizer(orient=wx.VERTICAL)
        cp = wx.BoxSizer(orient=wx.HORIZONTAL)
        p0 = wx.BoxSizer(orient=wx.VERTICAL)
        p0.Add(wx.StaticText(self.fieldPanel, label=u'Time/space location'), flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.pp = self.params_grid(self.fieldPanel, self.header1, '0', width=4, row=self.rows)
        p0.Add(self.pp)
        cp.Add(p0)
        p1 = wx.BoxSizer(orient=wx.VERTICAL)
        p1.Add(wx.StaticText(self.fieldPanel, label=u'Stress Tensor at a point'), flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.tp = self.params_grid(self.fieldPanel,self.header2, '', row = self.rows)
        p1.Add(self.tp, 1, wx.ALL|wx.EXPAND)
        cp.AddSpacer(15)
        cp.Add(p1)
        p3 = wx.BoxSizer(orient=wx.VERTICAL)
        p3.Add(wx.StaticText(self.fieldPanel, label=u'principal Components'), flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.sp = self.params_grid(self.fieldPanel,self.header3, '', row = self.rows)
        p3.Add(self.sp, 1, wx.ALL|wx.EXPAND)
        cp.Add(p3)
        p2.Add(cp)


        rsz.Add(p2)
        self.fieldPanel.SetSizer(rsz)
        sz.Add(self.fieldPanel)

        bp = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.row_ctrl = wx.SpinCtrl(self, min = 1, value = str(self.rows), style=wx.TE_PROCESS_ENTER)

        self.save_b = wx.Button(self, label=u'Save to File')
        self.b = wx.Button(self, label=u'Calculate Stress')
        self.load_b = wx.Button(self, label=u'Load from file')
        bp.Add(self.b, 1, wx.ALL|wx.EXPAND, 3)
        bp.Add(self.load_b, 1, wx.ALL|wx.EXPAND, 3)
        bp.Add(self.save_b, 1, wx.ALL|wx.EXPAND, 3)

        bp.Add(WrapStaticText(self, label=u'Rows: '), flag = wx.ALIGN_CENTER_VERTICAL)
        bp.Add(self.row_ctrl)
        sz.Add(bp)
     

        sz.AddSpacer(15)

        self.SetSizer(sz)

       # self.model.set_parameter('point_rows',self.rows)

class GridCalcPanel(SatPanel):
    """
    Defines the grid panel of the GUI
    """
    def __init__(self, *args, **kw):
        super(GridCalcPanel, self).__init__(*args, **kw)


        self.grid_vars_d = [
            ("MIN", u'Minimum value'),
            ("MAX", u'Maximum value'),
            ("NUM", u'Number of grid points')]

        self.grid_parameters_d = [
            ("LAT", u'Latitude'),
            ("LON", u'Longitude'),
            ("TIME", u'Time (Periapse = 0)'),
            ("ORBIT", u'Orbital position (Periapse = 0) [°]'),
            ("NSR_PERIOD", u'NSR period'),
            ("POLE_POSITION", u'Initial North Pole Location')]
        sz = wx.BoxSizer(orient=wx.VERTICAL)

        grid_id_p = wx.BoxSizer(orient=wx.HORIZONTAL)
        self.make_text_controls_with_labels(self, grid_id_p, [('GRID_ID', u"Grid ID")])

        self.sb = wx.Button(self, label=u"Save grid")
        self.lb = wx.Button(self, label=u"Load grid")
        
        gmcp = wx.FlexGridSizer(0, len(self.grid_vars_d) + 1)
        # grid points
        self.add_static_texts(self, gmcp, [('','')] + self.grid_vars_d)
        for p, d in self.grid_parameters_d[:2]:
            gmcp.Add(wx.StaticText(self, label=d))
            self.make_text_controls(self, gmcp, [ ("%s_%s" % (p,v), '') for v, dv in self.grid_vars_d ])
        for i in range(4):
            gmcp.AddSpacer(20)
        # orbital
        self.orbit_labels = self.add_static_texts(self, gmcp, [('',''), ('',u'Minimum'), ('',u'Maximum'), ('',u'Number of increments')])
        p, d = self.grid_parameters_d[3]
        self.orbit_labels.append(wx.StaticText(self, label=d))
        gmcp.Add(self.orbit_labels[-1])
        self.make_text_controls(self, gmcp, [('%s_%s' % (p,v), '') for v,d1 in self.grid_vars_d ])
        # nsr time
        for i in range(4):
            gmcp.AddSpacer(20)
        self.nsr_labels = self.add_static_texts(self, gmcp,
            [('', ''), ('', u'Start Time [yrs]'), ('', u'End Time [yrs]'), ('', u'Number of increments')])
        self.nsr_labels.append(wx.StaticText(self, label=u'Amount of NSR build up'))
        gmcp.Add(self.nsr_labels[-1])
        self.make_text_controls(self, gmcp, [ ('TIME_MIN', ''), ('nsr_time', ''), ('TIME_NUM', '') ])
        # Polar Wander
        for i in range(4):
            gmcp.AddSpacer(20)
        self.pw_labels = self.add_static_texts(self, gmcp,
            [('',''), ('', u'Final Pole Latitude [°]'), ('',u'Final Pole Longitude [°]'), ('',u'Number of increments')])
        self.pw_labels.append(wx.StaticText(self, label=u'Final Pole Location'))
        gmcp.Add(self.pw_labels[-1])
        self.make_text_controls(self, gmcp, [ ('FINAL_LAT', ''), ('FINAL_LONG', ''), ('NUM_INCREMENTS', '') ])
        top = wx.BoxSizer(orient=wx.HORIZONTAL)
        top.Add(grid_id_p)
        top.AddSpacer(6)
        top.Add(self.lb)
        top.AddSpacer(6)
        top.Add(self.sb)

        sz.Add(top)
        sz.AddSpacer(15)
        sz.Add(gmcp)
        sz.AddSpacer(15)
        sz.Add(wx.StaticText(self, label = u'Note: Number of latitude and longitude grid points must be equal'))

        self.SetSizer(sz)
        self.updating_range = False


        


###PLOT PANEL


class KPaFormatter(matplotlib.ticker.Formatter):
    def __call__(self, x, pos):
        return "%.f kPa" % (x/1000)



class StepSlider(matplotlib.widgets.Slider):
    """
    Custom designed class for discrete slider control at bottom of plot panel to control 
    satellite's orbital position.

    Used in add_orbit_controls and add_nsr_controls
    """
    def __init__(self, ax, label, valmin, valmax, numsteps, *args, **kw):
        self.steps_n = numsteps
        self.updating = False
        self.prev_val = kw.get('valinit', 0)
        matplotlib.widgets.Slider.__init__(self, ax, label, valmin, valmax, *args, **kw)
        ax.lines.remove(self.vline)
    
    def on_changed_f(self, val):
        pass

    def on_changed(self, f):
        def f2(val):
            if self.updating:
                return
            self.eventson = False
            self.updating = True
            val += self.valmin
            self.set_stepval(val)
            f(self.val)
            self.updating = False
            self.eventson = True
        
        self.on_changed_f = f
        matplotlib.widgets.Slider.on_changed(self, f2)
        
    def set_stepval(self, val):
        if val < self.valmin:

            self.set_val(self.valmin)
        elif val > self.valmax:
            #print 'val >'
            self.set_val(self.valmax)
        elif self.valmax - self.valmin > 0 and self.steps_n > 0:
            #print 'elseif'
            step = float(self.valmax - self.valmin)/self.steps_n
            n0 = int((val - self.valmin)/step)
            n1 = n0 + 1
            if abs(val - self.prev_val) > 0.7*step:
                self.prev_val = round((val - self.valmin)/step)*step + self.valmin
                self.set_val(self.prev_val)
            else:
                self.set_val(self.prev_val)

    def reset(self):
        self.updating = True
        matplotlib.widgets.Slider.reset(self)
        self.updating = False
    
    def first(self):
        self.set_stepval(self.valmin)
        # HERE initial_split()
    
    def last(self):
        self.set_stepval(self.valmax)
        # HERE EVERYTHING SHOULD BE GRAPHED
    
    def next(self):
    	print self.valmax
        step = float(self.valmax - self.valmin)/self.steps_n
        n = int((self.val - self.valmin)/step) + 1
        self.set_stepval(n*step + self.valmin)
        # ONLY GRAPH UP TO THIS POINT
    
    def prev(self):
        step = float(self.valmax - self.valmin)/self.steps_n
        n = int((self.val - self.valmin)/step) - 1
        self.set_stepval(n*step + self.valmin)
        # ONLY GRAPH UP TO THIS POINT



class CustomPlotToolbar(NavigationToolbar):
    def __init__(self, plotCanvase):
        # create default toolbar
        NavigationToolbar.__init__(self, plotCanvase)
        
        # remove unwanted button
        # stress plot only exists in rectangular bounds
        # may need to add in later if want movable scale bar
        # or only pan when zoommed
        POSITION_OF_PANNING_BUTTON = 3
        
        # remove unnecessary button (no subplots)
        POSITION_OF_CONFIGURE_SUBPLOT_BUTTON = 6
        self.DeleteToolByPos(POSITION_OF_CONFIGURE_SUBPLOT_BUTTON)


class MatPlotPanel(wx.Panel):
    """
    GUI object that holds the plot area
    """

    def __init__(self, *args, **kw):
        super(MatPlotPanel, self).__init__(*args, **kw)
        self.figure = Figure(figsize=(6,5),dpi=display_dpi)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.ax = self.figure.add_subplot(111)

        toolbar = CustomPlotToolbar(self.canvas)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, flag=wx.EXPAND|wx.ALL)
        toolbar.Realize()
        sizer.Add(toolbar, flag=wx.EXPAND|wx.ALL)
        self.SetSizer(sizer)
        self.SetMinSize((625, 400))

    def get_axes(self):
        return self.ax
    
    def draw(self):
        self.canvas.draw()

    def colorbar(self, mappable, *a, **kw):
        #return self.figure.colorbar(mappable, ax=self.ax, *a, **kw)
        return self.figure.colorbar(mappable, *a, **kw)


class StressPlotPanel(MatPlotPanel):
    """
    Contains controls for going through the time frame dictated in "Grid" Tab.
    Specifically, the [< | < | > | >] controls
    """
    scale_y    = 0.15
    orbit_y    = 0.11
    polar_y = 0.01
    nsr_y    = 0.06
    button_l = 0.04
    bbutton_l= 0.12
    slider_h = 0.04
    slider_x = scale_left + scale_bar_length + button_l*2

    def __init__(self, *args, **kw):
        super(StressPlotPanel, self).__init__(*args, **kw)
        #
        self.figure.subplots_adjust(bottom=0.25)
        # creates scale bar for the vectors (arrows) i.e. |-----| 91 kPa
        self.scale_ax = self.figure.add_axes([scale_left, self.scale_y, scale_bar_length, self.slider_h], frame_on=False)
        #
        self.add_orbit()
        self.add_polar()
        self.add_nsr()

    def get_ax_orbit(self):
        return self.figure.add_axes([scale_left, self.orbit_y, scale_bar_length, self.slider_h])

    def get_ax_nsr(self):
        return self.figure.add_axes([scale_left, self.nsr_y, scale_bar_length, self.slider_h])
    
    def get_ax_polar(self):
        return self.figure.add_axes([scale_left, self.polar_y, scale_bar_length, self.slider_h])

    def del_orbit(self):
        self.figure.delaxes(self.ax_orbit)
        self.del_orbit_controls()

    def del_nsr(self):
        self.figure.delaxes(self.ax_nsr)
        self.del_nsr_controls()
    
    def del_polar(self):
        self.figure.delaxes(self.ax_polar)
        self.del_polar_controls()

    def del_orbit_controls(self):
        for a in [self.ax_orbit_first, self.ax_orbit_prev, \
            self.ax_orbit_next, self.ax_orbit_last, self.ax_orbit_save]:
            self.figure.delaxes(a)

    def del_nsr_controls(self):
        for a in [self.ax_nsr_first, self.ax_nsr_prev, \
            self.ax_nsr_next, self.ax_nsr_last, self.ax_nsr_save]:
            self.figure.delaxes(a)


    def del_polar_controls(self):
        for a in [self.ax_polar_first, self.ax_polar_prev, \
            self.ax_polar_next, self.ax_polar_last, self.ax_polar_save]:
            self.figure.delaxes(a)

    def add_orbit(self):
        self.ax_orbit = self.get_ax_orbit()
        self.add_orbit_controls()
    
    def add_orbit_controls(self):
        x = self.slider_x
        self.ax_orbit_first = self.figure.add_axes([x, self.orbit_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_orbit_prev = self.figure.add_axes([x, self.orbit_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_orbit_next = self.figure.add_axes([x, self.orbit_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_orbit_last = self.figure.add_axes([x, self.orbit_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_orbit_save = self.figure.add_axes([x, self.orbit_y, self.bbutton_l, self.slider_h])
        
        # Note: StepSlider is custom designed class in/for gui
        self.orbit_slider = StepSlider(self.ax_orbit, 'Orbital position', 0, 1, 10, valinit=0, dragging=False)
        
        self.orbit_first_button = matplotlib.widgets.Button(self.ax_orbit_first, '[<')
        self.orbit_prev_button = matplotlib.widgets.Button(self.ax_orbit_prev, '<')
        self.orbit_next_button = matplotlib.widgets.Button(self.ax_orbit_next, '>')
        self.orbit_last_button = matplotlib.widgets.Button(self.ax_orbit_last, '>]')
        self.orbit_save_button = matplotlib.widgets.Button(self.ax_orbit_save, 'Save series')
        
        self.orbit_first_button.on_clicked(lambda e: self.orbit_slider.first())  # lambda functions
        self.orbit_prev_button.on_clicked(lambda e: self.orbit_slider.prev())    # ok, so are empirically necessary, but why?
        self.orbit_next_button.on_clicked(lambda e: self.orbit_slider.next())
        self.orbit_last_button.on_clicked(lambda e: self.orbit_slider.last())
        # hack
        self.orbit_save_button.on_clicked(lambda e: wx.CallLater(125, self.on_save_orbit_series, e))
    
    def add_polar(self):
        self.ax_polar = self.get_ax_polar()
        self.add_polar_controls()
    
    def add_polar_controls(self):
        x = self.slider_x
        self.ax_polar_first = self.figure.add_axes([x, self.polar_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_polar_prev = self.figure.add_axes([x, self.polar_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_polar_next = self.figure.add_axes([x, self.polar_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_polar_last = self.figure.add_axes([x, self.polar_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_polar_save = self.figure.add_axes([x, self.polar_y, self.bbutton_l, self.slider_h])
        
        # Note: StepSlider is custom designed class in/for gui
        self.polar_slider = StepSlider(self.ax_polar, 'Polar position', 0, 1, 10, valinit=0, dragging=False)
        
        self.polar_first_button = matplotlib.widgets.Button(self.ax_polar_first, '[<')
        self.polar_prev_button = matplotlib.widgets.Button(self.ax_polar_prev, '<')
        self.polar_next_button = matplotlib.widgets.Button(self.ax_polar_next, '>')
        self.polar_last_button = matplotlib.widgets.Button(self.ax_polar_last, '>]')
        self.polar_save_button = matplotlib.widgets.Button(self.ax_polar_save, 'Save series')
        
        self.polar_first_button.on_clicked(lambda e: self.polar_slider.first())
        self.polar_prev_button.on_clicked(lambda e: self.polar_slider.prev())
        self.polar_next_button.on_clicked(lambda e: self.polar_slider.next())
        self.polar_last_button.on_clicked(lambda e: self.polar_slider.last())
        # hack
        self.polar_save_button.on_clicked(lambda e: wx.CallLater(125, self.on_save_polar_series, e))
    
    
    def add_nsr(self):
        self.ax_nsr = self.get_ax_nsr()
        self.add_nsr_controls()
    
    def add_nsr_controls(self):
        x = self.slider_x
        self.ax_nsr_first = self.figure.add_axes([x, self.nsr_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_nsr_prev = self.figure.add_axes([x, self.nsr_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_nsr_next = self.figure.add_axes([x, self.nsr_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_nsr_last = self.figure.add_axes([x, self.nsr_y, self.button_l, self.slider_h])
        x += self.button_l
        self.ax_nsr_save = self.figure.add_axes([x, self.nsr_y, self.bbutton_l, self.slider_h])
        self.nsr_slider = StepSlider(self.ax_nsr, 'NSR position', 0, 1, 10, valinit=0, dragging=False, valfmt="%.1g")
        self.nsr_first_button = matplotlib.widgets.Button(self.ax_nsr_first, '[<')
        self.nsr_prev_button = matplotlib.widgets.Button(self.ax_nsr_prev, '<')
        self.nsr_next_button = matplotlib.widgets.Button(self.ax_nsr_next, '>')
        self.nsr_last_button = matplotlib.widgets.Button(self.ax_nsr_last, '>]')
        self.nsr_first_button.on_clicked(lambda e: self.nsr_slider.first())
        self.nsr_prev_button.on_clicked(lambda e: self.nsr_slider.prev())
        self.nsr_next_button.on_clicked(lambda e: self.nsr_slider.next())
        self.nsr_last_button.on_clicked(lambda e: self.nsr_slider.last())
        self.nsr_save_button = matplotlib.widgets.Button(self.ax_nsr_save, 'Save series')
        # hack
        self.nsr_save_button.on_clicked(lambda e: wx.CallLater(125, self.on_save_nsr_series, e))

    def change_slider(self, ax, slider, label=None, valmin=None, valmax=None, numsteps=None, valinit=None, valfmt=None):
        if label is None:
            label = slider.label.get_text()
        if valmin is None:
            valmin = slider.valmin
        if valmax is None:
            valmax = slider.valmax
        if numsteps is None:
            numsteps = slider.numsteps
        if valinit is None:
            valinit = slider.valinit
        if valfmt is None:
            valfmt = slider.valfmt
        f = slider.on_changed_f
        slider = StepSlider(ax, label, valmin, valmax, numsteps, valinit=valinit, dragging=False, valfmt=valfmt)
        slider.on_changed(f)
        return slider
    
    def change_orbit_slider(self, valmin, valmax, numsteps, valinit=None):
        if valinit is None:
            valinit = valmin
        self.figure.delaxes(self.ax_orbit)
        self.ax_orbit = self.get_ax_orbit()
        self.orbit_slider = self.change_slider(
            self.ax_orbit, self.orbit_slider, valmin=valmin, valmax=valmax, numsteps=numsteps, valinit=valinit)

    def change_nsr_slider(self, valmin, valmax, numsteps, valinit=None):
        if valinit is None:
            valinit = valmin
        self.figure.delaxes(self.ax_nsr)
        self.ax_nsr = self.get_ax_nsr()
        self.nsr_slider = self.change_slider(
            self.ax_nsr, self.nsr_slider, valmin=valmin, valmax=valmax, numsteps=numsteps, valinit=valmin, valfmt="%.1g")


    def change_polar_slider(self, valmin, valmax, numsteps, valinit=None):
        if valinit is None:
            valinit = valmin
        self.figure.delaxes(self.ax_polar)
        self.ax_polar = self.get_ax_polar()
        self.polar_slider = self.change_slider(
                                             self.ax_polar, self.polar_slider, valmin=valmin, valmax=valmax, numsteps=numsteps, valinit=valmin, valfmt="%.1g")

    def plot_scale(self, scale, valfmt):
        self.scale_ax.clear()
        while self.scale_ax.texts:
            self.scale_ax.texts.pop()
        self.scale_ax.set_xticks([])
        self.scale_ax.set_yticks([])
        self.scale_ax.text(-0.02, 0.5, 'Scale', transform=self.scale_ax.transAxes, va='center', ha='right')
        self.scale_ax.text(0.23, 0.5, valfmt % scale, transform=self.scale_ax.transAxes, va='center', ha='left')
        self.scale_ax.plot([0.00, 0.20], [0.5, 0.5], linestyle='solid', marker='|', color='black', lw=1)
        self.scale_ax.set_xlim(0.0, 1.0)

    def on_save_orbit_series(self, evt):
        try:
            dir_dialog(None, 
                message=u"Save calculation series on orbit period",
                style=wx.SAVE,
                action=self.save_orbit_series)
        except LocalError, e:
            error_dialog(self, str(e), e.title)

    def on_save_nsr_series(self, evt):
        try:
            dir_dialog(self,
                message=u"Save calculation series on nsr period",
                style=wx.SAVE,
                action=self.save_nsr_series)
        except LocalError, e:
            error_dialog(self, str(e), e.title)


    def on_save_polar_series(self, evt):
        try:
            dir_dialog(self,
                       message=u"Save calculation series on nsr period",
                       style=wx.SAVE,
                       action=self.save_polar_series)
        except LocalError, e:
            error_dialog(self, str(e), e.title)




class CycloidsPanel(SatPanel):
    """
    Defines the cycloids panel of the GUI

    NTS: Should restrict decimal places at some pt

    """
    def __init__(self, *args, **kw):
        super(CycloidsPanel, self).__init__(*args, **kw)
        sz = wx.BoxSizer(wx.VERTICAL)
        gridSizer = wx.FlexGridSizer(rows=7, cols=2, hgap=5, vgap=0)
        dirSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        filler = wx.BoxSizer(wx.HORIZONTAL)
        varyvSizer = wx.BoxSizer(wx.HORIZONTAL)
        which_dir = wx.StaticText(self, wx.ID_ANY, 'Propagation Direction: ')
        dirSizer.Add(which_dir, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        all_dir = ['East', 'West']
        self.start_dir = wx.ComboBox(self, size=(100, 50) ,choices=all_dir, style=wx.CB_DROPDOWN|wx.CB_READONLY)
        # bind
      
        # create load/save buttons
        self.save_bt = wx.Button(self, label='Save to file')
        self.load_bt = wx.Button(self, label='Load from file')
        buttonSizer.Add(self.load_bt, wx.ALIGN_CENTER, 10)
        buttonSizer.AddSpacer(5)
        buttonSizer.Add(self.save_bt, wx.ALIGN_CENTER)


        self.vary = wx.CheckBox(self, wx.ID_ANY, 'Vary Velocity   k = ')



        self.constant = wx.TextCtrl(self, wx.ID_ANY, '0', style=wx.TE_PROCESS_ENTER)
        self.constant.Disable()

        

        self.use_multiple = wx.CheckBox(self, wx.ID_ANY, 'Use loaded CSV file')

        varyvSizer.Add(self.vary, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        varyvSizer.Add(self.constant, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        
        # add widgets into grid
        # Set the TextCtrl to expand on resize
        
        
        fieldsToAdd = [('YIELD', 'Yield (Threshold) [kPa]: '),('PROPAGATION_STRENGTH','Propagation Strength [kPa]: '),('PROPAGATION_SPEED','Propagation Speed [m/s]: '), ('STARTING_LONGITUDE', 'Starting Longitude: '), ('STARTING_LATITUDE', 'Starting Latitude: ')]

        self.make_text_controls_with_labels(self, gridSizer, fieldsToAdd)
     
     
        gridSizer.Add(dirSizer)
        gridSizer.Add(self.start_dir)
        gridSizer.Add(varyvSizer)
        self.many_params = wx.Button(self, label='Load Multiple Cycloid Parameters')



        sz.Add(WrapStaticText(self,
            label=u'This tab calculates cycloids through combined diurnal and NSR stresses. Cycloids ' +
            u'are arcuate lineaments found on the surface of Europa. ' +
            u'They can be modeled and plotted on the following ' +
            u'Plot tab. The Yield Strength is the threshold that initiates fracture in the ice. ' +
            u'This fracture will propagate as long as the strength is below this threshold and greater than the ' +
            u'Propagation Strength. The Propagation Speed is usually <10 m/s. ' +
            u'For further information on cycloids see the Help menu.'),
            flag=wx.ALL|wx.EXPAND)
            #sz.Add(filler)
        
        sz.Add(buttonSizer, 0, wx.ALL, 5)
        #sz.Add(filler2)
        sz.Add(gridSizer, 0, wx.ALL|wx.EXPAND, 5)
        sz.Add(self.use_multiple,0, wx.ALL|wx.EXPAND, 5)
        sz.Add(self.many_params)
        self.SetSizer(sz)
        sz.Fit(self)


class ScalarPlotPanel(SatPanel):
    """
    Defines the plot panel of the GUI in terms of PlotPanel, which is in term of SatPanel
    """
    step_field = 'SCALAR_PLOT_STEP'


    def __init__(self, *args, **kw):
        super(ScalarPlotPanel, self).__init__(*args, **kw)
        self.load_step()

        self.orbit_hidden = self.nsr_hidden = self.polar_hidden = False
        #self.orbit_hidden = self.nsr_hidden = False
        self.projection_changed = False
        main_sz = wx.BoxSizer(orient=wx.VERTICAL)

        main_sz.Add(self.head_text(), flag=wx.EXPAND|wx.ALL)
        main_sz.AddSpacer(5)
        main_sz.Add(self.plot_sizer(), flag=wx.EXPAND|wx.ALL)
        main_sz.AddSpacer(5)

        main_sz.Add(self.lineaments_sizer())
        main_sz.AddSpacer(5)
        main_sz.Add(wx.StaticLine(self), 0, wx.ALL|wx.EXPAND, 5)
        main_sz.AddSpacer(5)
        main_sz.Add(self.cycloids_sizer())

        self.SetSizer(main_sz)
        self.Fit()
    
        self.changed = True
    
    
    
    def add_stepspin(self, sz):
        sz.Add(wx.StaticText(self, label=u"Tick mark increment"), flag=wx.ALIGN_CENTER_VERTICAL)
        self.stepspin = wx.SpinCtrl(self, initial=int(self.step), min=0, max=180)
        sz.Add(self.stepspin, flag=wx.ALL|wx.EXPAND)
        self.stepspin.SetValue(self.step)
        self.stepspin.Bind(wx.EVT_SPINCTRL, self.adjust_step)
    
    def adjust_step(self, evt):
        self.adjust_coord_step(self.stepspin.GetValue())
    
    def load_step(self):
        self.step = config.load_step(self.step_field)
        return self.step
    def save_step(self):
        config.save_step(self.step, self.step_field)
    
    def plot(self):
    
        self.plot_no_draw()
        self.draw()
        '''
        except LocalError, e:
                print self.model.grid_set() or self.model.satellite_set()
                if not (self.model.grid_set() or self.model.satellite_set()):
                    error_dialog(self, str(e), e.title)

        except Exception, e:
            print e
            if self.model.grid_set() and self.model.satellite_set():
                if not self.sc.get_stresses():
                    traceback.print_exc()
                    error_dialog(self, 'Stresses are not defined', 'Plot Error')
                else:
                    traceback.print_exc()
                    error_dialog(self, e.__class__.__name__ + ': ' + str(e), "Plot Error")
        '''
    def plot_no_draw(self):
        print 'im here'
        self.grid = self.model.get_grid()
        self.calc = self.model.get_calc()
        self.basemap_ax = self.get_basemap_ax()
        self.plot_grid_calc()
        self.draw_coords()
    
    def basemap_parameters(self, proj):
        p = { 'projection': proj }
        if proj in ['cyl', 'mill', 'merc']:
            if proj == 'merc':
                if self.model.get_param_value('LAT_MIN') <= -89.9:
                    self.model.set_parameter('LAT_MIN',-89.9)
                if self.model.get_param_value('LAT_MAX') >= 89.9:
                    self.model.set_parameter('LAT_MAX',  89.9)
            p.update({
                     'llcrnrlon': self.model.get_param_value('LON_MIN'),
                     'llcrnrlat': self.model.get_param_value('LAT_MIN'),
                     'urcrnrlon': self.model.get_param_value('LON_MAX'),
                     'urcrnrlat': self.model.get_param_value('LAT_MAX')})
        elif proj == 'ortho':
            p.update({
                     'lat_0': int(round((self.model.get_param_value('LAT_MIN')+self.model.get_param_value('LAT_MAX'))/2)),
                     'lon_0': int(round((self.model.get_param_value('LON_MIN')+self.model.get_param_value('LON_MAX'))/2))})
        else:
            p.update({'boundinglat': 0,
                     'lat_0': (self.model.get_param_value('LAT_MIN')+self.model.get_param_value('LAT_MAX'))/2,
                     'lon_0': (self.model.get_param_value('LON_MIN')+self.model.get_param_value('LON_MAX'))/2})
        
        return p
    
    def get_basemap_ax(self):
        ax = self.get_axes()
        ax.clear()
        p = self.basemap_parameters(self.model.get_param_value('projection'))
        p.update({'resolution': None, 'ax': ax})
        basemap_ax = basemap.Basemap(**p)
        return basemap_ax
    
    def draw_coords(self):
        # Draw a grid onto the plot -- independent of actual grid tab
        coord_lons  = numpy.arange(
        numpy.radians(self.model.get_param_value('LON_MIN')),
        numpy.radians(self.model.get_param_value('LON_MAX')),
        numpy.radians(self.step))
        coord_lons = numpy.resize(coord_lons, coord_lons.size + 1)
        coord_lons.put(coord_lons.size - 1, numpy.radians(self.model.get_param_value('LON_MAX')))
        coord_lats  = numpy.arange(
        numpy.radians(self.model.get_param_value('LAT_MIN')),
        numpy.radians(self.model.get_param_value('LAT_MAX')),
        numpy.radians(self.step))
        coord_lats = numpy.resize(coord_lats, coord_lats.size + 1)
        coord_lats.put(coord_lats.size - 1, numpy.radians(self.model.get_param_value('LAT_MAX')))
        parallel_labels = [1,0,0,1]
        parallel_xoffset = 0
        self.meridians = self.basemap_ax.drawmeridians(numpy.around(numpy.degrees(coord_lons)),
        labels=[1,0,0,1], linewidth=0.5, color='gray', yoffset=5)
        self.parallels = self.basemap_ax.drawparallels(numpy.around(numpy.degrees(coord_lats)),
        labels=parallel_labels, linewidth=0.5, color='gray', xoffset=parallel_xoffset)
        self.basemap_ax.drawmapboundary()

    def adjust_coord_step(self, step):
        """
            Change tick step of coordinate axes.
            """
        self.step = step
        self.save_step()
        #ax = self.get_axes()
        def clear(a):
            for ll, tt in a:
                map(self.ax.lines.remove, ll)
                map(self.ax.texts.remove, tt)
        clear(self.meridians.values())
        clear(self.parallels.values())
        self.plot()


    def head_text(self):
        return WrapStaticText(self,
            label=u"Display a rasterized scalar stress field defined by calculation on " +\
            u"satellite and grid parameters at the resolution defined by grid.  " +\
            u"Tension is positive\n ")

    def plot_sizer(self):
        self.plot_fields = {}
        self.plot_vectors = {}
        self.n_interp = 10

        self.tick_formatter = KPaFormatter()

        s = wx.BoxSizer(orient=wx.HORIZONTAL)

        self.scp = self.stress_plot_panel()
        self.init_orbit_slider()
        self.init_nsr_slider()
        #self.init_polar_slider()
        
        p = self.parameters_sizer()

        s.Add(self.scp, flag=wx.ALL|wx.EXPAND)
        s.AddSpacer(10)
        s.Add(p, flag=wx.ALL|wx.EXPAND)
        
        return s

    def stress_plot_panel(self):
        scp = StressPlotPanel(self)
        scp.canvas.callbacks.connect('motion_notify_event', self.on_move_in_plot)
        scp.orbit_slider.on_changed(self.on_orbit_updated)
        scp.nsr_slider.on_changed(self.on_nsr_updated)
        #scp.polar_slider.on_changed(self.on_polar_updated)
        scp.save_orbit_series = self.save_orbit_series
        scp.save_nsr_series = self.save_nsr_series
        #scp.save_polar_series = self.save_polar_series
        self.orbit_pos = int(self.model.get_parameter('ORBIT_MIN').get_value())
        self.nsr_pos = float(self.model.get_parameter('TIME_MIN').get_value())
        self.polar_pos = float(self.model.get_parameter('TIME_MIN').get_value())
        #self.polar_pos = self.sc.get_parameter(float,'TIME_MIN',0)
        self.updating = False
        scp.Fit()
        return scp

    # sets up the controls and cells to the right of plot in PlotPanel
    def parameters_sizer(self):
        lp = wx.BoxSizer(orient=wx.VERTICAL)

        # layout as two vertical columns (not sure about row parameter)
        spp1 = wx.FlexGridSizer(rows=1, cols=2)
       
        
        # Adds widget controlling projection type
        self.add_projection(spp1)
        # Adds tick mar increment widget
        self.add_stepspin(spp1)
        # Adds plot direction widget
        self.add_direction(spp1)
        # Adds blank space
        spp1.AddSpacer(10)
        spp1.AddSpacer(10)
        
        # Adds stress range (upper/lower bound included) widget
        self.add_scalectrl(spp1)

        spp1.AddSpacer(15)
        spp1.AddSpacer(15)

        # not sure what this does, but is necessary for plotting
        self.add_stress_field(spp1)

        spp1.Add(wx.StaticText(self, label=u'Plot stresses'), flag=wx.ALIGN_TOP)
        spp2 = wx.FlexGridSizer(rows=9, cols=1)
        # Adds set of radiobuttoms
        self.add_to_plot_stresses(spp2)
        spp1.Add(spp2)
        
        self.scp.plot_scale(self.scale(), "%.f kPa")

        self.ax = self.scp.get_axes()

        spp1.AddSpacer(15)
        spp1.AddSpacer(15)
        # adds widget displaying long, lat, and stress at cursor
        self.add_value_display(spp1)

        lp.Add(spp1)
        lp.AddSpacer(15)
        
        lp.Add(wx.StaticLine(self), 0, wx.ALL|wx.EXPAND, 5)

        return lp

    def update_parameters(self):
        self.show_needed_sliders()
        super(ScalarPlotPanel, self).update_parameters()

    def scale_spin(self, k):
        self.load_scale(k)
        if k < 0 and self.lbound is None:
            self.lbound = -100
        elif k > 0 and self.ubound is None:
            self.ubound = 100
        ctrl = wx.SpinCtrl(self, min=-100000000, max=100000000)
        if k < 0:
            ctrl.SetValue(int(self.lbound))
        else:
            ctrl.SetValue(int(self.ubound))
        ctrl.Bind(wx.EVT_SPINCTRL, self.select_scale)
        return ctrl

    #@into_hbox
    def add_projection(self, sizer):
        self.make_combobox2_controls(self, sizer, 'projection', u'Display projection',
        [('cyl', u'Cylindrical Equidistant'),
        ('mill', u'Miller Cylindrical'),
        ('merc', u'Mercator'),
        ('ortho', u'Orthographic'),
        ('npaeqd', u'North-Polar'),
        ('spaeqd', u'South-Polar')])

    def add_direction(self, sizer):
        self.make_combobox2_controls(self, sizer, 'direction', u'Plot direction',[('east', u'East Positive'), ('west', u'West Positive')])

    #@into_hbox
    # function for adding color scalebar/legend of stress plot
    def add_scalectrl(self, sizer):
        sizer.Add(wx.StaticText(self, label=u"Stress range:"), flag=wx.ALIGN_CENTER_VERTICAL)
        sizer.AddSpacer(15)
        sizer.Add(wx.StaticText(self, label=u"Lower bound [kPa]"), flag=wx.ALIGN_CENTER_VERTICAL)
        self.lbound_ctrl = self.scale_spin(-1)
        sizer.Add(self.lbound_ctrl, flag=wx.ALL|wx.EXPAND)
        sizer.Add(wx.StaticText(self, label=u"Upper bound [kPa]"), flag=wx.ALIGN_CENTER_VERTICAL)
        self.ubound_ctrl = self.scale_spin(1)
        sizer.Add(self.ubound_ctrl, flag=wx.ALL|wx.EXPAND)


    def add_stress_field(self, sizer):
        self.make_combobox2_controls(self, sizer, 'field', u'Plot gradient',
        [('tens', u'σ1'),
        ('comp', u'σ3'),
        ('mean', u'(σ1 + σ3)/2'),
        ('diff', u'σ1 - σ3'),
        (None, u'None')])



    def add_to_plot_stresses(self, sizer):
        self.model.set_parameter('to_plot_principal_vectors', True)
        self.model.set_parameter('to_plot_shear_vectors', False)
        self.model.set_parameter('to_plot_longitude_vectors', False)
        self.model.set_parameter('to_plot_latitude_vectors', False)
        self.make_checkbox_controls(self, sizer,
        [('to_plot_principal_vectors', u'principal'),
        ('to_plot_latitude_vectors', u'latitude'),
        ('to_plot_longitude_vectors', u'longitude'),
        ('to_plot_shear_vectors', u'shear')])

    def add_value_display(self, sizer):
        self.val_p = self.make_text_controls(self, sizer,
        [ ('LAT', u'Latitude:'), ('LON', u'Longitude:'),('VAL', u'Stress [kPa]:')])


    ###########################################################################
    # Plot Tab Load/Save buttons for lineament and cycloids and helper functions
    def load_save_buttons(self):
        """
        creates and bind the buttons for loading and saving files
        """
        gridSizer = wx.FlexGridSizer(rows=2, cols=2, hgap=15, vgap=5)

        # create and bind buttons
        shapeLoad = wx.Button(self, label=u'Load from shape file')
        shapeLoad.Bind(wx.EVT_BUTTON, self.on_load_shape)
        
        shapeSave = wx.Button(self, label=u'Save as shape file')
        shapeSave.Bind(wx.EVT_BUTTON, self.on_save_shape)

        netLoad = wx.Button(self, label=u'Load fom NetCDF file')
        netLoad.Bind(wx.EVT_BUTTON, self.on_load_netcdf)

        netSave = wx.Button(self, label=u'Save as NetCDF file')
        netSave.Bind(wx.EVT_BUTTON, self.on_save_netcdf)

        # add widgets to grid
        gridSizer.AddMany([
            (shapeLoad, 0, wx.ALIGN_CENTER|wx.EXPAND),
            (shapeSave, 0, wx.ALIGN_CENTER|wx.EXPAND),
            (netLoad, 0, wx.ALIGN_CENTER|wx.EXPAND),
            (netSave, 0, wx.ALIGN_CENTER| wx.EXPAND)])

        return gridSizer

    def on_load_shape(self, evt):
        try:
            file_dialog(self,
                message = u"Load from shape file",
                style = wx.OPEN,
                wildcard = 'Shape files (*.shp)|*.shp',
                action = self.load_shape)
        except Exception, e:
            error_dialog(self, str(e), u'Shape Load Error')
    
    def load_shape(self, filename):
        # walk around char const * restriction
        sf = os.path.splitext(str(filename))[0] + '.shp'
        self.loaded['data'] = shp2lins(sf, stresscalc=self.calc)
        self.loaded['lines'] = []
        d = wx.ColourDialog(self, self.loaded['color'])
        if (d.ShowModal() == wx.ID_OK):
            self.loaded['color'] = d.GetColourData()
        self.plot()

    def on_save_shape(self, evt):
        file_dialog(self,
            message = u"Save to shape file",
            style = wx.SAVE | wx.OVERWRITE_PROMPT,
            wildcard = 'Shape files (*.shp)|*.shp',
            defaultFile = 'lineaments.shp',
            action = self.save_shape)

    def save_shape(self, filename):
        lins2shp(self.loaded['data'] + self.generated['data'], filename)

    def on_load_netcdf(self, evt):
        try:
            file_dialog(self,
                message=u"Load from NetCDF file",
                style=wx.OPEN,
                wildcard=u'NetCDF files (*.nc)|*.nc',
                action=self.load_netcdf)
        except LocalError, e:
            error_dialog(self, str(e), e.title)
    
    def load_netcdf(self, filename):
        self.sc.load_netcdf(filename)
        self.update_parameters()
        self.plot()

    def on_save_netcdf(self, evt):
        try:
            file_dialog(self,
                message=u"Save to NetCDF file",
                style=wx.SAVE | wx.OVERWRITE_PROMPT,
                defaultFile='gridcalc.nc',
                wildcard=u'NetCDF files (*.nc)|*.nc',
                action=self.sc.save_netcdf)
        except LocalError, e:
            error_dialog(self, str(e), e.title)

    ###########################################################################
    # Defining lineament controls and related functions
    def lineaments_sizer(self):
        """
        Defines sizer for controls for lineament plotting
        """
        # define vars
        #self.lin_p = {}  not used anywhere else, could remove
        self.l_count = 2
        self.generated = { 'data': [], 'color': wx.ColourData(), 'lines': [] }
        self.loaded = { 'data': [], 'color': wx.ColourData(), 'lines': [] }
        self.first_run = True
        self.model.set_parameter('to_plot_lineaments',True)

        # define sizers
        lins = wx.BoxSizer(wx.HORIZONTAL)
        lins_ckSizer = wx.BoxSizer(wx.HORIZONTAL)

        # setup widgets
        self.plot_lins = wx.CheckBox(self, label='Show ')
        self.view_parameters['to_plot_lineaments'] = self.plot_lins
        self.plot_lins.Bind(wx.EVT_CHECKBOX, self.generate_lins)

        self.l_count_tc = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.l_count_tc.SetValue(str(self.l_count))
        self.l_count_tc.Bind(wx.EVT_TEXT, self.generate_lins)

        # construct ckSizer
        lins_ckSizer.AddSpacer(10)
        lins_ckSizer.Add(self.plot_lins, 0, 20)
        lins_ckSizer.Add(self.l_count_tc, wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        lins_ckSizer.Add(wx.StaticText(self, label=u" Lineaments"), wx.ALL|wx.ALIGN_CENTER_VERTICAL)

        # add checkbox
        lins.Add(lins_ckSizer)
        lins.AddSpacer(15)
        # add buttons
        lins.Add(self.load_save_buttons(), wx.ALL|wx.ALIGN_RIGHT)

        return lins

    def generate_lins(self, evt):
        print 'generate_lins'
        try:
            if self.plot_lins.GetValue():     # plot only if box is checked
                self.l_count = int(self.l_count_tc.GetValue())
            else:
                self.l_count = 0

            self.first_run = False
            b = wx.BusyInfo(u"Performing calculations. Please wait.", self)
            wx.SafeYield()
            self.generated['data'] = self.lingen(self.l_count)
            self.generated['lines'] = []
            del b
            self.plot()
        except:
            self.l_count_tc.SetValue(str(self.l_count))

        self.plot_lineaments()

        print 'end generate_lins'

    def lingen(self, number):
        print 'lingen'
        ll = []
        for lat in numpy.linspace(0, numpy.radians(90), number+2)[1:-1]:
            ll.append(lingen_nsr(self.calc, init_lon=0, init_lat=lat))
        ll += [Lineament(lons=l.lons, lats=-l.lats, stresscalc=l.stresscalc) for l in ll]
        ll += [Lineament(lons=l.lons+satstress.physcon.pi, lats=l.lats, stresscalc=l.stresscalc) for l in ll]
        print ll
        return ll
 
    def plot_lineaments(self):
        for l in [self.generated, self.loaded]:
            if l['data']:
                l['lines'] = plotlinmap(l['data'], map=self.basemap_ax, color=self.mpl_color(l['color'].GetColour()))[0]

    ###########################################################################
    # Defining cycloid controls and related functions
    def cycloids_sizer(self):
        """
        Defines sizer containing controls for cycloid plotting
        """
        self.cycl_generated = { 'cycdata': [], 'color': wx.ColourData(), 'arcs': [] }
        self.cycl_loaded = { 'cycdata': [], 'color': wx.ColourData(), 'arcs': [] }
        self.first_run = True   # for lineaments

        # create sizers
        cycl = wx.BoxSizer(wx.HORIZONTAL)
        ckSizer = wx.BoxSizer(wx.VERTICAL)


        self.plot_cycl = wx.CheckBox(self, label='Show Cycloids')
        # wrap in sizer
        ckSizer.Add(self.plot_cycl, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL)
        # bind to event
        self.plot_cycl.Bind(wx.EVT_CHECKBOX, self.generate_cycl)

        saveMany = wx.Button(self, label="Save Multiple Cycloids")
        saveMany.Bind(wx.EVT_BUTTON, self.save_many_cycloids)
        ckSizer.AddSpacer(5)
        ckSizer.Add(saveMany)

        # add grid to sizer
        cycl.AddSpacer(10)
        cycl.Add(ckSizer, wx.ALL|wx.ALIGN_LEFT)
        cycl.AddSpacer(5)
        cycl.AddSpacer(15)
        cycl.Add(self.load_save_buttons(), wx.ALL|wx.ALIGN_RIGHT)

        return cycl

    def generate_cycl(self, evt):
        if self.plot_cycl.GetValue(): # plot only if box is checked
            self.model.set_parameter('to_plot_cycloids', True)
            self.plot()
        else:
            self.model.set_parameter('to_plot_cycloids', False)
            self.plot()

    def plot_cycloids(self):
        c_controller =self.controller.cp_controller
 
        if self.model.get_param_value('to_plot_many_cycloids'):
            for i, cycloid_params in enumerate(c_controller.params_for_cycloids.items()):
                if not c_controller.cycloids.has_key(i) or c_controller.many_changed:
                    c_controller.cycloids[i] = Cycloid(self.calc, **cycloid_params[1])
                c_controller.cycloids[i].plotcoordsonbasemap(self.basemap_ax, self.orbit_pos)
            c_controller.many_changed = False

        
        else:
            if (c_controller.cyc == None or c_controller.cycloids_changed):
                c_controller.cyc = Cycloid(self.calc, self.model.get_param_value('YIELD'), self.model.get_param_value('PROPAGATION_STRENGTH'), self.model.get_param_value('PROPAGATION_SPEED'), \
                                      self.model.get_param_value('STARTING_LATITUDE'), self.model.get_param_value('STARTING_LONGITUDE'), self.model.get_param_value('STARTING_DIRECTION'), \
                                      self.model.get_param_value('VARY_VELOCITY'),self.model.get_param_value('k',0),self.model.get_param_value('ORBIT_MAX', 360, float), 0.3)
                c_controller.cycloid_changed = False
            
            c_controller.cyc.plotcoordsonbasemap(self.basemap_ax, self.orbit_pos)
            
            

    

    def save_many_cycloids(self, evt):
        # if a set of parameters from *.csv hasn't been uploaded, treat it like an error
        # with a popup window
        if not self.sc.parameters["to_plot_many_cycloids"]:
            errorMsg = """Please upload a set of cycloid parameters from *.csv file."""
            msg = wx.MessageDialog(self, errorMsg, "No input file found!", wx.OK | wx.ICON_ERROR)
            msg.ShowModal()
            msg.Destroy()

        # otherwise generate and save plots in designated folder
        else:
            chooseFolder = wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE)
            
            # so that folderName can accessed outside
            folderName = ""

            if chooseFolder.ShowModal() == wx.ID_OK:
                folderName = chooseFolder.GetPath()

            # Blanks out the entire window, which prevents people from changing tabs
            # or doing anything else, which happens naturally anyways.
            # self.Hide()

            i = 0
            while i < len(self.parameters['YIELD']):

                # create cycloid
                threshold = float(self.parameters['YIELD'][i])
                strength = float(self.parameters['PROPAGATION_STRENGTH'][i])
                speed = float(self.parameters['PROPAGATION_SPEED'][i])
                lon = float(self.parameters['STARTING_LONGITUDE'][i])
                lat = float(self.parameters['STARTING_LATITUDE'][i])
                propdir = self.parameters['STARTING_DIRECTION']
                
                print threshold, strength, speed, lon, lat, propdir
                print self.calc
                print "\n"

                plotcoordsonbasemap(self.calc, self.basemap_ax,
                                    threshold, strength, speed, lon, lat,
                                    propdir,
                                    self.sc.get_parameter(float, 'ORBIT_MAX', 360))

                # save cycloid
                plotName = str(threshold) + "_" + str(strength) + "_" +  str(speed) + "_" + str(lat) + "_" + str(lon) + "_" + str(propdir)
                self.scp.figure.savefig(folderName + '/' + plotName + ".png", bbox_inches='tight')

                # To have one cycloid saved per image, clear basemap if cycloid was plotted
                if self.ax.lines != []:
                    # self.ax.lines.pop(0)
                    self.ax.lines = []

                i += 1
            
            # when thread is done, show GUI again
            # self.Show()


    ###########################################################################
    def on_orbit_updated(self, val):
        if self.updating:
            return
        self.orbit_pos = self.scp.orbit_slider.val
        self.updating = True
        self.scp.nsr_slider.first()
        self.nsr_pos = 0
        self.updating = False
        self.plot()

    def on_nsr_updated(self, val):
        if self.updating:
            return
        self.nsr_pos = self.scp.nsr_slider.val
        self.updating = True
        self.scp.orbit_slider.first()
        self.orbit_pos = 0
        self.updating = False
        self.plot()
    
    
    def on_polar_updated(self, val):
        if self.updating:
            return
        self.polar_pos = self.scp.polar_slider.val
        self.updating = True
        self.scp.orbit_slider.first()
        self.orbit_pos = 0
        self.updating = False
        self.plot()


    def prepare_plot(self):
        b = wx.BusyInfo(u"Performing calculations. Please wait.", self)
        wx.SafeYield()
        self.prepare_plot_series()
        del b
    
    def get_grid_time(self):
        if self.orbit_pos > self.nsr_pos:
            s = self.model.get_satellite()
            return self.orbit_pos/360.0*s.orbit_period()
        else:
            return self.nsr_pos*seconds_in_year

    def mk_change_param(self, k):
        def on_change(evt):
            if k == 'direction':
                #handles change of east-west positivity
                # reverse
                temp_min = -self.sc.get_parameter(float, "LON_MAX")
                temp_max = -self.sc.get_parameter(float, "LON_MIN")
                self.sc.set_parameter("LON_MIN", temp_min)
                self.sc.set_parameter("LON_MAX", temp_max)
                self.plot()
            else:
                self.sc.set_parameter(k, self.parameters[k].GetValue())
                self.plot()
        return on_change

    def load_scale(self, k):
        try:
            if k < 0:
                self.lbound = int(config.load('PLOT_LBOUND'))
            else:
                self.ubound = int(config.load('PLOT_UBOUND'))
        except:
            if k < 0:
                self.lbound = None
            else:
                self.ubound = None
        if k < 0:
            return self.lbound
        else:
            return self.ubound

    def save_scale(self):
        config.save(PLOT_LBOUND=self.lbound, PLOT_UBOUND=self.ubound)
        
    def select_scale(self, evt):
        l = self.lbound_ctrl.GetValue()
        u = self.ubound_ctrl.GetValue()
        try:
            fl = int(l)
            fu = int(u)
            if self.lbound != l or self.ubound != u:
                self.lbound = l
                self.ubound = u
                self.save_scale()
                self.scp.plot_scale(self.scale(), "%.f kPa")
                self.select_color_range(fl*1000, fu*1000)
        except:
            self.lbound_ctrl.SetValue(self.lbound)
            self.ubound_ctrl.SetValue(self.ubound)

    def select_color_range(self, vmin, vmax):
        self.plot()
        self.cb.set_clim(vmin, vmax)
        self.cb.update_bruteforce(self.im)
        self.cb.draw_all()
        self.draw()

    def get_axes(self):
        return self.ax


    def draw(self):
        self.scp.draw()

    def on_move_in_plot(self, evt):
        if evt.inaxes:
        
            x,y = self.basemap_ax(evt.xdata, evt.ydata, inverse=True)
            i = int((x - self.model.get_param_value('LON_MIN'))/(self.model.get_param_value('LON_MAX') - self.model.get_param_value('LON_MIN') + 1e-2)*self.model.get_param_value('LON_NUM')*self.n_interp)
            j = int((y - self.model.get_param_value('LAT_MIN'))/(self.model.get_param_value('LAT_MAX') - self.model.get_param_value('LAT_MIN') + 1e-2)*self.model.get_param_value('LAT_NUM')*self.n_interp)
            x1, y1, plot_field1 = self.plot_fields[self.get_grid_time()][self.model.get_param_value('field')]
            self.val_p['LON'].SetValue("%.2f" % x)
            self.val_p['LAT'].SetValue("%.2f" % y)
            self.val_p['VAL'].SetValue("%.2f" % (plot_field1[i,j]/1000.))
       

    def colorbar(self, replot_colorbar):
        try:
            self.cb
            if replot_colorbar:
                self.adjust_to_tight()
                self.scp.figure.delaxes(self.cb.ax)
                self.cb = self.scp.colorbar(self.im, ax=self.ax, format=self.tick_formatter)
        except Exception, e:
            self.adjust_to_tight()
            self.cb = self.scp.colorbar(self.im, ax=self.ax, format=self.tick_formatter)

    def consider_obliq_lons(self, lx, rx):
        if self.model.get_param_value('Obliquity'):
            if int(round(lx)) % 90 == 0:
                lx += 1
            if int(round(rx)) % 90 == 0:
                rx -= 1
        return lx, rx

    def consider_obliq_lats(self, ly, hy):
        if self.model.get_param_value('Obliquity'):
            if int(round(ly)) % 90 == 0:
                ly += 1
            if int(round(hy)) % 90 == 0:
                hy -= 1
        return ly, hy

    def consider_lons(self):
        lx = self.model.get_param_value('LON_MIN')
        rx = self.model.get_param_value('LON_MAX')
        lx, rx = self.consider_obliq_lons(lx, rx)
        if self.model.get_param_value('projection') == 'ortho' and rx - lx >= 180:
            cx = int(round((lx + rx)/2))
            lx = cx - 90 + 1
            rx = cx + 90 - 1
        return numpy.linspace(lx, rx, self.model.get_param_value('LON_NUM')*self.n_interp)

    def consider_lats(self):
        ly = self.model.get_param_value('LAT_MIN')
        hy = self.model.get_param_value('LAT_MAX')
        ly, hy = self.consider_obliq_lats(ly, hy)
        proj = self.model.get_param_value('projection')
        if proj == 'spaeqd' and hy > 0 and ly < 0:
            hy = 0
        elif proj == 'npaeqd' and hy > 0 and ly < 0:
            ly = 0
        elif proj == 'ortho' and hy - ly >= 180:
            cy = int(round((hy + ly)/2))
            ly = cy - 90 + 1
            hy = cy + 90 - 1
        return numpy.linspace(ly, hy, self.model.get_param_value('LAT_NUM')*self.n_interp)

    def prepare_plot_series(self):
        self.plot_fields.clear()
        self.plot_vectors.clear()
        sat = self.model.get_satellite()

        lons = self.consider_lons()
        lats  = self.consider_lats()
        phis, thetas = numpy.meshgrid(lons, lats)
        x,y = self.basemap_ax(phis, thetas)
        i,j = numpy.meshgrid(
            numpy.linspace(0, self.model.get_param_value('LON_NUM') - 1, self.model.get_param_value('LON_NUM')*self.n_interp),
            numpy.linspace(0, self.model.get_param_value('LAT_NUM') - 1, self.model.get_param_value('LAT_NUM')*self.n_interp))

        self.vector_mesh_lons, self.vector_mesh_lats = self.vector_meshes()

        # monkey patching not to touch library code
        def imshow(plot_field, cmap=None, **kw):
            plot_field1 = scipy.ndimage.map_coordinates(plot_field, [i,j])
            self.plot_fields[self.plot_time][self.plot_field] = (x, y, plot_field1)

        def quiver(x, y, u, v, **kw):
            self.plot_vectors[self.plot_time][self.plot_vector].append((x, y, u, v, kw))

        _imshow = self.basemap_ax.imshow
        self.basemap_ax.imshow = imshow
        _quiver = self.basemap_ax.quiver
        self.basemap_ax.quiver = quiver

        orbit_period = sat.orbit_period()
        o = self.model.get_param_value('ORBIT_MIN', 0,float)
        om = self.model.get_param_value('ORBIT_MAX', 0,float)
        n = self.model.get_param_value('ORBIT_NUM', 0,float)
        if n > 0:
            s = (om - o)/n
            while o <= om:
                self.plot_time = o/360.0*orbit_period
                self.prepare_plot_for_time()
                o += s
        nm = self.model.get_param_value('TIME_MIN', 0,float)
        s = self.model.get_param_value('nsr_time', 0,float)
        n = self.model.get_param_value('TIME_NUM', 0,float)
        for k in range(0, n+1):
            self.plot_time = (s*k + nm)*seconds_in_year
            self.prepare_plot_for_time()
        self.basemap_ax.imshow = _imshow
        self.basemap_ax.quiver = _quiver

    def prepare_plot_for_time(self):
        # we use self.plot_time instead of passing it as parameter 
        # because it is used in redefined imshow and quiver in function above
        self.plot_fields[self.plot_time] = {}
        lon_min, lon_max = self.consider_obliq_lons(self.model.get_param_value('LON_MIN'),
                self.model.get_param_value('LON_MAX'))
        lat_min, lat_max = self.consider_obliq_lats(self.model.get_param_value('LAT_MIN'),
                self.model.get_param_value('LAT_MAX'))
        for self.plot_field in ['tens', 'comp', 'mean', 'diff']:
            scalar_grid(
                stresscalc = self.calc,
                nlons = self.model.get_param_value('LON_NUM'),
                nlats = self.model.get_param_value('LAT_NUM'),
                min_lon = numpy.radians(lon_min),
                max_lon = numpy.radians(lon_max),
                min_lat = numpy.radians(lat_min),
                max_lat = numpy.radians(lat_max),
                time_t = self.plot_time,
                field = self.plot_field,
                basemap_ax = self.basemap_ax)
        # self.plot_vector for same reasons as self.plot_time
        self.plot_vector = 'principal'
        self.plot_vectors[self.plot_time] = { self.plot_vector: [] }
        vector_points1(stresscalc=self.calc,
            lons = self.vector_mesh_lons,
            lats = self.vector_mesh_lats,
            time_t = self.plot_time,
            plot_greater = True,
            plot_lesser = True,
            plot_comp = True,
            plot_tens = True,
            scale = self.scale()*vector_mult,
            basemap_ax = self.basemap_ax)
        for self.plot_vector in ['latitude', 'longitude', 'shear']:
            self.plot_vectors[self.plot_time][self.plot_vector] = []
            vector_points2(stresscalc=self.calc,
                lons = self.vector_mesh_lons,
                lats = self.vector_mesh_lats,
                time_t = self.plot_time,
                plot_norm_lat = (self.plot_vector == 'latitude'),
                plot_norm_lon = (self.plot_vector == 'longitude'),
                plot_shear =( self.plot_vector == 'shear'),
                scale = self.scale()*vector_mult,
                basemap_ax = self.basemap_ax)

    def scale(self):
        def max_abs(*v):
            ''' finds the maximum of the absolute values of [vectors?] '''
            # how diff from max(map(abs, v))?
            return max(*map(abs, v))
        return max_abs(self.ubound, self.lbound)

    def plot_gradient(self):
        
        print self.get_grid_time()
        x, y, plot_field1 = self.plot_fields[self.get_grid_time()][self.model.get_param_value('field')]
        l = int(self.lbound) * 1000
        u = int(self.ubound) * 1000
        self.im = self.basemap_ax.pcolormesh(x, y, numpy.transpose(plot_field1), cmap='gist_rainbow_r', vmin=l, vmax=u)

    def plot_grid_calc(self):
        replot_colorbar = False
        print self.model.get_param_value('field')
        if self.changed:
            print 'h'
            self.orbit_pos = self.model.get_param_value('ORBIT_MIN', 0, int)
            self.nsr_pos = self.model.get_param_value('TIME_MIN', 0,float)
            self.hide_sliders()
            self.show_needed_sliders()
            self.prepare_plot()
            self.changed = False
            replot_colorbar = True
        elif self.projection_changed:
            self.prepare_plot()
        if self.model.get_param_value('field'):
            self.plot_gradient()
        if self.model.get_param_value('to_plot_principal_vectors'):
            self.plot_principal_vectors()
        if self.model.get_param_value('to_plot_latitude_vectors') \
        or self.model.get_param_value('to_plot_longitude_vectors') \
        or self.model.get_param_value('to_plot_shear_vectors'):
            self.plot_stress_vectors()
        if self.model.get_param_value('to_plot_lineaments'):
            self.plot_lineaments()
               
        if self.model.get_param_value('to_plot_cycloids'):
            self.plot_cycloids()
        
        self.colorbar(replot_colorbar)

    def adjust_to_tight(self):
        [lat0, lat1, lon0, lon1] = map(float, [ self.model.get_param_value(x) for x in ['LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX']])
        l = (lon1 - lon0)/(lat1 - lat0)*scale_bar_length
        s = (l - scale_bar_length)/2
        #self.scp.figure.subplots_adjust(left=scale_left - s, right=scale_left + scale_bar_length + s + 0.3*l)
        self.scp.figure.subplots_adjust(left = scale_left - s,# - 0.03,
            right = scale_left + scale_bar_length + 1.5*s + 0.1)

    def vector_meshes(self):
        lon_min, lon_max = self.consider_obliq_lons(self.model.get_param_value('LON_MIN'),
                self.model.get_param_value('LON_MAX'))
        lat_min, lat_max = self.consider_obliq_lats(self.model.get_param_value('LAT_MIN'),
                self.model.get_param_value('LAT_MAX'))
        vector_grid_lons  = numpy.linspace(
            numpy.radians(lon_min),
            numpy.radians(lon_max),
            self.model.get_param_value('LON_NUM'))
        vector_grid_lats  = numpy.linspace(
            numpy.radians(lat_min),
            numpy.radians(lat_max),
            self.model.get_param_value('LAT_NUM'))
        vector_mesh_lons, vector_mesh_lats = numpy.meshgrid(vector_grid_lons, vector_grid_lats)

        vector_mesh_lons = numpy.ravel(vector_mesh_lons)
        vector_mesh_lats = numpy.ravel(vector_mesh_lats)
        return vector_mesh_lons, vector_mesh_lats

    def plot_stress_vectors(self):
        if self.model.get_param_value('to_plot_latitude_vectors'):
            for x, y, u, v, kw in self.plot_vectors[self.get_grid_time()]['latitude']:
                self.basemap_ax.quiver(x, y, u, v, **kw)
        if self.model.get_param_value('to_plot_longitude_vectors'):
            for x, y, u, v, kw in self.plot_vectors[self.get_grid_time()]['longitude']:
                self.basemap_ax.quiver(x, y, u, v, **kw)
        if self.model.get_param_value('to_plot_shear_vectors'):
            for x, y, u, v, kw in self.plot_vectors[self.get_grid_time()]['shear']:
                self.basemap_ax.quiver(x, y, u, v, **kw)

    def plot_principal_vectors(self):
        for x, y, u, v, kw in self.plot_vectors[self.get_grid_time()]['principal']:
            kw['scale'] = float(self.scale()*vector_mult)
            self.basemap_ax.quiver(x, y, u, v, **kw)
    
    def mpl_color(self, color):
        return map(lambda c: float(c)/255, color[0:3])

    def show_needed_sliders(self):
        if (not self.model.get_param_value('Nonsynchronous Rotation') == 'None' or not self.model.get_param_value('Nonsynchronous Rotation'))\
        and not self.model.get_param_value('TIME_MIN') == 'None' and not self.model.get_param_value('nsr_time') == 'None' \
        	and not self.model.get_param_value('TIME_NUM') == 'None':
            self.reveal_nsr_slider()
        else:
			self.hide_nsr_slider()


        if (self.model.get_param_value('Diurnal', False) or self.model.get_param_value('Obliquity', False)) or self.model.get_param_value('Polar Wander', False) \
			and not self.model.get_param_value('ORBIT_MIN') == None and not self.model.get_param_value('ORBIT_MAX') == None \
			and not self.model.get_param_value('ORBIT_NUM') == None:
			print 'success'
			self.reveal_orbit_slider()

        else:
            self.hide_orbit_slider()

        
    
        self.hide_polar_slider() #For elastic model only need orbit
       	

    def hide_orbit_slider(self):
        if not self.orbit_hidden:
            self.orbit_hidden = True
            self.scp.del_orbit()
    
    def hide_nsr_slider(self):
        if not self.nsr_hidden:
            self.nsr_hidden = True
            self.scp.del_nsr()



    def init_orbit_slider(self):
		self.scp.change_orbit_slider(
        float(self.model.get_parameter('ORBIT_MIN').get_value(0)),
        float(self.model.get_parameter('ORBIT_MAX').get_value(1)),  
        float(self.model.get_parameter('ORBIT_NUM').get_value(10)),
      	self.orbit_pos)

    def init_nsr_slider(self):
        nm = self.model.get_param_value('TIME_MIN', 0,float)
        self.scp.change_nsr_slider(nm, nm + self.model.get_param_value('nsr_time', 0,float)*self.model.get_param_value('TIME_NUM', 0,float),
        self.model.get_param_value('TIME_NUM', 1,int), self.nsr_pos)
        
	def init_polar_slider(self):
		nm = self.model.get_param_value('TIME_MIN', 0,float)
		self.scp.change_polar_slider(
		nm,
		nm + self.model.get_param_value('nsr_time', 0,float)*self.model.get_param_value('TIME_NUM', 0,float),
		self.model.get_param_value('TIME_NUM', 1,int),
		self.nsr_pos)

    
    def hide_polar_slider(self):
        if not self.polar_hidden:
            self.polar_hidden = True
            self.scp.del_polar()
    

    def hide_sliders(self):
        self.hide_nsr_slider()
        self.hide_orbit_slider()
        #self.hide_polar_slider()
      	
    def reveal_orbit_slider(self):
        if self.orbit_hidden:
            self.orbit_hidden = False
            self.scp.add_orbit()
            self.init_orbit_slider()
            self.scp.orbit_slider.on_changed(self.on_orbit_updated)
            self.scp.save_orbit_series = self.save_orbit_series

    def reveal_nsr_slider(self):
        if self.nsr_hidden:
            self.nsr_hidden = False
            self.scp.add_nsr()
            self.scp.nsr_slider.on_changed(self.on_nsr_updated)
            self.init_nsr_slider()
            self.scp.save_nsr_series = self.save_nsr_series

   
    def reveal_polar_slider(self):
        if self.polar_hidden:
            self.polar_hidden = False
            self.scp.add_polar()
            self.scp.polar_slider.on_changed(self.on_polar_updated)
            self.init_polar_slider()
            self.scp.save_polar_series = self.save_polar_series
    


    def hide_orbit_controls(self):
        self.scp.del_orbit_controls()
        self.scp.orbit_slider.on_changed(lambda v: v)
    
    def hide_nsr_controls(self):
        self.scp.del_nsr_controls()
        self.scp.nsr_slider.on_changed(lambda v: v)

    def reveal_orbit_controls(self):
        self.scp.add_orbit_controls()
        self.scp.save_orbit_series = self.save_orbit_series
        self.scp.orbit_slider.on_changed(self.on_orbit_updated)

    def reveal_nsr_controls(self):
        self.scp.add_nsr_controls()
        self.scp.save_nsr_series = self.save_nsr_series
        self.scp.nsr_slider.on_changed(self.on_nsr_updated)

    def save_orbit_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_orbit_pos = self.orbit_pos
        sat = self.sc.get_satellite()
        orbit_period = sat.orbit_period()
        o = self.sc.get_parameter(float, 'ORBIT_MIN', 0)
        om = self.sc.get_parameter(float, 'ORBIT_MAX', 0)
        n = self.sc.get_parameter(float, 'ORBIT_NUM', 0)
        s = (om - o)/n
        self.hide_orbit_controls()

        localtime = time.asctime(time.localtime(time.time()))
        location = dir + "/" + self.sc.parameters['SYSTEM_ID']
        directory = location + "/" + localtime
        if os.path.isdir(location):
            os.mkdir(directory)
        else:
            os.mkdir(location)
            os.mkdir(directory)

        while o <= om:
            self.orbit_pos = o
            self.plot_no_draw()
            self.scp.orbit_slider.set_val(self.orbit_pos)
            self.scp.figure.savefig("%s/orbit_%03d.%02d.png" %
                (directory, int(self.orbit_pos), round(100.*(self.orbit_pos - int(self.orbit_pos)))),
                bbox_inches='tight', pad_inches=1.5)
            o += s
        self.orbit_pos = old_orbit_pos
        self.reveal_orbit_controls()
        self.init_orbit_slider()
        self.scp.orbit_slider.set_val(self.orbit_pos)
        self.plot()
        del b
    
    def save_nsr_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_nsr_pos = self.nsr_pos
        nm = self.sc.get_parameter(float, 'TIME_MIN', 0)
        s = self.sc.get_parameter(float, 'nsr_time', 0)
        n = self.sc.get_parameter(int, 'TIME_NUM', 0)
        self.hide_nsr_controls()

        localtime = time.asctime(time.localtime(time.time()))
        location = dir + "/" + self.sc.parameters['SYSTEM_ID']
        directory = location + "/" + localtime
        if os.path.isdir(location):
            os.mkdir(directory)
        else:
            os.mkdir(location)
            os.mkdir(directory)
            
        for k in range(0, n+1):
            self.nsr_pos = nm + s*k
            self.scp.nsr_slider.set_val(self.nsr_pos)
            self.plot_no_draw()
            self.scp.figure.savefig("%s/nsr_%03d.png" % (directory, k), bbox_inches='tight', pad_inches=0.5)
        self.nsr_pos = old_nsr_pos
        self.reveal_nsr_controls()
        self.init_nsr_slider()
        self.scp.nsr_slider.set_val(self.nsr_pos)
        self.plot()
        del b


    
    def save_polar_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_polar_pos = self.polar_pos
        nm = self.sc.get_parameter(float, 'TIME_MIN', 0)
        s = self.sc.get_parameter(float, 'polar_time', 0)
        n = self.sc.get_parameter(int, 'TIME_NUM', 0)
        self.hide_polar_controls()
        for k in range(0, n+1):
            self.polar_pos = nm + s*k
            self.scp.polar_slider.set_val(self.polar_pos)
            self.plot_no_draw()
            self.scp.figure.savefig("%s/polar_%03d.png" % (dir, k), bbox_inches='tight', pad_inches=0.5)
        self.polar_pos = old_polar_pos
        self.reveal_polar_controls()
        self.init_polar_slider()
        self.scp.polar_slider.set_val(self.polar_pos)
        self.plot()
        del b
    




class View(wx.Frame):

    def __init__(self, parent,model, controller):
        wx.Frame.__init__(self, parent)

        self.view_parameters = {}

        self.p = SatStressPanel(self,model=model,controller = controller, view_parameters = self.view_parameters)

        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.GetSizer().Add(self.p, 1, wx.ALL|wx.EXPAND, 10)
        menubar = wx.MenuBar()

        ##### 'File' option of menubar #####
        File = wx.Menu()
        self.export = File.Append(wx.ID_SAVE, '&Export\tCtrl+S', 'Save all variables')
        self.load = File.Append(wx.ID_OPEN, '&Load\tCtrl+O', 'Load a set of variables')
        self.quit = File.Append(wx.ID_ANY, '&Quit\tCtrl+Q', 'Quit Application')

        menubar.Append(File,"File")

        ##### 'Information' option of menubar #####        
        Information = wx.Menu()

        About = wx.Menu()
        rights = About.Append(wx.ID_ANY, '&Copyright')
        self.Bind(wx.EVT_MENU, self.onRights, rights)
        updates = About.Append(wx.ID_ANY, '&Version')
        self.Bind(wx.EVT_MENU, self.onUpdates, updates)
        contact = About.Append(wx.ID_ANY, '&Contact')
        self.Bind(wx.EVT_MENU, self.onContacts, contact)
        develop = About.Append(wx.ID_ANY, '&Development')
        self.Bind(wx.EVT_MENU, self.onDevelopment, develop)

        Information.AppendMenu(wx.ID_ANY, "&About", About)
        Information.AppendSeparator()

        References = wx.Menu()
        #ref = References.Append(wx.ID_ANY, '&General')
        #self.Bind(wx.EVT_MENU, self.onRef, ref)
        Diurnalref = References.Append(wx.ID_ANY, '&Diurnal')
        self.Bind(wx.EVT_MENU, self.onDiurnalref, Diurnalref)
        NSRref = References.Append(wx.ID_ANY, '&Nonsynchronous Rotation')
        self.Bind(wx.EVT_MENU, self.onNSRref, NSRref)
        Obliquityref = References.Append(wx.ID_ANY, '&Obliquity')
        self.Bind(wx.EVT_MENU, self.onObliquityref, Obliquityref)
        ISTref = References.Append(wx.ID_ANY, '&Ice Shell Thickening')
        self.Bind(wx.EVT_MENU, self.onISTref, ISTref)
        PWref = References.Append(wx.ID_ANY, '&Polar Wander')
        self.Bind(wx.EVT_MENU, self.onPWref, PWref)
        Cycloidsref = References.Append(wx.ID_ANY, '&Cycloids')
        self.Bind(wx.EVT_MENU, self.onCycloidsref, Cycloidsref)

        Information.AppendMenu(wx.ID_ANY, "&References", References)

        menubar.Append(Information, "&Information")

        ##### 'Help' option of menubar ######
        Help = wx.Menu()
        Tutorial = Help.Append(wx.ID_ANY, '&Getting Started\tf1')
        self.Bind(wx.EVT_MENU, self.onTutorial, Tutorial)
        HelpSat = Help.Append(wx.ID_ANY, '&Satellite Tab')
        self.Bind(wx.EVT_MENU, self.onHelpSat, HelpSat)
        HelpStress = Help.Append(wx.ID_ANY, '&Stresses Tab')
        self.Bind(wx.EVT_MENU, self.onHelpStresses, HelpStress)
        HelpPoint = Help.Append(wx.ID_ANY, '&Point Tab')
        self.Bind(wx.EVT_MENU, self.onHelpPoint, HelpPoint)
        HelpGrid = Help.Append(wx.ID_ANY, '&Grid Tab')
        self.Bind(wx.EVT_MENU, self.onHelpGrid, HelpGrid)
        HelpCycloids = Help.Append(wx.ID_ANY, '&Cycloids Tab')
        self.Bind(wx.EVT_MENU, self.onHelpCycloids, HelpCycloids)
        HelpPlot = Help.Append(wx.ID_ANY, '&Plot Tab')
        self.Bind(wx.EVT_MENU, self.onHelpPlot, HelpPlot)
        menubar.Append(Help, "&Help")

        self.SetMenuBar(menubar)

        exit_id = wx.NewId()
        wx.EVT_MENU(self, exit_id, self.exit)
        accel = wx.AcceleratorTable([
            (wx.ACCEL_CTRL, ord('W'), exit_id)])
        self.SetAcceleratorTable(accel)
        
        # Bind our events from the close dialog 'x' on the frame
        self.Bind(wx.EVT_CLOSE, self.OnCloseFrame)

        # SetSizeHints(minW, minH, maxW, maxH)
        # This function effectively enforces a lower bound to SatStressGUI window resizing.
        # To allow for unrestricted window resizing, simply remove this line.
        self.SetSizeHints(1045,690,2000, 2000)

        self.Fit()
        self.Show(True)
        self.CenterOnScreen()
        self.p.SetFocus()

    def get_ctrl_obj(self,param_name):
        return self.view_parameters[param_name]

    def onRights(self, evt):
        # indentation (lack thereof) necessary to prevent tab spaces every newline in source code
        # not sure if the need for such indentation or lack thereof is b/c of python or wx
        # alternative is to use concatentation
        spiel = u"""ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any \
commercial use must be negotiated with the Office of Technology Transfer at the \
California Institute of Technology. \n\n
This software may be subject to U.S. export control laws and regulations. \
By accepting this document, the user agrees to comply with all applicable \
U.S. export laws and regulations. User has the responsibility to obtain export \
licenses, or other export authority as may be required before exporting such \
information to foreign countries or providing access to foreign persons. """

        copyright = "Copyright 2016, by the California Institute of Technology."
        #Update year whenever a new version is released.

        self.makeMsgDialog(spiel, copyright)

    def onDevelopment(self, evt):
        spiel = u"""SatStressGUI V4.0 was developed at the Jet Propulsion Laboratory, \
California Institute of Technology and is based on SatStressGUI. \
SatStressGUI was developed by the Planetary Geology Research group at the University of Idaho \
SatStress GUI is based on SatStress, which was designed by Zane Selvans and is available at \
http://code.google.com/p/satstress and most recently at https://github.com/zaneselvans/satstress \
\n\n SatStressGUI 4.0 has been created upon efforts by \
Alex Patthoff, Robert Pappalardo, Jonathan Kay, Lee Tang, \
Simon Kattenhorn, C.M. Cooper, Emily S. Martin, \
David Dubois, Ben J. Ayton, Jessica B. Li, \
Andre Ismailyan, Peter Sinclair."""
        
        self.makeMsgDialog(spiel, u'Developers')

    def onUpdates(self, evt):
        updates = u"""This is Version 4.0 of SatStressGUI.  For more information, please visit: \n\n\
https://github.com/SatStressGUI/SatStressGUI\n\n\
In this version, several bugs were fixed, and a new stressing mechanism (Polar Wander) was added.\
To find detailed notes of all the changes, please visit the GitHub page."""
        
        self.makeMsgDialog(updates, u'Version 4.0')

    def onRef(self, evt):
        references = u""" For more information, please see:\n\n \
1) Wahr, J., Z. A. Selvans, M. E. Mullen, A. C. Barr, G. C. Collins, \
M. M. Selvans, and R. T. Pappalardo, Modeling stresses on satellites due to non-synchronous rotation \
and orbital eccentricity using gravitational potential theory, \
Icarus, Volume 200, Issue 1, March 2009, Pages 188-206.\n\n \
2) See chapter on Geodynamics of Europa's Ice Shell by Francis Nimmo and Michael Manga in \
Europa for more information about the ice shell thickening model.\n\n \
3) See Hoppa, G.V., Tufts, B.R., Greenberg, R., Geissler, P.E., 1999b. Formation of cycloidal \
features on Europa. Science 285, 1899-1902, or chapter on Geologic Stratigraphy and Evolution of \
Europa's surface by Thomas Doggett, Ronald Greeley, Patricio Figueredo and Ken Tanaka in Europa \
for additional information on cycloid formation, for diurnal potential including obliquity \n\n \
4) Jara-Orue, H. M., & Vermeersen, B. L. (2011). Effects of low-viscous layers and a non-zero \
obliquity on surface stresses induced by diurnal tides and non-synchronous rotation: The \
case of Europa. Icarus, 215(1), 417-438, for stress cuased by ice shell thickening."""
        self.makeMsgDialog(references, u'Science References')

    def onContacts(self, evt):
        # Create a message dialog box
        self.makeMsgDialog(u"Alex Patthoff via Patthoff@jpl.nasa.gov",
                           u"Primary Contact")

    def onDiurnalref(self, evt):
        Resources = u"""Diurnal tidal stresses arise when a satellite is in an eccentric orbit. \
This is due to two reasons. \
First, the amplitude of the planet's gravitational force is greater at periapse than it is at apoapse. \
Secondly, the planet is rotating slightly faster (compared to its synchronous rotation rate) at periapse \
and slightly slower (again compared to its synchronous rotation rate) at apoapse. \
This results in a 'librational tide', where the planet appears to rock back and forth in the sky.\n\n\
For more information on diurnal tides, please see:\n\
Wahr, J., Z. A. Selvans, M. E. Mullen, A. C. Barr, G. C. Collins, \
M. M. Selvans, and R. T. Pappalardo, Modeling stresses on satellites due to non-synchronous rotation \
and orbital eccentricity using gravitational potential theory, \
Icarus, Volume 200, Issue 1, March 2009, Pages 188-206.
"""
        self.makeMsgDialog(Resources, u'About Diurnal Tides')

    def onNSRref(self, evt):
        Resources = u"""Nonsynchronous rotation (NSR) occurs when a satellite's lithosphere is decoupled from its core. \
When this happens, the tidal bulge of the shell causes it to experience a net torque, and could rotate more quickly than the synchronous rate. \
Thus, the planet appears to move across the sky, and the tidal bulge moves beneath the shell. \
This results in surface stresses. \
The period of this rotation should be > 10,000 years.\n\n\
For more information on NSR, please see:\n\
Wahr, J., Z. A. Selvans, M. E. Mullen, A. C. Barr, G. C. Collins, \
M. M. Selvans, and R. T. Pappalardo, Modeling stresses on satellites due to non-synchronous rotation \
and orbital eccentricity using gravitational potential theory, \
Icarus, Volume 200, Issue 1, March 2009, Pages 188-206.
"""
        self.makeMsgDialog(Resources, u'About Nonsynchronous Rotation')

    def onObliquityref(self, evt):
        Resources = u"""A satellite's obliquity (or axial tilt) is the angle between it rotational axis and its orbital axis. \
A satellite of zero obliquity will have a rotational axis perpendicular to its orbital plane. \
However, when the obliquity is nonzero, it causes the stresses due to diurnal tides and non-synchronous rotation to be asymmetric.\n\n\
For more information on stresses due to oblique orbits, see:\n\
Jara-Orue, H. M., & Vermeersen, B. L. (2011). Effects of low-viscous layers and a non-zero \
obliquity on surface stresses induced by diurnal tides and non-synchronous rotation: The \
case of Europa. Icarus, 215(1), 417-438, for stress cuased by ice shell thickening.
"""
        self.makeMsgDialog(Resources, u'About Olibque Orbits')

    def onISTref(self, evt):
        Resources = u"""As satellites age, they could become cooler. \
This would result in more of the liquid ocean freezing, increasing the thickness of the icy crust. \
This process would force the ice shell to expand, putting extensional stress on the surface.\n\n\
For more information on Ice Shell Thickening as a stressing mechanism, please see:\n\
Nimmo, F. (2004). Stresses generated in cooling viscoelastic ice shells: Application \
to Europa. Journal of Geophysical Research: Planets (1991-2012), 109(E12).
"""
        self.makeMsgDialog(Resources, u'About Ice Shell Thickening')


    def onPWref(self, evt):
        Resources = u"""
Polar Wander is the apparent movement of a satellite's rotational pole due to nonsynchronous reorientation of the satellite's crust. \
If a satellite's crust is not coupled to its core, it may experience nonsynchronous rotation (NSR). \
Sometimes, this also results in a reorientation of the poles. \
The north pole appears to wander over the surface as the crust reorients itself. \
This results in stressing, due to the tidal bulge of the core and ocean moving beneath the crust, \
as well as the parent planet appearing to hange its location in the sky. \n\n\
This stressing mechanism is calculated using an elastic model.\n\n\
For more information on Polar Wander as a stressing mechanism, please see:\n\
    Matsuyama, Isamu, and Francis Nimmo. "Tectonic patterns on reoriented and despun planetary bodies." Icarus 195, no. 1 (2008): 459-473.\n\
    Matsuyama, Isamu, Francis Nimmo, and Jerry X. Mitrovica. "Planetary reorientation." Annual Review of Earth and Planetary Sciences 42 (2014): 605-634.
"""
        self.makeMsgDialog(Resources, u'About Polar Wander')

    def onCycloidsref(self, evt):
        Resources = u""" Cycloids are arcuate lineaments found on the surface of Europa.  \
They are thought to be created when a fracture in the ice is propagated because of the stresses. \
In order for a cycloid to be created, the tensile stress at the location must exceed the tensile strength of the ice.\
Once the fracture has started, it will propagate through the ice at a certain velocity.\
This velocity could be constant, or could vary depending on the magnitude of the stress.\
During the cycloid's propagation, the satellite will continue orbiting around its primary.\
This causes the stress field on the satellite to change, making the cycloids curve.\
When the stress is no longer greater than the requisite propagation strength, the cycloid stops moving.\
If the stress reaches the propagation strength again, it will continue.\n\n\
For more information, please see:\n\
    Hoppa, G.V., Tufts, B.R., Greenberg, R., Geissler, P.E., 1999b. Formation of cycloidal \
features on Europa. Science 285, 1899-1902"""
        self.makeMsgDialog(Resources, u'About Cycloids')

    def onTutorial(self, evt):
        Tutorial = u"""Welcome to SatStressGUI!  This program is designed to model stresses icy satellites \
experience as they orbit their primary.  For more information on this program and the mathematics behind it, \
check the "Information" menu. \n\n\
1) Input the satellite's physical parameters on the Satellite tab.\n\
2) Select which stresses to apply in the Stresses tab.\n\
- When using Diurnal and NSR, either input Love numbers and check the box marked "Input Love Numbers", or \
leave them blank to allow the program to calculate Love numbers based on the satellite's physical properties.\n\
- Please note that most stresses do not function well together.\n\
- Obliquity must be used with either Diurnal or NSR.\n\
3) In the Grid tab, input a latitude and longitude range to examine.\n\
- The number of grid points must be equal for both latitude and longitude.\n\
4) Also in the Grid tab, input the relevant information for the selected stresses.\n\
5) Change to the Plot tab to see the stress maps.\n\
- For more information on how to use the maps, see "Plot" in the Help Menu.\n\
6) Use the Point tab to calculate the stress at up to 10 discrete points in space and time.
"""
        self.makeMsgDialog(Tutorial, u'Getting Started')

    def onHelpSat(self, evt):
        Help = u"""The Satellite Tab is used to input the physical properties of the satellite.\n\n\
- Each entry should use the units denoted in the square brackets next to the box.\n\
- The viscoelastic model used assumes that the satellite has two icy layers, a liquid ocean, and a solid core.\n\
- The NSR period is usually on the order of 100,000 years.  If you are not using NSR, you can leave it as 'infinity'.\n\
- The orbital eccentricity must be < 0.25.  Otherwise the program cannot reasonably calculate stresses.\n\
- If you have changed a number, but nothing seems to happen, try hitting 'Enter' in the box you changed.\n\
"""
        self.makeMsgDialog(Help, u'The Satellite Tab')

    def onHelpStresses(self, evt):
        Help = u"""The Stresses Tab is used to select which stresses to use.\n\n\
- For Diurnal and NSR stresses, the h2, k2, and l2 boxes should be left blank, unless the user wants to input their own values. \
Checking the "Input Love Numbers" box will allow you to use custom Love numbers. \
When inputting custom love numbers, you must use the format <Re> +/ <Im>j.  Do not use scientific notation. \
1.2 + 3e-05j would look like 1.2+0.00003j.\n\
- Most stresses should be used independently, however the Obliquity stress must be used with Diurnal or NSR.\n\
- The Thermal Diffusivity of the Ice Shell Thickening stress does not currently function.\n\
"""
        self.makeMsgDialog(Help, u'The Stresses Tab')

    def onHelpPoint(self, evt):
        Help = u"""The Point Tab can be used to calculate the stress at up to 10 discrete points in space and time.\n\n\
- Enter a latitude, longitude, year, and orbital position for up to 10 points.\n\
- Press the "Calculate Stress" button.\n\
- Use the "Save to File" button to save the results as a .cvs file.\n\n\
- θ: Latitude (-90.00 to 90.00) [°]\n\
- φ: Longitude (-180.00 to180.00 (positive West or East to choose from)) [°]\n\
- t: Time since periapse (Periapse = 0) [yrs], used for secular stress calculations\n\
- orbital pos: Orbital position since periapse (Periapse = 0) [°], used for diurnal stress calculations\n\
- Stt: East-West component of stress field [kPa]\n\
- Spt: Off diagonal component of stress field [kPa]\n\
- Spp: North-South component of stress field [kPa]\n\
- σ1: Maximum tension [kPa]\n\
- σ3: Maximum compression [kPa]\n\
- α: The angle between σ1 and due north (clockwise is positive) [°]
"""
        self.makeMsgDialog(Help, u'The Point Tab')


    def onHelpGrid(self, evt):
        Help = u"""The Grid Tab is used to specify what section of the satellite to look at.\n\n\
- For more information about each stress, see the Information menu.
- NOTE: The number of latitude and longitude grid points must be equal.\n\
- To examine the whole moon, use a latitude range from -90 to 90 and a longitude range of -180 to 180.\n\
- Each row will only activate when the appropriate stress is enabled.\n\
- The "Orbital Position" row is used to track diurnal stress from the satellite's orbit.  The satellite starts at the minimum position, and moves to the maximum position. \
Inputting 0 to 360 degrees will be one full orbit.  Additional orbits can be added by increasing the maximum beyond 360 degrees.\n\
- The "Amount of NSR Buildup" row is used to determine how long the ice shell has been rotating. \
The Start Time is when the plotting starts, and the End Time is when the plotting ends.\n\
- The "Final Pole Location" is used for the Polar Wander stress.
"""
        self.makeMsgDialog(Help, u'The Grid Tab')

    def onHelpCycloids(self, evt):
        Help = u"""The Cycloids Tab allows the user to generate a cycloidal feature on the map.\n\n\
- The cycloids are modeled and plotted on the Plot Tab.\n\
- The Yield Threshold is how much stress must be put on the crust to break the ice and initiate a fracture.\n\
- The Propagation Strength is how much stress must be put on the crust to make the split continue, and the split continues at the Propagation Speed.\n\
- The Starting Latitude and Longitude determine where the cycloid begins, and the Direction determines the curvature of the cycloid.\n\
- NOTE: The Vary Velocity option is currently untested.\n\
- For more information on cycloids, see the Information menu.
"""
        self.makeMsgDialog(Help, u'The Cycloids Tab')

    def onHelpPlot(self, evt):
        Help = u"""The Plot Tab shows a map of the stresses on the surface of the satellite.\n\n\
- Tension on the map is shown as positive, and compression as negative.
- You can step through the plots by using the buttons to the bottom right of the graph.\n\
- Each individual plot can be saved by using the save button to the lower left of the graph, and the series can be saved using the "Save Series" \
button to the lower right.\n\
- The panel on the right allows manipulation of the map, changing the scale and type of map, as well as the stresses showed.\n\
- The bottom panel enables and disables cycloids.\n\
- NOTE: The cycloids cannot be saved as shape or netcdf files currently.\n\
- NOTE: The Lineaments features does not function currently.
"""
        self.makeMsgDialog(Help, u'The Plot Tab')
       
    def makeMsgDialog(self, msg, title):
        msg = wx.MessageDialog(self, msg, title, wx.OK | wx.ICON_INFORMATION)
        msg.ShowModal()
        msg.Destroy
    
    # Makes sure the user was intending to quit the application
    # at some point, make this conditional to if not changes have been made, no popup
    def OnCloseFrame(self, event):
            self.exit(event)

    def exit(self, evt):
        sys.exit(0)

class BaseController:
    def __init__(self,model, view):
        self.model = model
        self.view = view

    def bind_all(self):
        for param_name, ctrlObj in self.view.view_parameters.items():
            if isinstance(ctrlObj, wx.TextCtrl):
                if (self.model.get_parameter(param_name).get_param_type() in ('int', 'float')):  
                    ctrlObj.Bind(wx.EVT_CHAR, self.OnChar)
                ctrlObj.Bind(wx.EVT_TEXT, self.OnText)
            '''
            elif isinstance(ctrlObj, list):
                print 'hey'
                for i, ctrl in enumerate(ctrlObj):
                    if (self.model.get_parameter(param_name).get_param_type() in ('int', 'float')):  
                        ctrl.Bind(wx.EVT_CHAR, self.OnChar)
                    ctrl.Bind(wx.EVT_TEXT, lambda evt, index = i: self.OnText_for_lists(evt, index))

            '''
    def OnChar(self,event):
        charEntered= event.GetKeyCode()

        if (charEntered >= 48 and charEntered <= 57) or charEntered == 8 or charEntered == 9 or charEntered == 13: #doesn't allow for floats currently since period cant be entered
            event.Skip()


    def OnText(self,event):
        if not event.GetEventObject().GetValue() == 'None':
            self.set_parameter(event.GetEventObject().GetName(),event.GetEventObject().GetValue())
            

    def OnText_for_lists(self, event, i):
        if not event.GetEventObject().GetValue() == 'None':
            self.model.set_parameter(event.GetEventObject().GetName(), event.GetEventObject().GetValue(), point=i)

    def set_parameter(self,param_name, value , point=-1):
        '''
        @param_name - parameter to change
        @value - value to set the parameter to
        @point - if param is a ParameterList, the index of the value to change.
        '''
        ctrl = None

        if point >=0:   #for points tab only
            ctrl = self.view.view_parameters[param_name][point]
            ctrl.SetValue(value)

        else:
            param = self.model.get_parameter(param_name)
            param.set_value(str(value))
            if param.get_category() == 'cycloids':
                self.cp_controller.cycloids_changed = True
            elif param.get_category() in ['stresses_var', 'satellite_var', 'grid_var']:
                print 'truit'
                self.view.p.spp.changed = True
            try:
                #update model
                ctrl = self.view.view_parameters[param_name]
                
                #update view
                if ( isinstance(ctrl, wx.TextCtrl) or isinstance(ctrl, wx.ComboBox) ):
                    ctrl.SetValue(str(value))
                elif(isinstance(ctrl, wx.CheckBox)):
                    ctrl.SetValue(bool(value))
            
            except Exception, e:
                print e


class CycloidsPanelController(BaseController):
    def __init__(self, model, view, cycloids_panel):
        BaseController.__init__(self,model,view)
        self.panel = cycloids_panel

        self.cyc = None
        self.cycloids_changed = True


        self.panel.constant.Disable()
        
        self.panel.use_multiple.Disable()
        self.panel.use_multiple.SetValue(0)
        self.model.set_parameter('to_plot_many_cycloids',  False)

        self.model.set_parameter('k', 0)


        self.panel.start_dir.Bind(wx.EVT_COMBOBOX, self.EvtSetDir)
        self.panel.save_bt.Bind(wx.EVT_BUTTON, self.on_save_cyclparams)
        self.panel.load_bt.Bind(wx.EVT_BUTTON, self.on_load_cyclparams)
        self.panel.vary.Bind(wx.EVT_CHECKBOX, self.EvtSetVary)
        self.panel.many_params.Bind(wx.EVT_BUTTON, self.on_load_many)
        self.panel.constant.Bind(wx.EVT_TEXT, self.EvtSetConstant)
        self.panel.use_multiple.Bind(wx.EVT_CHECKBOX, self.EvtSetUseMultiple)

        self.panel.view_parameters['to_plot_many_cycloids'] = self.panel.use_multiple
        self.panel.view_parameters['VARY_VELOCITY'] = self.panel.vary
        self.panel.view_parameters['STARTING_DIRECTION'] = self.panel.start_dir
        self.panel.view_parameters['k'] = self.panel.constant




    def on_load_many(self, evt):
        try:
            file_dialog(self.panel,
                message=u'Load from .csv file',
                style=wx.OPEN,
                wildcard=u'*.csv',
                action=self.load_many_params)
        except LocalError, e:
            error_dialog(self, str(e), e.title)

    def on_save_cyclparams(self, evt):
        try:
            file_dialog(self.panel,
                message = u'Save cycloid parameters to file',
                style = wx.SAVE | wx.OVERWRITE_PROMPT,
                wildcard = 'Cycloid files (*.cyc)|*.cyc',
                defaultFile = 'cycloid_params.cyc',
                action = self.save_cyclparams)
        except Exception, e:
            error_dialog(self, str(e), u'Error saving cycloid parameters')

    def save_cyclparams(self, filename):
        tmp = False
        if filename is None:
            filename = os.tempnam(None, 'grid')
            tmp = True
        f = open(filename, 'w')
        for param in self.model.get_parameters_by_category('cycloids'):
            k = param.get_name()
            v = param.get_value()
            if k == 'VARY_VELOCITY' and not v:
                f.write(k + " = False" + "\n")
            else:
                if not self.model.get_param_value(k) in (None, 'None'):
                    f.write(k + " = " + str(self.model.get_param_value(k)) + "\n")
                else:
                    f.write(k + " = None" + "\n")

        f.close()
        return filename, tmp

    def on_load_cyclparams(self, evt):
        try:
            file_dialog(self.panel,
                message = u"Load cycloid parameters from file",
                style = wx.OPEN,
                wildcard = 'Cycloid files (*.cyc)|*.cyc',
                action = self.load_cyclparams)
        except Exception, e:
            error_dialog(self.panel, str(e), u'Error loading cycloid parameters')


    def load_cyclparams(self, filename):
        try:
            f = open(filename)
        except:
            error_dialog(self.panel, 'File error', 'Cannot open file')
        
        for p, v in nvf2dict(f).items():
            if not p in ('k','VARY_VELOCITY', 'STARTING_DIRECTION'):
                self.set_parameter(p,v)

            elif p == 'k':
                if v == 'None':
                    self.model.set_parameter(p, 0)
                    self.panel.constant.SetValue(0)
                else:
                    self.model.set_parameter(p, float(v))

            elif p == 'VARY_VELOCITY':
                if v == 'True' or v == '1':
                    self.panel.constant.Enable()
                    self.panel.vary.SetValue(1)
                else:
                    self.panel.constant.Disable()
                self.model.set_parameter(p,v)

            elif p == 'STARTING_DIRECTION':
                self.model.set_parameter(p,v)
                self.panel.start_dir.SetValue(v)


        f.close()



    #For loading multiple cycloids
    def load_many_params(self, filename):
        self.panel.use_multiple.Enable()
        self.panel.use_multiple.SetValue(True)
        self.EvtSetUseMultiple(None)
        self.model.set_parameter('to_plot_many_cycloids',True)
        self.many_changed = True
        
        paramFile = open(filename, 'rU')
        try:
            rows = list(csv.reader(paramFile))
            params_to_load = rows[0]
            
            self.params_for_cycloids = {}
            i = 0
            for row in rows[1:]:
                self.params_for_cycloids[i] = {}
            
                for j, param in enumerate(params_to_load):
                    self.params_for_cycloids[i].update({param: row[j]})
                self.params_for_cycloids[i].update({'degree_step':0.1})
                i += 1
    
        except:
            error_dialog(self.panel,"Error loading file")
        
        paramFile.close()

    
    def EvtSetDir(self, event):
        self.model.set_parameter('STARTING_DIRECTION', event.GetString())
    
    def EvtSetYeild(self, event):
        assert(float(event.GetString() > 0))
        self.model.set_parameter('YIELD', float(event.GetString()))

    def EvtSetPropStr(self, event):
        assert(float(event.GetString() > 0))
        self.model.set_parameter('PROPAGATION_STRENGTH', float(event.GetString()))

    def EvtSetPropSpd(self, event):
        assert(float(event.GetString() > 0))
        self.model.set_parameter('PROPAGATION_SPEED', float(event.GetString()))
    
    def EvtSetVary(self, event):
        self.model.set_parameter['VARY_VELOCITY'] = self.panel.vary.GetValue()

        if self.panel.vary.GetValue():
            self.panel.constant.Enable()
        else:
            self.panel.constant.Disable()

    def EvtSetConstant(self, event):
        self.model.set_parameter('k', float(event.GetString()))

    def EvtSetUseMultiple(self, event):
        if self.panel.use_multiple.GetValue():
            self.model.set_parameter('to_plot_many_cycloids',  True)
            for ctrl in [self.panel.view_parameters[ p.get_name() ] for p in self.model.get_parameters_by_category('cycloids') ]:
                ctrl.Disable()

        else:
            self.model.set_parameter('to_plot_many_cycloids', False)
            self.panel.use_multiple.SetValue(False)
            for ctrl in [self.panel.view_parameters[p.get_name()] for p in self.model.get_parameters_by_category('cycloids')]:
                ctrl.Enable()


    def EvtSetStartLat(self, event):
        lat = float(event.GetString())
        assert(lat <= 90)
        assert(lat >= -90)
        self.model.set_parameter('STARTING_LATITUDE', float(event.GetString()))

    def EvtSetStartLon(self, event):
        lon = float(event.GetString())
        assert(lon <= 180)
        assert(lon >= -180)
        self.model.set_parameter('STARTING_LONGITUDE',  float(event.GetString()))

    def EvtRandLat(self, event):
        # generates random lat to the 2nd decimal place (current precision of GUI)
        rand_startlat = float("%.2f" % random.uniform(-90, 90))
        # set it to parameter
        self.sc.parameters['STARTING_LATITUDE'] = rand_startlat
        # display it in textctrl
        input_startlat.SetValue('%s', rand_startlat)

    def EvtRandLon(self, event):
        rand_startlon = float("%.2f" % random.uniform(-180, 180))
        # set it to parameters
        self.sc.parameters['STARTING_LONGITUDE'] = rand_startlon
        # display in textctrl
        input_startlon.SetValue('%s', rand_startlon)

class StressListController(BaseController):
    def __init__(self, model, view, stresses_panel):
        BaseController.__init__(self,model,view)
        self.panel = stresses_panel
        self.panel.Bind(wx.EVT_TEXT, self.set_h2Diurn, self.panel.h2Diurn)
        self.panel.Bind(wx.EVT_TEXT, self.set_k2Diurn, self.panel.k2Diurn)
        self.panel.Bind(wx.EVT_TEXT, self.set_l2Diurn, self.panel.l2Diurn)
        self.panel.Bind(wx.EVT_CHECKBOX, self.useUserLove_diurn, self.panel.userDiurn)
        self.panel.Bind(wx.EVT_TEXT, self.set_h2NSR,self.panel.h2NSR)
        self.panel.Bind(wx.EVT_TEXT, self.set_k2NSR, self.panel.k2NSR)
        self.panel.Bind(wx.EVT_TEXT, self.set_l2NSR, self.panel.l2NSR)
        self.panel.Bind(wx.EVT_CHECKBOX, self.useUserLove_nsr, self.panel.userNSR)
    
        self.panel.Bind(wx.EVT_TEXT, self.set_phiRi, self.panel.PWphiRi)
        self.panel.Bind(wx.EVT_TEXT, self.set_thetaRf, self.panel.PWthetaRf)
        self.panel.Bind(wx.EVT_TEXT, self.set_phiRf, self.panel.PWphiRf)
        self.panel.Bind(wx.EVT_TEXT, self.set_thetaTi, self.panel.PWthetaTi)
        self.panel.Bind(wx.EVT_TEXT, self.set_thetaTf, self.panel.PWthetaTf)
        self.panel.Bind(wx.EVT_TEXT, self.set_phiTf, self.panel.PWphiTf)
        
        self.view.get_ctrl_obj('Diurnal').Bind(wx.EVT_CHECKBOX, self.on_set_diurn)
        self.view.get_ctrl_obj('Nonsynchronous Rotation').Bind(wx.EVT_CHECKBOX, self.on_set_nsr)
        self.view.get_ctrl_obj('Ice Shell Thickening').Bind(wx.EVT_CHECKBOX, self.on_set_ist)
        self.view.get_ctrl_obj('Obliquity').Bind(wx.EVT_CHECKBOX, self.on_set_obliq)
        self.view.get_ctrl_obj('Polar Wander').Bind(wx.EVT_CHECKBOX,self.on_set_polar)
        
        self.disable_istparams()
        self.disable_obliq()
        self.disable_polar()
    
    
    def disable_display_diurnlove(self):
        for widg in [self.panel.h2, self.panel.k2, self.panel.l2,
                     self.panel.h2Diurn, self.panel.k2Diurn, self.panel.l2Diurn,
                     self.panel.userDiurn]:
            widg.Disable()
    
    def enable_display_diurnlove(self):
        for widg in [self.panel.h2, self.panel.k2, self.panel.l2,
                     self.panel.h2Diurn, self.panel.k2Diurn, self.panel.l2Diurn,
                     self.panel.userDiurn]:
            widg.Enable()
    
    def disable_display_nsrlove(self):
        for widg in [self.panel.h2, self.panel.k2, self.panel.l2,
                     self.panel.h2NSR, self.panel.k2NSR, self.panel.l2NSR,
                     self.panel.userNSR]:
            widg.Disable()
    
    def enable_display_nsrlove(self):
        for widg in [self.panel.h2, self.panel.k2, self.panel.l2,
                     self.panel.h2NSR, self.panel.k2NSR, self.panel.l2NSR,
                     self.panel.userNSR]:
            widg.Enable()
    
    
    def disable_istparams(self):
        for e in [self.panel.delta_label, self.view.get_ctrl_obj('delta_tc'),
                  self.panel.diffusivity_label, self.view.get_ctrl_obj('diffusivity') ]:
            e.Disable()
    
    
    
    def enable_istparams(self):
        """Don't yet enable diffusivity as it is only rleevant for the viscoelastic case."""
        for e in [self.panel.delta_label, self.view.get_ctrl_obj('delta_tc') ]:
            e.Enable()
    
    def disable_obliq(self):
        for e in [self.panel.obliq_label, self.view.get_ctrl_obj('obliquity'),
                  self.panel.periapsis_label, self.view.get_ctrl_obj('periapsis_arg') ]:
            e.Disable()
    
    def enable_obliq(self):
        for e in [self.panel.obliq_label, self.view.get_ctrl_obj('obliquity'),
                  self.panel.periapsis_label, self.view.get_ctrl_obj('periapsis_arg') ]:
            e.Enable()
    
    
    def enable_polar(self):
        for e in [self.panel.Latitude_label, self.panel.PWthetaRi, self.panel.Longitude_label, self.panel.PWphiRi]:
            e.Enable()
    
    def disable_polar(self):
        for e in [self.panel.Latitude_label, self.panel.PWthetaRi, self.panel.Longitude_label, self.panel.PWphiRi]:
            e.Disable()
    
    def on_set_diurn(self, evt):
        state = self.view.get_ctrl_obj('Diurnal').GetValue()
        self.set_parameter('Diurnal', state)
        if state:
            self.enable_display_diurnlove()
        else:
            self.disable_display_diurnlove()
    def on_set_nsr(self, evt):
        state = self.view.get_ctrl_obj('Nonsynchronous Rotation').GetValue()
        self.set_parameter('Nonsynchronous Rotation', state)
        if state:
            self.enable_display_nsrlove()
        else:
            self.disable_display_nsrlove()
    
    def on_set_ist(self, evt):
        state = self.view.get_ctrl_obj('Ice Shell Thickening').GetValue()
        self.set_parameter('Ice Shell Thickening', state)
        if state:
            self.enable_istparams()
        else:
            self.disable_istparams()
    
    def on_set_obliq(self, evt):
        state = self.view.get_ctrl_obj('Obliquity').GetValue()
        self.set_parameter('Obliquity', state)
        if state:
            self.enable_obliq()
        else:
            self.disable_obliq()
    def on_set_polar(self,evt):
        state = self.view.get_ctrl_obj('Polar Wander').GetValue()
        self.set_parameter('Polar Wander', state)
        if state:
            self.enable_polar()
        else:
            self.disable_polar()
    
    def parse_complex(self, string):
        real, imag = re.split(r'[+-]', string)
        if imag.startswith('i') or imag.startswith('j'):
            return float(real), float(imag[1:])
        elif imag.endswith('i') or imag.endswith('j'):
            return float(real), float(imag[:-1])
    
    def useUserLove_diurn(self, evt):
        if self.panel.userDiurn.GetValue():
            self.panel.model.stress_d['Diurnal'].useUser = True
        else:
            self.panel.model.stress_d['Diurnal'].useUser = False
    
    def useUserLove_nsr(self, evt):
        if self.panel.userDiurn:
            self.model.stress_d['Nonsynchronous Rotation'].useUser = True
        else:
            self.model.stress_d['Nonsynchronous Rotation'].useUser = False
    
    def set_h2Diurn(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Diurnal'].loveUser.update_h2(self.parse_complex(evt.GetString()))
    
    def set_k2Diurn(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Diurnal'].loveUser.update_k2(self.parse_complex(evt.GetString()))
    
    def set_l2Diurn(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Diurnal'].loveUser.update_l2(self.parse_complex(evt.GetString()))
    
    def set_h2NSR(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Nonsynchronous Rotation'].loveUser.update_h2(fself.parse_complex(evt.GetString()))
    
    def set_k2NSR(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Nonsynchronous Rotation'].loveUser.update_k2(self.parse_complex(evt.GetString()))
    
    def set_l2NSR(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Nonsynchronous Rotation'].loveUser.update_l2(self.parse_complex(evt.GetString()))
    
    def set_thetaRi(self, evt):
        #self.sc.stresses_changed = True
        print 'balloony'
        self.model.stress_d['Polar Wander'].UserCoordinates.update_thetaRi(float(evt.GetString()))
    
    def set_phiRi(self, evt):
        #self.sc.stresses_changed = True
        print 'balloony'

        self.model.stress_d['Polar Wander'].UserCoordinates.update_phiRi(float(evt.GetString()))

    def set_thetaRf(self, evt):
        print 'balloony'

        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_thetaRf(float(evt.GetString()))

    def set_phiRf(self, evt):
        print 'balloony'

        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_phiRf(float(evt.GetString()))

    def set_thetaTi(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_thetaTi(float(evt.GetString()))  

    def set_phiTi(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_phiTi(float(evt.GetString()))

    def set_thetaTf(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_thetaTf(float(evt.GetString()))

    def set_phiTf(self, evt):
        #self.sc.stresses_changed = True
        self.model.stress_d['Polar Wander'].UserCoordinates.update_phiTf(float(evt.GetString()))

class GridPanelController(BaseController):
    def __init__(self, model, view, grid_panel):
        BaseController.__init__(self,model,view)
        self.grid_panel = grid_panel
        self.view.view_parameters['nsr_time'].SetMinSize((250, 10))

        self.grid_panel.sb.Bind(wx.EVT_BUTTON, self.on_save)
        self.grid_panel.lb.Bind(wx.EVT_BUTTON, self.on_load)
    
    def update_fields(self):
        '''
            Disable/enable the correct text fields based on which stresses are selected
            '''
        pass
    
    def enable_nsr(self):
        for p in ['TIME_MIN', 'nsr_time', 'TIME_NUM']:
            self.view.get_ctrl_obj(p).Enable()
        for sts in self.grid_panel.nsr_labels:
            sts.Enable()
    
    def disable_nsr(self):
        for p in ['TIME_MIN', 'nsr_time', 'TIME_NUM']:
            self.view.get_ctrl_obj(p).Disable()
        for sts in self.grid_panel.nsr_labels:
            sts.Disable()
    
    def enable_orbit(self):
        for p in ['ORBIT_MIN', 'ORBIT_MAX', 'ORBIT_NUM']:
            self.view.get_ctrl_obj(p).Enable()
        for sts in self.grid_panel.orbit_labels:
            sts.Enable()
    
    def disable_orbit(self):
        for p in ['ORBIT_MIN', 'ORBIT_MAX', 'ORBIT_NUM']:
            self.view.get_ctrl_obj(p).Disable()
        for sts in self.grid_panel.orbit_labels:
            sts.Disable()
    
    def enable_pw(self):
        for p in ['FINAL_LAT', 'FINAL_LONG', 'NUM_INCREMENTS']:
            self.view.get_ctrl_obj(p).Enable()
        for sts in self.grid_panel.pw_labels:
            sts.Enable()
    
    def disable_pw(self):
        for p in ['FINAL_LAT', 'FINAL_LONG', 'NUM_INCREMENTS']:
            self.view.get_ctrl_obj(p).Disable()
        for sts in self.grid_panel.pw_labels:
            sts.Disable()

    def parameter_yrs2secs(self, p):
        v = self.model.get_param_value(p, float)
        if v:
            self.model.set_parameter(p, "%g" % (v*seconds_in_year))

    # converts seconds to years in the paramters
    def parameter_secs2yrs(self, p):
        v = self.model.get_param_value(p, float)
        if v:
            self.model.set_parameter(p, "%g" % (float(self.model.get_param_value(p, float))/seconds_in_year))


    def on_save(self, evt):
        try:
            file_dialog(self.grid_panel,
                message=u"Save to grid file",
                style=wx.SAVE | wx.OVERWRITE_PROMPT,
                wildcard=u'Grid files (*.grid)|*.grid',
                defaultFile=self.model.get_param_value('GRID_ID') + '.grid',
                action=self.save_grid)
        except KeyError, e:
            error_dialog(self.grid_panel, str(e) + ' not defined', 'Grid Error')
        except LocalError, e:
            error_dialog(self.grid_panel, str(e), e.title)

    def on_load(self, evt):
        try:
            file_dialog(self.grid_panel,
                message=u"Load from grid file",
                style=wx.OPEN,
                wildcard=u'Grid files (*.grid)|*.grid',
                action=self.load_grid)
        except LocalError, e:
            error_dialog(self.grid_panel, str(e), e.title)

    def save_grid(self, filename):
        tmp = False
        if filename is None:
            filename = os.tempnam(None, 'grid')
            tmp = True
        f = open(filename, 'w')
        try:
            t_min = self.model.get_param_value('TIME_MIN')
            t_max = self.model.get_param_value('TIME_MAX')
            self.parameter_yrs2secs('TIME_MIN')
            self.parameter_yrs2secs('TIME_MAX')
        except:
            pass
        for param in self.model.get_parameters_by_category('grid_var'):
            f.write(param.get_name() + ' = ' + str(param.get_value()) + '\n')
        try:
            self.model.set_parameter('TIME_MIN', t_min)
            self.model.set_parameter('TIME_MAX', t_max)
        except:
            pass
        f.close()
        if not tmp:
            self.grid_save_changed = False
        return filename, tmp

    def load_grid(self,filename):
        f = open(filename)
        try:
            for p,d in self.grid_panel.grid_parameters_d:
                for v,dv in self.grid_panel.grid_vars_d:
                    self.set_parameter("%s_%s" % (p,v), '')
            for p, v in nvf2dict(f).items():
                try:
                    self.set_parameter(p, v)
                except:
                    pass
            try:
                self.parameter_secs2yrs('TIME_MIN')
                self.parameter_secs2yrs('TIME_MAX')
            except:
                pass
            try:
                self.set_parameter('nsr_time', str(self.model.get_param_value('TIME_MAX', float) - self.get_param_value('TIME_MIN', float)))
            except:
                pass
            self.grid_save_changed = False
            self.grid_changed = True
        except Exception, e:
            print e.__class__.__name__, e
            raise LocalError(e, u'Grid Error')
        finally:
            f.close()



class PointPanelController(BaseController):
    def __init__(self, model, view, point_panel):
        BaseController.__init__(self,model,view)
        self.rows = 0
        init_rows = 20
        self.spin_value = init_rows
        self.panel = point_panel
        self.model.set_parameter('point_rows', init_rows)

        for p, d in self.panel.headers:
            self.panel.view_parameters[p] = []

        self.set_num_rows(init_rows)
                  
        

       
        self.panel.row_ctrl.Bind(wx.EVT_SPINCTRL, self.spinCtrl)
        self.panel.row_ctrl.Bind(wx.EVT_TEXT, self.spinCtrl)
        #self.row_ctrl.Bind(wx.EVT_SPIN_DOWN, lambda evt, szr = pp: self.spin_down(evt, szr))
        # Here we bind the load and save buttons to the respective events
        self.panel.b.Bind(wx.EVT_BUTTON, self.on_calc)
        self.panel.load_b.Bind(wx.EVT_BUTTON, self.load)
        self.panel.save_b.Bind(wx.EVT_BUTTON, self.save)
    
        self.updating = False
        
        for i in range(self.rows):
            self.panel.view_parameters['orbit'][i].Bind(wx.EVT_KILL_FOCUS, lambda evt, row = i: self.on_orbit_update(evt, row))
            self.panel.view_parameters['orbit'][i].Bind(wx.EVT_TEXT, lambda evt, row = i: self.on_orbit_update(evt, row))
            self.panel.view_parameters['t'][i].Bind(wx.EVT_KILL_FOCUS, lambda evt, row = i: self.on_t_update(evt, row))
            self.panel.view_parameters['t'][i].Bind(wx.EVT_TEXT, lambda evt, row = i: self.on_t_update(evt, row))


    #updates the orbit text ctrls when t is changed
    def on_t_update(self, evt, row = 1):
        self.updating = True
        try:
            self.model.set_parameter('t', self.panel.view_parameters['t'][row].GetValue(), point = row)
            sat = self.model.get_satellite()
            o = str(float(self.model.get_param_value('t',point=row - 1))/sat.orbit_period()*360.0*seconds_in_year)
            self.panel.view_parameters['orbit'][row].SetValue(o)
            self.model.set_parameter('orbit', o, point = row)
        except:
            traceback.print_exc()
        self.updating = False
    
    #updates the t text ctrls when orbital position is changed
    def on_orbit_update(self, evt, row = 1):
        self.updating = True
        try:
            self.model.set_parameter('orbit', self.panel.view_parameters['orbit'][row].GetValue(), point = row)
            sat = self.model.get_satellite()
            t = str(float(self.model.get_param_value('orbit',point=row - 1))/360.0*sat.orbit_period()/seconds_in_year)
            self.panel.view_parameters['t'][row].SetValue(t)
            self.model.set_parameter('t', t, point = row)

        except:
            traceback.print_exc()
        self.updating = False
    
    def on_calc(self, evt):
        self.panel.b.SetFocus()
        self.model.calc_tensor(self.rows)
        for p,d in self.panel.header2+self.panel.header3:
            for i in range(self.rows):
                self.panel.view_parameters[p][i].SetValue(self.model.get_param_value(p,point=i))
            
        #r_dialog(self, str(e), e.title)

    def spinCtrl(self, evt):
        spin_value = evt.GetEventObject().GetValue()

        if spin_value == '':
            spin_value = 1
        
        if (int(spin_value) > self.rows):
            self.onUp(int(spin_value))
        else:
            self.spin_down(int(spin_value))

    def onUp(self, spin_value):
        self.panel.pp.SetRows(spin_value)
        self.panel.tp.SetRows(spin_value)
        self.panel.sp.SetRows(spin_value)
        for i in range(spin_value-self.rows):
            self.rows +=1
            self.add_row(self.panel.fieldPanel, self.panel.pp, self.panel.header1,'0')
            self.add_row(self.panel.fieldPanel,self.panel.tp, self.panel.header2, '')
            self.add_row(self.panel.fieldPanel,self.panel.sp, self.panel.header3, '')

        self.panel.fieldPanel.Layout()
        self.panel.Layout()
        self.panel.fieldPanel.SetupScrolling()
        self.model.set_parameter('point_rows',self.rows)
        

    def add_row(self, panel, sz, params_d, defaultval):
        for p,d in params_d:
            text = wx.TextCtrl(self.panel.fieldPanel, style=wx.TE_PROCESS_ENTER, name = p)
            sz.Add(text, flag=wx.ALL|wx.EXPAND)
            text.Bind(wx.EVT_CHAR,self.OnChar)
            text.Bind(wx.EVT_TEXT, lambda evt, index = self.rows-1: self.OnText_for_lists(evt, index))
            self.panel.view_parameters[p].append(text)
            self.model.set_parameter(p,defaultval, point=self.rows)
            text.SetValue(defaultval)




    def spin_down(self, spin_value):
        self.panel.pp.SetRows(spin_value)
        self.panel.tp.SetRows(spin_value)
        self.panel.sp.SetRows(spin_value)
        for i in range(self.rows - spin_value):
            for p,d in self.panel.headers:
                self.panel.view_parameters[p][-1].Destroy()
                del self.panel.view_parameters[p][-1]
                self.model.get_parameter(p).delete_last()

        self.rows = spin_value
        self.model.set_parameter('point_rows',self.rows)
        self.panel.fieldPanel.Layout()

    def load(self, evt):
        try:
            file_dialog(self.panel,
                        message=u"Load from CSV file",
                        style=wx.OPEN,
                        wildcard='CSV files (*.csv)|*.csv',
                        action=self.load_entries)
        except Exception, e:
            traceback.print_exc()

    def set_num_rows(self,num_rows):
        self.panel.pp.SetRows(num_rows)
        self.panel.sp.SetRows(num_rows)
        self.panel.tp.SetRows(num_rows)
        rows_to_add = num_rows - self.rows
        if (rows_to_add):
            
            for j in range(rows_to_add):
                self.rows +=1
                self.add_row(self.panel.fieldPanel,self.panel.pp, self.panel.header1, '0')
                self.add_row(self.panel.fieldPanel,self.panel.tp, self.panel.header2, '')
                self.add_row(self.panel.fieldPanel,self.panel.sp, self.panel.header3, '')
        
        else:
            for j in range(rows_to_add):
                for p,d in self.panel.headers:
                    self.view,view_parameters[p][-1].Destroy()
                    del self.view.view_parameters[p][-1]
                    self.model.get_parameter[p].delete_last()

        self.panel.row_ctrl.SetValue(num_rows)
        self.panel.fieldPanel.Layout()
        self.panel.fieldPanel.SetupScrolling()
        self.model.set_parameter('point_rows',self.rows)



    def load_entries(self, filename):
        f = open(filename)
        csvreader = csv.reader(f)
        coord = csvreader.next()  #Skip headers
        data = list(csvreader)
        self.set_num_rows(len(data))
        
        try:
            keys = ['theta', 'phi', 't', 'orbit']
            
            for i,coord in enumerate(data):
                for key in keys:
                    val = coord[keys.index(key)]
                    self.panel.view_parameters[key][i].SetValue(val)
                    self.model.set_parameter(key, val, point = i)
        except:
            traceback.print_exc()
        finally:
            f.close()
            self.panel.fieldPanel.Layout()
            self.panel.fieldPanel.SetupScrolling()
            self.panel.Layout()

    
    #opens save dialog
    def save(self, evt):
        file_dialog(self.panel,
                    message=u"Save to CSV file",
                    style=wx.SAVE,
                    wildcard='CSV files (*.csv)|*.csv',
                    defaultFile='untitled.csv',
                    action=self.save_pointcalc)
    
    #parses text ctrls and writes to csv
    def save_pointcalc(self, filename=None):
        tmp = False
        if not filename:
            filename = os.tempnam(None, 'csv')
            tmp = True
        f = open(filename, 'wb')
        writer = csv.writer(f)
        headers = [u'theta [degrees]', 'phi [degrees]', 't [yrs]', 'orbital pos [degrees]', \
                   'Stt [kPa]', 'Spt [kPa]', 'Spp [kPa]', \
                   'sigma1 [kPa]', 'sigma3 [kPa]', 'alpha [degrees]'] 
        writer.writerow(headers)
        keys = ['theta', 'phi', 't', 'orbit',\
        "Ttt", "Tpt", "Tpp", \
        "s1", "s3", "a"]
        for i in range(self.rows):
            row = [self.model.get_param_value(key, point=i) for key in keys]
            writer.writerow(row)
        f.close()


class ScalarPlotPanelController(BaseController):
    step_field = 'STEP'

    def __init__(self, model, view, ScalarPlot_panel):
        BaseController.__init__(self,model,view)
        self.panel = ScalarPlot_panel
        
        #self.panel.stepspin.SetValue(self.step)
        #self.panel.stepspin.Bind(wx.EVT_SPINCTRL, self.adjust_step)

        self.orbit_pos = 0
        self.nsr_pos = 0
        self.panel = ScalarPlot_panel
        self.orbit_hidden = self.nsr_hidden = self.polar_hidden = False
        self.changed = True
        for p in ['LAT', 'LON', 'VAL']:
            self.panel.val_p[p].SetEditable(False)

        self.view.get_ctrl_obj('projection').Bind(wx.EVT_TEXT, self.on_change_projection)
        self.view.get_ctrl_obj('field').Bind(wx.EVT_TEXT, self.OnText)
        self.view.get_ctrl_obj('to_plot_principal_vectors').Bind(wx.EVT_TEXT, self.OnText)
        self.view.get_ctrl_obj('to_plot_latitude_vectors').Bind(wx.EVT_TEXT, self.OnText)                
        self.view.get_ctrl_obj('to_plot_longitude_vectors').Bind(wx.EVT_TEXT, self.OnText)
        self.view.get_ctrl_obj('to_plot_shear_vectors').Bind(wx.EVT_TEXT, self.OnText)

            
    def on_change_projection(self,evt):
        self.model.set_parameter('projection', evt.GetEventObject().GetValue())
        self.panel.prepare_plot()

    def on_orbit_updated(self, val):
        if self.updating:
            return
        self.orbit_pos = self.panel.scp.orbit_slider.val
        self.updating = True
        self.panel.scp.nsr_slider.first()
        self.nsr_pos = 0
        self.updating = False
        self.plot()

    def on_nsr_updated(self, val):
        if self.updating:
            return
        self.nsr_pos = self.panel.scp.nsr_slider.val
        self.updating = True
        self.panel.scp.orbit_slider.first()
        self.orbit_pos = 0
        self.updating = False
        self.plot()
    
    def prepare_plot(self):
        b = wx.BusyInfo(u"Performing calculations. Please wait.", self)
        wx.SafeYield()
        self.prepare_plot_series()
        del b
    
    def get_grid_time(self):
        if self.orbit_pos > self.nsr_pos:
            s = self.model.get_satellite()
            print self.orbit_pos/360.0*s.orbit_period()
            return self.orbit_pos/360.0*s.orbit_period()
        else:
            print self.nsr_pos*seconds_in_year
            return self.nsr_pos*seconds_in_year

    def mk_change_param(self, k):
        def on_change(evt):
            if k == 'direction':
                #handles change of east-west positivity
                # reverse
                temp_min = -self.sc.get_parameter(float, "LON_MAX")
                temp_max = -self.sc.get_parameter(float, "LON_MIN")
                self.sc.set_parameter("LON_MIN", temp_min)
                self.sc.set_parameter("LON_MAX", temp_max)
                self.plot()
            else:
                self.sc.set_parameter(k, self.parameters[k].GetValue())
                self.plot()
        return on_change

    def load_scale(self, k):
        try:
            if k < 0:
                self.lbound = int(config.load('PLOT_LBOUND'))
            else:
                self.ubound = int(config.load('PLOT_UBOUND'))
        except:
            if k < 0:
                self.lbound = None
            else:
                self.ubound = None
        if k < 0:
            return self.lbound
        else:
            return self.ubound

    def save_scale(self):
        config.save(PLOT_LBOUND=self.lbound, PLOT_UBOUND=self.ubound)
        
    def select_scale(self, evt):
        l = self.lbound_ctrl.GetValue()
        u = self.ubound_ctrl.GetValue()
        try:
            fl = int(l)
            fu = int(u)
            if self.lbound != l or self.ubound != u:
                self.lbound = l
                self.ubound = u
                self.save_scale()
                self.scp.plot_scale(self.scale(), "%.f kPa")
                self.select_color_range(fl*1000, fu*1000)
        except:
            self.lbound_ctrl.SetValue(self.lbound)
            self.ubound_ctrl.SetValue(self.ubound)

    def select_color_range(self, vmin, vmax):
        self.plot()
        self.cb.set_clim(vmin, vmax)
        self.cb.update_bruteforce(self.im)
        self.cb.draw_all()
        self.draw()


    
    def on_polar_updated(self, val):
        if self.updating:
            return
        self.polar_pos = self.panel.scp.polar_slider.val
        self.updating = True
        self.panel.scp.orbit_slider.first()
        self.orbit_pos = 0
        self.updating = False
        self.plot()

    def bind_load_save_buttons(self):
        self.panel.shapeLoad.Bind(wx.EVT_BUTTON, self.on_load_shape)
        self.panel.shapeSave.Bind(wx.EVT_BUTTON, self.on_save_shape)
        self.panel.netLoad.Bind(wx.EVT_BUTTON, self.on_load_netcdf)
        self.panel.netSave.Bind(wx.EVT_BUTTON, self.on_save_netcdf)


    def plot_sizer_controller(self):

        self.init_orbit_slider()
        self.init_nsr_slider()
        self.init_polar_slider()

        self.plot_fields = {}
        self.plot_vectors = {}
        self.n_interp = 10
        self.tick_formatter = KPaFormatter()
        self.panel.scp.canvas.callbacks.connect('motion_notify_event', self.on_move_in_plot)
        self.panel.scp.orbit_slider.on_changed(self.on_orbit_updated)
        self.panel.scp.nsr_slider.on_changed(self.on_nsr_updated)
        self.panel.scp.polar_slider.on_changed(self.on_polar_updated)
        #self.panel.scp.save_orbit_series = self.save_orbit_series
        self.panel.scp.save_nsr_series = self.save_nsr_series
        self.panel.scp.save_polar_series = self.save_polar_series
        self.orbit_pos = int(self.model.get_parameter('ORBIT_MIN').get_value())
        self.nsr_pos = float(self.model.get_parameter('TIME_MIN').get_value())
        self.polar_pos = float(self.model.get_parameter('TIME_MIN').get_value())
        self.updating = False

    def lineaments_sizer_controller(self):
        self.l_count = 2
        self.generated = { 'data': [], 'color': wx.ColourData(), 'lines': [] }
        self.loaded = { 'data': [], 'color': wx.ColourData(), 'lines': [] }
        self.first_run = True
        self.model.set_parameter('to_plot_lineaments',True)
        self.panel.plot_lins.Bind(wx.EVT_CHECKBOX, self.generate_lins)
        self.panel.l_count_tc.SetValue(str(self.l_count))
        self.panel.l_count_tc.Bind(wx.EVT_TEXT, self.generate_lins)


    def cycloids_sizer_controller(self):
        self.cycl_generated = { 'cycdata': [], 'color': wx.ColourData(), 'arcs': [] }
        self.cycl_loaded = { 'cycdata': [], 'color': wx.ColourData(), 'arcs': [] }
        self.first_run = True   # for lineaments

        self.panel.plot_cycl.Bind(wx.EVT_CHECKBOX, self.generate_cycl)
        self.panel.save_many.Bind(wx.EVT_BUTTON, self.save_many_cycloids)

        # create sizers
    def generate_lins(self, evt):
        print 'generate_lins'
        
        if self.panel.plot_lins.GetValue():     # plot only if box is checked
            self.l_count = int(self.l_count_tc.GetValue())
        else:
            self.l_count = 0

        self.first_run = False
        b = wx.BusyInfo(u"Performing calculations. Please wait.", self)
        wx.SafeYield()
        self.generated['data'] = self.lingen(self.l_count)
        self.generated['lines'] = []
        del b
        self.plot()
        
        self.plot_lineaments()

        print 'end generate_lins'


    def generate_cycl(self, evt):
        if self.panel.plot_cycl.GetValue(): # plot only if box is checked
            # print self.basemap_ax
            # print self.calc
            # print self.parameters
            self.model.set_parameter('to_plot_cycloids', True)
            self.plot()
        else:
            self.model.set_parameter('to_plot_cycloids', False)
            self.plot()

    #@into_hbox
    def save_orbit_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_orbit_pos = self.orbit_pos
        sat = self.model.get_satellite()
        orbit_period = sat.orbit_period()
        o = self.model.get_param_value('ORBIT_MIN', 0, float)
        om = self.model.get_param_value('ORBIT_MAX', 0, float)
        n = self.model.get_param_value('ORBIT_NUM', 0, float)
        s = (om - o)/n
        self.hide_orbit_controls()

        localtime = time.asctime(time.localtime(time.time()))
        location = dir + "/" + elf.model.get_param_value('SYSTEM_ID')
        directory = location + "/" + localtime
        if os.path.isdir(location):
            os.mkdir(directory)
        else:
            os.mkdir(location)
            os.mkdir(directory)

        while o <= om:
            self.orbit_pos = o
            self.plot_no_draw()
            self.scp.orbit_slider.set_val(self.orbit_pos)
            self.scp.figure.savefig("%s/orbit_%03d.%02d.png" %
                (directory, int(self.orbit_pos), round(100.*(self.orbit_pos - int(self.orbit_pos)))),
                bbox_inches='tight', pad_inches=1.5)
            o += s
        self.orbit_pos = old_orbit_pos
        self.reveal_orbit_controls()
        self.init_orbit_slider()
        self.scp.orbit_slider.set_val(self.orbit_pos)
        self.plot()
        del b
    
    def save_nsr_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_nsr_pos = self.nsr_pos
        nm = self.sc.get_parameter(float, 'TIME_MIN', 0)
        s = self.sc.get_parameter(float, 'nsr_time', 0)
        n = self.sc.get_parameter(int, 'TIME_NUM', 0)
        self.hide_nsr_controls()

        localtime = time.asctime(time.localtime(time.time()))
        location = dir + "/" + self.sc.parameters['SYSTEM_ID']
        directory = location + "/" + localtime
        if os.path.isdir(location):
            os.mkdir(directory)
        else:
            os.mkdir(location)
            os.mkdir(directory)
            
        for k in range(0, n+1):
            self.nsr_pos = nm + s*k
            self.scp.nsr_slider.set_val(self.nsr_pos)
            self.plot_no_draw()
            self.scp.figure.savefig("%s/nsr_%03d.png" % (directory, k), bbox_inches='tight', pad_inches=0.5)
        self.nsr_pos = old_nsr_pos
        self.reveal_nsr_controls()
        self.init_nsr_slider()
        self.scp.nsr_slider.set_val(self.nsr_pos)
        self.plot()
        del b

    def save_polar_series(self, dir='.'):
        b = wx.BusyInfo(u"Saving images. Please wait.", self)
        wx.SafeYield()
        old_polar_pos = self.polar_pos
        nm = self.sc.get_parameter(float, 'TIME_MIN', 0)
        s = self.sc.get_parameter(float, 'polar_time', 0)
        n = self.sc.get_parameter(int, 'TIME_NUM', 0)
        self.hide_polar_controls()
        for k in range(0, n+1):
            self.polar_pos = nm + s*k
            self.scp.polar_slider.set_val(self.polar_pos)
            self.plot_no_draw()
            self.scp.figure.savefig("%s/polar_%03d.png" % (dir, k), bbox_inches='tight', pad_inches=0.5)
        self.polar_pos = old_polar_pos
        self.reveal_polar_controls()
        self.init_polar_slider()
        self.scp.polar_slider.set_val(self.polar_pos)
        self.plot()
        del b



    def save_many_cycloids(self, evt):
        # if a set of parameters from *.csv hasn't been uploaded, treat it like an error
        # with a popup window
        if not self.sc.parameters["to_plot_many_cycloids"]:
            errorMsg = """Please upload a set of cycloid parameters from *.csv file."""
            msg = wx.MessageDialog(self, errorMsg, "No input file found!", wx.OK | wx.ICON_ERROR)
            msg.ShowModal()
            msg.Destroy()

        # otherwise generate and save plots in designated folder
        else:
            chooseFolder = wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE)
            
            # so that folderName can accessed outside
            folderName = ""

            if chooseFolder.ShowModal() == wx.ID_OK:
                folderName = chooseFolder.GetPath()

            # Blanks out the entire window, which prevents people from changing tabs
            # or doing anything else, which happens naturally anyways.
            # self.Hide()

            i = 0
            while i < len(self.parameters['YIELD']):

                # create cycloid
                threshold = float(self.parameters['YIELD'][i])
                strength = float(self.parameters['PROPAGATION_STRENGTH'][i])
                speed = float(self.parameters['PROPAGATION_SPEED'][i])
                lon = float(self.parameters['STARTING_LONGITUDE'][i])
                lat = float(self.parameters['STARTING_LATITUDE'][i])
                propdir = self.parameters['STARTING_DIRECTION']
                
                print threshold, strength, speed, lon, lat, propdir
                print self.calc
                print "\n"

                plotcoordsonbasemap(self.calc, self.basemap_ax,
                                    threshold, strength, speed, lon, lat,
                                    propdir,
                                    self.sc.get_parameter(float, 'ORBIT_MAX', 360))

                # save cycloid
                plotName = str(threshold) + "_" + str(strength) + "_" +  str(speed) + "_" + str(lat) + "_" + str(lon) + "_" + str(propdir)
                self.scp.figure.savefig(folderName + '/' + plotName + ".png", bbox_inches='tight')

                # To have one cycloid saved per image, clear basemap if cycloid was plotted
                if self.ax.lines != []:
                    # print self.ax.lines
                    # self.ax.lines.pop(0)
                    self.ax.lines = []

                i += 1
            
            # when thread is done, show GUI again
            # self.Show()
    def on_page_change(self):
        #self.panel.changed = True
        self.panel.plot()
        print 'hey'


    def OnChar(self,event):
        charEntered= event.GetKeyCode()
        if (charEntered >= 48 and charEntered <= 57) or charEntered == 8 or charEntered == 9 or charEntered == 13 or charEntered == 45 or charEntered ==46:
            event.Skip()

    def OnText(self,event):
        if not event.GetEventObject().GetValue() == 'None':
            self.sc.parameters[event.GetEventObject().GetName()] = float(event.GetEventObject().GetValue())
        
        else:
            self.sc.parameters[event.GetEventObject().GetName()] = None
    def load_many(self, evt):
        try:
            file_dialog(self,
                message=u'Load from .csv file',
                style=wx.OPEN,
                wildcard=u'*.csv',
                action=self.load_many_params2)
        except LocalError, e:
            error_dialog(self.panel, str(e), e.title)
    def on_save_cyclparams(self, evt):
        try:
            file_dialog(self,
                message = u'Save cycloid parameters to file',
                style = wx.SAVE | wx.OVERWRITE_PROMPT,
                wildcard = 'Cycloid files (*.cyc)|*.cyc',
                defaultFile = 'cycloid_params.cyc',
                action = self.save_cyclparams)
        except Exception, e:
            error_dialog(self.panel, str(e), u'Error saving cycloid parameters')

    def save_cyclparams(self, filename):
        tmp = False
        if filename is None:
            filename = os.tempnam(None, 'grid')
            tmp = True
        f = open(filename, 'w')
        for k,v in self.sc.cycloid_parameters_d.items():
            if k == 'VARY_VELOCITY' and not v:
                f.write(k + " = False" + "\n")
            else:
                if self.sc.parameters.has_key(k):
                    f.write(k + " = " + str(self.sc.parameters[k]) + "\n")
                else:
                    f.write(k + " = None" + "\n")

        f.close()
        
        if not tmp:
            self.grid_save_changed = False
        return filename, tmp

    def on_load_cyclparams(self, evt):
        try:
            file_dialog(self.panel,
                message = u"Load cycloid parameters from file",
                style = wx.OPEN,
                wildcard = 'Cycloid files (*.cyc)|*.cyc',
                action = self.load_cyclparams)
        except Exception, e:
            error_dialog(self.panel, str(e), u'Error loading cycloid parameters')


    def load_cyclparams(self, filename):
        try:
            f = open(filename)
        except:
            error_dialog(self.panel, 'File error', 'Cannot open file')
        
        for p, v in nvf2dict(f).items():
            if not p in ('k','VARY_VELOCITY', 'STARTING_DIRECTION'):
                if v == 'None':
                    self.sc.parameters[p] = 'None'
                else:
                    self.sc.parameters[p] = float(v)

            elif p == 'k':
                if v == 'None':
                    self.sc.parameters[p] = 0
                    self.constant.SetValue(0)
                else:
                    self.sc.parameters[p] = float(v)

            elif p == 'VARY_VELOCITY':
                if v == 'True' or v == '1':
                    self.constant.Enable()
                    self.vary.SetValue(1)
                else:
                    self.constant.Disable()
                self.sc.parameters[p] = v

            elif p == 'STARTING_DIRECTION':
                self.sc.parameters[p] = v
                self.start_dir.SetValue(v)

        self.updateFields()
        self.cycloid_saved = True
        f.close()



    #For loading multiple cycloids
    def load_many_params(self, filename):
        self.use_multiple.Enable()
        self.use_multiple.SetValue(True)
        self.EvtSetUseMultiple(None)
        self.sc.parameters['to_plot_many_cycloids'] = True
        self.sc.many_changed = True
        
        paramFile = open(filename, 'rU')
        try:
            rows = list(csv.reader(paramFile))
            params_to_load = rows[0]
            
            self.sc.params_for_cycloids = {}
            i = 0
            for row in rows[1:]:
                self.sc.params_for_cycloids[i] = {}
            
                for j, param in enumerate(params_to_load):
                    self.sc.params_for_cycloids[i].update({param: row[j]})
                self.sc.params_for_cycloids[i].update({'degree_step':0.1})
                i += 1
    
        except:
            error_dialog(self,"Error loading file")
        
        paramFile.close()

    def load_many_params2(self, filename, num_cycloids_per_plot=18):
        self.use_multiple.Enable()
        self.use_multiple.SetValue(True)
        self.EvtSetUseMultiple(None)
        self.sc.parameters['to_plot_many_cycloids'] = True
        self.sc.many_changed = True

        paramFile = open(filename, 'rU')
        rows = list(csv.reader(paramFile))
        params_to_load = rows[0]
        
        self.sc.params_for_cycloids = {}
        i = 0
        while rows[self.currentIndex][0] == '':
            self.currentIndex +=1
        for row in rows[self.currentIndex : self.currentIndex+num_cycloids_per_plot+1]:
            print row
        
            self.sc.params_for_cycloids[i] = {}
            for j, param in enumerate(params_to_load):
                self.sc.params_for_cycloids[i].update({param: str(row[j]) })
            self.sc.params_for_cycloids[i].update({'degree_step':0.3})
            i += 1

        self.currentIndex += num_cycloids_per_plot+1
        f = open('lastIndex.txt','w')
        f.write(str(self.currentIndex))
        f.close()


        #except:
         #3   error_dialog(self,"Error loading file")
        
        paramFile.close()





    def updateFields(self):
        if self.sc.parameters['VARY_VELOCITY'] == 'True' or self.sc.parameters['VARY_VELOCITY'] == '1':
            self.vary.SetValue(True)
            if self.sc.parameters.has_key('k'):
                self.constant.Enable()
                self.constant.SetValue(str(self.sc.parameters['k']))

        if self.sc.parameters.has_key('STARTING_DIRECTION'):
            self.start_dir.SetValue(self.sc.parameters['STARTING_DIRECTION'])
        
        
        for p, textctrl in self.textCtrls.items():
            if self.sc.parameters.has_key(p):
                    textctrl.SetValue(str(self.sc.parameters[p]))



    def load_shape(self, filename):
        # walk around char const * restriction
        sf = os.path.splitext(str(filename))[0] + '.shp'
        self.loaded['data'] = shp2lins(sf, stresscalc=self.calc)
        self.loaded['lines'] = []
        d = wx.ColourDialog(self, self.loaded['color'])
        if (d.ShowModal() == wx.ID_OK):
            self.loaded['color'] = d.GetColourData()
        self.plot()


class MainController(BaseController):

    def __init__(self, app):
        self.model = DataModel()
        self.view = View(None, self.model, self)
        self.view.Bind(wx.EVT_MENU,self.onExport,self.view.export)
        self.view.Bind(wx.EVT_MENU, self.onLoad,self.view.load)
        self.view.Bind(wx.EVT_MENU, self.onQuit, self.view.quit)

        self.stp_controller = StressListController( self.model, self.view, self.view.p.stp)
        self.gp_controller = GridPanelController(self.model, self.view, self.view.p.gp)
        self.pp_controller = PointPanelController(self.model, self.view,  self.view.p.pp)
        self.cp_controller = CycloidsPanelController(self.model, self.view,  self.view.p.cp)
        self.spp_controller = ScalarPlotPanelController(self.model, self.view,  self.view.p.spp)
        self.view.p.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.page_changed)
        self.bind_all()
        self.view.Show()

    def page_changed(self,evt):
        if isinstance(evt.EventObject.GetChildren()[evt.Selection], GridCalcPanel): #Certain fields might have to be disabled if stresses are changed
            self.gp_controller.update_fields()

        elif isinstance(evt.EventObject.GetChildren()[evt.Selection], ScalarPlotPanel):
            self.spp_controller.on_page_change() 

        evt.Skip()


    def onExport(self,evt):
        file_dialog(self.view,
            message=u"Save configuration",
            style=wx.SAVE | wx.OVERWRITE_PROMPT,
            wildcard='Satstress files (*.sats)|*.sats',
            action=self.save_file)
      

    def onLoad(self,evt):
        file_dialog(self.view,
            message=u"Load configuration",
            style=wx.OPEN,
            wildcard='Satstress files (*.sats)|*.sats',
            action=self.load_file)
        
    
    def load_file(self,filename):
        f = open(filename)
        file_dict = nvf2dict(f)
        try:
            self.pp_controller.set_num_rows(float(file_dict['point_rows']))
        except:
            pass
        for k,v in file_dict.items():
            if k == 'point_rows':
                pass
            elif str(v)[0] == '[':  #Load in a list
                l = eval(v)
                for i in range(len(l)):
                    self.set_parameter(k,l[i],point=i)
            else:
                self.set_parameter(k,v)

    def save_file(self,filename):
        f = open(filename,'w')
        for param_name, param in self.model.get_params_dict().items():
            v = param.get_value()
            if v or v == 'to_plot_many_cycloids': #Don't want to save to_plot_many_cycloids simply because this option shouldn't be loaded since the cycloids from the cycloids file aren't saved
                f.write(param_name + ' = ' + str(v) + '\n')
        f.close()

    
    def onQuit(self, evt):
        self.view.Close()

   



def main():
    #make Mac OS app be able to run calcLoveWahr4Layer from Resources
    #directory in application bundle
    os.environ['PATH'] += os.path.pathsep+os.path.abspath(os.curdir)
    app = wx.App(1) # The 0 aka false parameter means "don't redirect stdout and stderr to a window"
    controller = MainController(app)
    app.MainLoop()

if __name__ == '__main__':
    main()
