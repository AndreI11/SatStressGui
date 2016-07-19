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

class Model:
    def __init__(self):
        pass

class View(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, *args, **kw)
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.p = SatStressPanel(self)
        self.GetSizer().Add(self.p, 1, wx.ALL|wx.EXPAND, 10)

        menubar = wx.MenuBar()

        ##### 'File' option of menubar #####
        File = wx.Menu()
        export = File.Append(wx.ID_SAVE, '&Export\tCtrl+S', 'Save all variables')
        self.Bind(wx.EVT_MENU,self.onExport, export)
        load = File.Append(wx.ID_OPEN, '&Load\tCtrl+O', 'Load a set of variables')
        self.Bind(wx.EVT_MENU, self.onLoad, load)
        quit = File.Append(wx.ID_ANY, '&Quit\tCtrl+Q', 'Quit Application')
        self.Bind(wx.EVT_MENU, self.onQuit, quit)

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
    
    def onExport(self,evt):
        try:
            file_dialog(self,
                    message=u"Save configuration",
                    style=wx.SAVE | wx.OVERWRITE_PROMPT,
                    wildcard='Satstress files (*.sats)|*.sats',
                    action=self.saveFile)
        except Exception, e:
            error_dialog(self, str(e), u'Save Error')

    def onLoad(self,evt):
        try:
            file_dialog(self,
                        message=u"Load configuration",
                        style=wx.OPEN,
                        wildcard='Satstress files (*.sats)|*.sats',
                        action=self.loadFile)
        except Exception, e:
            error_dialog(self, str(e), u'Load Error')
    
    def loadFile(self,filename):
        f = open(filename)
        
        for k,v in nvf2dict(f).items():
            if k == 'point_rows':
                self.p.tp.set_num_rows(float(v))
            if str(v)[0] == '[':  #Load in a list
                l = eval(v)
                for i in range(1, len(l)):
                    self.p.sc.set_parameter(k, l[i], point = i)
            else:
                self.p.sc.set_parameter(k,v)
        
        self.p.sc.grid_changed = True
        self.p.sc.nsr_period_seconds2years()
        self.p.cy.updateFields() #Update the text fields in cycloids tab.
        self.p.nb.GetCurrentPage().update_parameters() #Update the current page's fields

    def saveFile(self,filename):
        f = open(filename,'w')
        for p,v in self.p.sc.parameters.items():
            if v or v == 'to_plot_many_cycloids': #Don't want to save to_plot_many_cycloids simply because this option shouldn't be loaded since the cycloids from the cycloids file aren't saved
                f.write(p + ' = ' + str(v) + '\n')
        f.close()

    def onQuit(self, evt):
        self.Close()


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
        if self.p.sc.saveable_changed():
            dialog = wx.MessageDialog(self,
                message = "To save your parameters and/or plot, return to the relavent tab and click the appropriate button",
                caption = "Are you sure you want to quit without saving?")
            response = dialog.ShowModal() # show and disallows other input until closed

            if (response == wx.ID_OK):
                self.exit(event)
            else:
                event.StopPropagation()
        # if all saveable parameters have been saved, no need for popup window
        else:
            self.exit(event)

    def exit(self, evt):
        """ seems like a 
        """
        sys.exit(0)

   

class Controller:
    def __init__(self, app):
           self.model = Model()
           self.view1 = View(None)
           self.view1.Show()
           
   



def main():
    #make Mac OS app be able to run calcLoveWahr4Layer from Resources
    #directory in application bundle
    os.environ['PATH'] += os.path.pathsep+os.path.abspath(os.curdir)
    app = wx.App(1) # The 0 aka false parameter means "don't redirect stdout and stderr to a window"
    controller = Controller(App)
    app.MainLoop()

if __name__ == '__main__':
    main()
