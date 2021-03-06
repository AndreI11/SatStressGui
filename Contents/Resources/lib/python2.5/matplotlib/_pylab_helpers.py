"""
Manage figures for pyplot interface.
"""

import sys, gc

def error_msg(msg):
    print >>sys.stderr, msgs

class Gcf(object):
    """
    Manage a set of integer-numbered figures.

    This class is never instantiated; it consists of two class
    attributes (a list and a dictionary), and a set of static
    methods that operate on those attributes, accessing them
    directly as class attributes.

    Attributes:

        *figs*:
          dictionary of the form {*num*: *manager*, ...}

        *_activeQue*:
          list of *managers*, with active one at the end

    """
    _activeQue = []
    figs = {}

    @staticmethod
    def get_fig_manager(num):
        """
        If figure manager *num* exists, make it the active
        figure and return the manager; otherwise return *None*.
        """
        figManager = Gcf.figs.get(num, None)
        if figManager is not None:
            Gcf.set_active(figManager)
        return figManager

    @staticmethod
    def destroy(num):
        """
        Try to remove all traces of figure *num*.

        In the interactive backends, this is bound to the
        window "destroy" and "delete" events.
        """
        if not Gcf.has_fignum(num): return
        figManager = Gcf.figs[num]

        # There must be a good reason for the following careful
        # rebuilding of the activeQue; what is it?
        oldQue = Gcf._activeQue[:]
        Gcf._activeQue = []
        for f in oldQue:
            if f != figManager:
                Gcf._activeQue.append(f)

        del Gcf.figs[num]
        #print len(Gcf.figs.keys()), len(Gcf._activeQue)
        figManager.destroy()
        gc.collect()

    @staticmethod
    def has_fignum(num):
        """
        Return *True* if figure *num* exists.
        """
        return num in Gcf.figs

    @staticmethod
    def get_all_fig_managers():
        """
        Return a list of figure managers.
        """
        return Gcf.figs.values()

    @staticmethod
    def get_num_fig_managers():
        """
        Return the number of figures being managed.
        """
        return len(Gcf.figs.values())

    @staticmethod
    def get_active():
        """
        Return the manager of the active figure, or *None*.
        """
        if len(Gcf._activeQue)==0:
            return None
        else: return Gcf._activeQue[-1]

    @staticmethod
    def set_active(manager):
        """
        Make the figure corresponding to *manager* the active one.
        """
        oldQue = Gcf._activeQue[:]
        Gcf._activeQue = []
        for m in oldQue:
            if m != manager: Gcf._activeQue.append(m)
        Gcf._activeQue.append(manager)
        Gcf.figs[manager.num] = manager

