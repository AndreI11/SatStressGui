from satstress.satstress import *
from satstress.lineament import plotlinmap, Lineament, lingen_nsr, shp2lins, lins2shp  
from satstress.cycloid import Cycloid, plotcoordsonbasemap
from satstress.stressplot import scalar_grid, vector_points
import satstress.physcon
import satstress.gridcalc

import traceback


seconds_in_year = 31556926.0  ## 365.24 days


class LocalError(Exception):
    def __init__(self, e, title):
        self.msg = str(e)
        self.title = title
    
    def __str__(self):
        return self.msg

class Parameter:
    def __init__(self, name=None, value=None, param_type=None, category=None):
        '''
        @name - name of the parameter that will be used to identify it. NSR_PERIOD, etc
        @value - the value, such as 25, 'Europa', etc
        @type - the data type of the parameter. Either 'string', 'int', or 'float'
        @category - 
        '''
        self.value = value
        self.name = name
        self.param_type = param_type
        self.category= category

    def set_value(self, value):
        self.value = value

    def set_name(self,name):
        self.name = name

    def set_param_type(self, param_type):
        self.param_type = param_type
    def get_name(self):
        return self.name

    def get_value(self, default_val=0):
        if self.value == None:
            return default_val
        return self.value

    def get_category(self):
        return self.category

    def get_param_type(self):
        return self.param_type

class ParameterList:
    def __init__(self, name=None, values=None, param_type=None, category=None):
        '''
            @name - name of the parameter that will be used to identify it. NSR_PERIOD, etc
            @valueList - the values, such as 25, 100, etc that will be stored
            @type - the data type of the parameter. Either 'string', 'int', or 'float'
            @category -
            '''
        if values == None:
            values = []
        self.values = values
        self.name = name
        self.param_type = param_type
        self.category= category
    

    def set_value(self, value, point):
    	if point >= len(self.values):
    		self.values.append(value)
    	else:
        	self.values[point] = value

    def set_name(self,name):
        self.name = name
    
    def set_param_type(self, param_type):
        self.param_type = param_type
    def get_name(self):
        return self.name
    
    def get_value(self, default_val=0, point=None):
    	if len(self.values) == 0:
    		return []
        elif point == None:
        	return self.values
        else:
        	return self.values[point]
    def get_values(self):
    	return
    def get_category(self):
        return self.category
    
    def get_param_type(self):
        return self.param_type

    def get_num_values(self):
    	return len(values)

    def delete_last(self):
    	del self.values[-1]


class DataModel(object):
    parameters = {
        'SYSTEM_ID': Parameter(category='satellite_var'),
        'PLANET_MASS': Parameter(category='satellite_var',param_type = 'int'),
        'ORBIT_ECCENTRICITY': Parameter(category='satellite_var'),
        'ORBIT_SEMIMAJOR_AXIS': Parameter(category='satellite_var'),
        'NSR_PERIOD': Parameter(category='satellite_var'),
        'Diurnal' : Parameter(param_type = 'bool', category='stresses_var'),
        'Nonsynchronous Rotation' : Parameter(param_type = 'bool', category='stresses_var'),
        'Obliquity' : Parameter(param_type = 'bool', category='stresses_var'),
        'periapsis_arg' : Parameter(category='stresses_var'),
        'obliquity' : Parameter(param_type = 'bool', category='stresses_var'),
        'Ice Shell Thickening' : Parameter(param_type = 'bool', category='stresses_var'),
        'delta_tc' : Parameter(category='stresses_var'),
        'diffusivity' : Parameter(category='stresses_var'),
        'Polar Wander' : Parameter(param_type = 'bool', category='stresses_var'),
        'PWthetaRi' : Parameter(category='stresses_var'),
        'PWphiRi' : Parameter(category='stresses_var'),
        'PWthetaRf' : Parameter(category='stresses_var'),
        'PWphiRf' : Parameter(category='stresses_var'),
        'PWthetaTi' : Parameter(category='stresses_var'),
        'PWthetaTf' : Parameter(category='stresses_var'),
        'PWphiTf' : Parameter(category='stresses_var'),
        'PWphiTi' : Parameter(category='stresses_var'),
        'MIN' : Parameter(category='grid_var'),
        'MAX' : Parameter(category='grid_var'),
        'NUM' : Parameter(category='grid_var'),
        'TIME_MIN' : Parameter(category='grid_var'),
        'TIME_MAX' : Parameter(category='grid_var'),
        'nsr_time' : Parameter(category='grid_var'),
        'TIME_NUM' : Parameter(category='grid_var'),
        'FINAL_LAT' : Parameter(category='grid_var'),
        'FINAL_LONG' : Parameter(category='grid_var'),
        'NSR_PERIOD_MIN' : Parameter(category='grid_var'),
        'NSR_PERIOD_MAX' : Parameter(category='grid_var'),
        'NSR_PERIOD_NUM' : Parameter(category='grid_var'),
		'POLE_POSITION_MIN' : Parameter(category='grid_var'),
        'POLE_POSITION_MAX' : Parameter(category='grid_var'),
        'POLE_POSITION_NUM' : Parameter(category='grid_var'),
        'NUM_INCREMENTS' : Parameter(category='grid_var'),
        'LAT_NUM' : Parameter(category='grid_var'),
        'LON_NUM' : Parameter(category='grid_var'),
        'LON_MIN' : Parameter(category='grid_var'),
        'LAT_MIN' : Parameter(category='grid_var'),
        'LAT_MAX' : Parameter(category='grid_var'),
        'LON_MAX' : Parameter(category='grid_var'),
        'MAX' : Parameter(category='grid_var'),
        'GRID_ID' : Parameter(category='grid_var'),
        'ORBIT_NUM' : Parameter(category='grid_var'),
        'ORBIT_MIN' : Parameter(category='grid_var'),
        'ORBIT_MAX' : Parameter(category='grid_var'),
        'LAT' : Parameter(category='grid_param'),
        'LON' : Parameter(category='grid_param'),
        'TIME' : Parameter(category='grid_param'),
        'ORBIT' : Parameter(category='grid_param'),
        'to_plot_cycloids': Parameter(param_type='bool', category='GUI'),
        'to_plot_many_cycloids': Parameter(param_type='bool',category='GUI'),
        'to_plot_lineaments': Parameter(param_type='bool',category='GUI'),
        'to_plot_principal_vectors': Parameter(param_type='bool',category='GUI'),
        'to_plot_shear_vectors': Parameter(param_type='bool',category='GUI'),
        'to_plot_longitude_vectors': Parameter(param_type='bool',category='GUI'),
        'to_plot_latitude_vectors': Parameter(param_type='bool',category='GUI'),
        'projection': Parameter(value='cyl',category='GUI'),
        'direction': Parameter(value='east',category='GUI'),
        'field': Parameter(value='tens',category='GUI'),
        'VAL': Parameter(category='GUI'),
        'phi': ParameterList(category='Points'),
        'theta': ParameterList(category='Points'),
        'orbit': ParameterList(category='Points'),
        't': ParameterList(category='Points'),
        'Tpt': ParameterList(category='Points'),
        'Spt': ParameterList(category='Points'),
        'a': ParameterList(category='Points'),
        'Tpp': ParameterList(category='Points'),
        's3': ParameterList(category='Points'),        
        'Ttt': ParameterList(category='Points'),
        's1': ParameterList(category='Points'),
        'point_rows': Parameter(category='Points'),
        'YIELD' : Parameter(category='cycloids'),
        'PROPAGATION_STRENGTH' : Parameter(category='cycloids'),
        'PROPAGATION_SPEED' : Parameter(category='cycloids'),
        'STARTING_LATITUDE' : Parameter(category='cycloids'),
        'STARTING_LONGITUDE' : Parameter(category='cycloids'),
        'STARTING_DIRECTION' : Parameter(category='cycloids'),
        'VARY_VELOCITY': Parameter(category='cycloids'),
        'k': Parameter(category='cycloids')
        }
    def __init__(self):
        

        for i in range(0,4):
            DataModel.parameters.update(
            {
            'LAYER_ID_%s' % i: Parameter(category='layer_var'),
            'DENSITY_%s' % i: Parameter(category='layer_var'),
            'TENSILE_STR_%s' % i: Parameter(category='layer_var'),
            'YOUNGS_MODULUS_%s' % i: Parameter(category='layer_var'),
            'POISSONS_RATIO_%s' % i: Parameter(category='layer_var'),
            'THICKNESS_%s' % i: Parameter(category='layer_var'),
            'VISCOSITY_%s' % i: Parameter(category='layer_var')
            })
    

        self.set_param_names()

        self.stress_d = {
        u'Nonsynchronous Rotation': NSR,
        u'Diurnal': Diurnal,
        u'Ice Shell Thickening': IST,
        u'Obliquity': DiurnalObliquity,
        u'Polar Wander': PolarWander}



        self.stresses = None
        self.satellite = None
        self.grid = None
        self.calc = None
        #self.satellite = self.make_satellite()
        #self.grid = self.make_grid()


    def calculate_stress(self):
        
        self.calc = StressCalc(self.get_stresses())
        self.satellite_changed = self.grid_changed = self.stresses_changed = False
        self.calc_changed = True
        return self.calc
        


    # updates calculations
    def get_calc(self, k=None):
        self.calculate()
        return self.calc

    # calculates tensor stresses
    def calc_tensor(self, rows=1):
        for i in range(rows):
            theta, phi, t = [ float(self.get_param_value(p, point=i)) for p in ['theta', 'phi', 't'] ]
            t *= seconds_in_year
            theta, phi = map(numpy.radians, [theta, phi])
            calc = self.get_calc()
            Ttt, Tpt, Tpp = [ "%g" % (x/1000.) for x in calc.tensor(numpy.pi/2 - theta, phi, t)]
            self.set_parameter("Ttt", Ttt, point=i)
            self.set_parameter("Tpt", Tpt, point=i) 
            self.set_parameter("Tpp", Tpp, point=i) # http://pirlwww.lpl.arizona.edu/~hoppa/science.html
            s1, a1, s3, a3 = calc.principal_components(numpy.pi/2-theta, phi, t)
            self.set_parameter("s1", "%g" % (s1/1000.), point=i) 
            self.set_parameter("s3", "%g" % (s1/1000.), point=i) 
            self.set_parameter("a", "%g" % (s1/1000.), point=i) 

    def get_stresses(self):
    	if not self.satellite:
    		self.make_satellite()
    	self.stresses = []
    	for param_name in self.stress_d.keys():
    		if self.get_param_value(param_name):
    			self.stresses.append(self.stress_d[param_name](self.satellite))

        print 'stresses:', self.stresses
        return self.stresses
    def make_satellite(self):

        sat_dict = {}
        for param in self.get_parameters_by_category('satellite_var') + self.get_parameters_by_category('layer_var'):
            sat_dict.update({param.get_name(): param.get_value()})
        self.satellite = Satellite(sat_dict)
        
    def make_grid(self):

        grid_dict = {}
        for param in self.get_parameters_by_category('grid_var')+self.get_parameters_by_category('satellite_var'):
            grid_dict.update({param.get_name(): param.get_value()})
        if not self.satellite:
            self.make_satellite()
        self.grid = satstress.gridcalc.Grid(self.satellite,grid_dict)

    def calculate(self):
        self.calc = StressCalc(self.get_stresses())
        self.calc_changed = True
        return self.calc
        
    def get_calc(self):
    	return self.calculate()
        


    def get_grid(self):
        if not self.grid:
            self.make_grid()
        return self.grid
            
    def get_satellite(self):
        if not self.satellite:
            self.make_satellite()
        return self.satellite

    def is_set(self,params):
        for param in params:
            if not param.get_value():
                return False
        return True

    def satellite_set(self):
        return self.is_set(self.get_parameters_by_category('satellite_var'))

    def grid_set(self):
        return self.is_set(self.get_parameters_by_category('grid_var'))

    def stresses_set(self):
        return self.is_set(self.get_parameters(['Diurnal']))

    def set_param_names(self):
        for p,v in DataModel.parameters.items():
            v.set_name(p)

    def get_parameter(self, param_name):
        return DataModel.parameters[param_name]

    def get_params_dict(self):
        return self.parameters

    def get_parameters(self, params_list):
        parameters = []
        for param_name in params_list:
            DataModel.parameters.append(DataModel.parameters[param_name])
        return parameters


    def get_param_value(self,param_name,default_val=None, default_param_type=str, point = -1):
    	param =DataModel.parameters[param_name]

    	if (point >=0):
    		return param.get_value(point=point)

    	if param.get_param_type() == 'bool':
    		if  param.get_value() == 'None' or param.get_value() == 'False' or param.get_value() == False:
    			return False
    		else:
    			return True
    			

    	else:
	        returnVal = default_val if not param.get_value() else param.get_value()
	        if param.get_param_type():
	            return param.get_param_type()(returnVal)
	        return default_param_type(returnVal)

    def get_parameters_by_category(self,category):
        return [parameter for param_name, parameter in DataModel.parameters.items() if parameter.get_category() == category]        


    def set_parameter(self,param_name, value, point= -1):
    	if (point >= 0):
    		DataModel.parameters[param_name].set_value(value,point)
    	else:
        	DataModel.parameters[param_name].set_value(value)




