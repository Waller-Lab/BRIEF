"""
Implement optics algorithms for optical phase tomography using GPU

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
October 22, 2018
"""

import numpy as np
import arrayfire as af
import scipy.io as sio
import contexttimer
import matplotlib.pyplot as plt
from opticaltomography import settings
from opticaltomography.opticsmodel import MultiTransmittance, MultiPhaseContrast
from opticaltomography.opticsmodel import Defocus, Aberration
from opticaltomography.opticsutil import ImageRotation, calculateNumericalGradient
from opticaltomography.regularizers import Regularizer

np_complex_datatype = settings.np_complex_datatype
np_float_datatype   = settings.np_float_datatype
af_float_datatype   = settings.af_float_datatype
af_complex_datatype = settings.af_complex_datatype

class AlgorithmConfigs:
    """
    Class created for all parameters for tomography solver
    """
    def __init__(self):
        self.method              = "FISTA"
        self.stepsize            = 1e-2
        self.max_iter            = 20
        self.error               = []
        self.reg_term            = 0.0     #L2 norm
        self.multiillu           = False    #illuminate the sample from multiple sources simultaneously, true; illuminate one by one, false
        self.groupillu           = False    #group illuminations        

        #FISTA
        self.fista_global_update = False
        self.restart             = False

        #total variation regularization
        self.total_variation     = False
        self.reg_tv              = 1.0 #lambda
        self.max_iter_tv         = 15
        self.order_tv            = 1
        self.total_variation_gpu = False
        
        #lasso
        self.lasso               = False
        self.reg_lasso           = 1.0

        #positivity constraint
        self.positivity_real     = (False, "larger")
        self.positivity_imag     = (False, "larger")
        self.pure_real           = False
        self.pure_imag           = False

        #aberration correction
        self.pupil_update        = False
        self.pupil_global_update = False
        self.pupil_step_size     = 1.0
        self.pupil_update_method = "gradient"

        #batch gradient update
        self.batch_size          = 1

        #random order update
        self.random_order        = False

class PhaseObject3D:
    """
    Class created for 3D objects.
    Depending on the scattering model, one of the following quantities will be used:
    - Refractive index (RI)
    - Transmittance function (Trans)
    - PhaseContrast
    - Scattering potential (V) 

    shape:              shape of object to be reconstructed in (x,y,z), tuple
    voxel_size:         size of each voxel in (x,y,z), tuple
    RI_obj:             refractive index of object(Optional)
    RI:                 background refractive index (Optional)
    slice_separation:   For multislice algorithms, how far apart are slices separated, array (Optional)
    """
    def __init__(self, shape, voxel_size, RI_obj = None, RI = 1.0, slice_separation = None):
        assert len(shape) == 3, "shape should be 3 dimensional!"
        self.shape           = shape
        self.RI_obj          = RI * np.ones(shape, dtype = np_complex_datatype) if RI_obj is None else RI_obj.astype(np_complex_datatype)
        self.RI              = RI
        self.pixel_size      = voxel_size[0]
        self.pixel_size_z    = voxel_size[2]

        if slice_separation is not None:
            #for discontinuous slices
            assert len(slice_separation) == shape[2]-1, "number of separations should match with number of layers!"
            self.slice_separation = np.asarray(slice_separation).astype(np_float_datatype)
        else:
            #for continuous slices
            self.slice_separation = self.pixel_size_z * np.ones((shape[2]-1,), dtype = np_float_datatype)

    def convertRItoTrans(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.trans_obj       = np.exp(1.0j*k0*(self.RI_obj - self.RI)*self.pixel_size_z)

    def convertRItoPhaseContrast(self):
        self.contrast_obj    = self.RI_obj - self.RI 

    def convertRItoV(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        self.V_obj           = k0**2 * (self.RI**2 - self.RI_obj**2)

    def convertVtoRI(self, wavelength):
        k0                   = 2.0 * np.pi / wavelength
        B                    = -1.0 * (self.RI**2 - self.V_obj.real/k0**2)
        C                    = -1.0 * (-1.0 * self.V_obj.imag/k0**2/2.0)**2
        RI_obj_real          = ((-1.0 * B + (B**2-4.0*C)**0.5)/2.0)**0.5
        RI_obj_imag          = -0.5 * self.V_obj.imag/k0**2/RI_obj_real
        self.RI_obj          = RI_obj_real + 1.0j * RI_obj_imag

class TomographySolver:
    """
    Highest level solver object for tomography problem

    phase_obj_3d:               phase_obj_3d object defined from class PhaseObject3D
    fx_illu_list:               illumination coordinate in x, default = [0] (on axis)
    fy_illu_list:               illumination coordinate in y
    fz_illu_list:               illumination coordinate in z
    rotation_angle_list:        angles of rotation in tomogrpahy
    propagation_distance_list:  defocus distances for each illumination
    """
    def __init__(self, phase_obj_3d, fx_illu_list = [0], fy_illu_list = [0], fz_illu_list = [0], rotation_angle_list = [0], propagation_distance_list = [0], **kwargs):
        self.phase_obj_3d    = phase_obj_3d
        self.wavelength      = kwargs["wavelength"]
        #Rotation angels and objects
        self.rot_angles      = rotation_angle_list
        self.number_rot      = len(self.rot_angles)
        self.rotation_pad    = kwargs.get("rotation_pad", True)

        #Illumination source coordinates
        assert len(fx_illu_list) == len(fy_illu_list)
        self.fx_illu_list    = fx_illu_list
        self.fy_illu_list    = fy_illu_list
        self.fz_illu_list    = fz_illu_list
        self.number_illum    = len(self.fx_illu_list)

        #Aberation object
        self._aberration_obj = Aberration(phase_obj_3d.shape[:2], phase_obj_3d.pixel_size, self.wavelength, kwargs["na"], pad = False)

        #Defocus distances and object
        self.prop_distances  = propagation_distance_list
        self._defocus_obj    = Defocus(phase_obj_3d.shape[:2], phase_obj_3d.pixel_size, **kwargs)
        self.number_defocus  = len(self.prop_distances)

        #Scattering models and algorithms
        self._opticsmodel    = {"MultiTrans":                  MultiTransmittance,
                                "MultiPhaseContrast":          MultiPhaseContrast,
                                }
        self._algorithms     = {"GradientDescent":    self._solveFirstOrderGradient,
                                "FISTA":              self._solveFirstOrderGradient
                               }
        self.scat_model_args = kwargs
        
    def setScatteringMethod(self, model = "MultiTrans"):
        """
        Define scattering method for tomography

        model: scattering models, it can be one of the followings:
               "MultiTrans", "MultiPhaseContrast"(Used in the paper)
        """
        self.scat_model      = model
        if hasattr(self, '_scattering_obj'):
            del self._scattering_obj

        if model == "MultiTrans":
            self.phase_obj_3d.convertRItoTrans(self.wavelength)
            self.phase_obj_3d.convertRItoV(self.wavelength)
            self._x          = self.phase_obj_3d.trans_obj
            if np.any(self.rot_angles != [0]):
                self._rot_obj    = ImageRotation(self.phase_obj_3d.shape, axis=0, pad = self.rotation_pad, pad_value = 1, \
                                                 flag_gpu_inout = True, flag_inplace = True)
        elif model == "MultiPhaseContrast":
            if not hasattr(self.phase_obj_3d, 'contrast_obj'):
                self.phase_obj_3d.convertRItoPhaseContrast()
            self._x          = self.phase_obj_3d.contrast_obj   #this line is the only line excuted in this moment.
            if np.any(self.rot_angles != [0]):
                self._rot_obj    = ImageRotation(self.phase_obj_3d.shape, axis=0, pad = self.rotation_pad, pad_value = 0, \
                                                 flag_gpu_inout = True, flag_inplace = True)
        else:
            if not hasattr(self.phase_obj_3d, 'V_obj'):
                self.phase_obj_3d.convertRItoV(self.wavelength)
            self._x          = self.phase_obj_3d.V_obj
            if np.any(self.rot_angles != [0]):
                self._rot_obj    = ImageRotation(self.phase_obj_3d.shape, axis=0, pad = self.rotation_pad, pad_value = 0, \
                                                 flag_gpu_inout = True, flag_inplace = True)
        self._scattering_obj = self._opticsmodel[model](self.phase_obj_3d, **self.scat_model_args)

    def forwardPredict(self, field = False):
        """
        Uses current object in the phase_obj_3d to predict the amplitude of the exit wave
        Before calling, make sure correct object is contained
        """
        obj_gpu              = af.to_array(self._x)
        with contexttimer.Timer() as timer:
            forward_scattered_predict= []
            if self._scattering_obj.back_scatter:#no back scattering
                back_scattered_predict = []
            for rot_idx in range(self.number_rot):#no roration. range(self.number_rot)=0
                forward_scattered_predict.append([])
                if self._scattering_obj.back_scatter:
                    back_scattered_predict.append([])
                if self.rot_angles[rot_idx] != 0:
                    self._rot_obj.rotate(obj_gpu, self.rot_angles[rot_idx])
                for illu_idx in range(self.number_illum):
                    fx_illu       = self.fx_illu_list[illu_idx]#every illumination source
                    fy_illu       = self.fy_illu_list[illu_idx]
                    fz_illu_layer = self.fz_illu_list[illu_idx]
                    fields = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj = obj_gpu)

                    if field:
                        forward_scattered_predict[rot_idx].append(np.array(fields["forward_scattered_field"]))
                        if self._scattering_obj.back_scatter:
                            back_scattered_predict[rot_idx].append(np.array(fields["back_scattered_field"]))
                    else:
                        #execute this case, get only absolute value of fields['forward_scattered_field']
                        forward_scattered_predict[rot_idx].append(np.abs(fields["forward_scattered_field"]))
                        if self._scattering_obj.back_scatter:
                            back_scattered_predict[rot_idx].append(np.abs(fields["back_scattered_field"]))
                if self.rot_angles[rot_idx] != 0:
                    self._rot_obj.rotate(obj_gpu, -1.0*self.rot_angles[rot_idx])                        
        
        if len(forward_scattered_predict[0][0].shape)==2:
            forward_scattered_predict = np.array(forward_scattered_predict).transpose(2, 3, 1, 0)
            #Not working-----------------------------------------------------------------------------------
#             field_temp  = af.to_array(np.array(forward_scattered_predict).transpose(2, 3, 1, 0))
#             for a in range(6):
#                 field_temp = field_temp + af.shift(field_temp, a+1, 0, 0) +\
#                           af.shift(field_temp, -1*(a+1), 0, 0) +\
#                           af.shift(field_temp, 0, (a+1), 0) +\
#                           af.shift(field_temp, 0, -1*(a+1), 0)
#             forward_scattered_predict = field_temp / 30
#-----------------------------------------------------------------------------------------------------------
        elif len(forward_scattered_predict[0][0].shape)==3:
            forward_scattered_predict = np.array(forward_scattered_predict).transpose(2, 3, 4, 1, 0)
            #forward_scattered_predict=forward_scattered_predict #seems no difference compared to transpose one
        if self._scattering_obj.back_scatter:
            if len(back_scattered_predict[0][0].shape)==2:
                back_scattered_predict = np.array(back_scattered_predict).transpose(2, 3, 1, 0)
            elif len(back_scattered_predict[0][0].shape)==3:
                back_scattered_predict = np.array(back_scattered_predict).transpose(2, 3, 4, 1, 0)
            return forward_scattered_predict, back_scattered_predict,fields
        else:
            return forward_scattered_predict,fields
    
    def checkGradient(self, delta = 1e-4):
        """
        check if the numerical gradient is similar to the analytical gradient. Only works for 64 bit data type.
        """
        assert af_float_datatype == af.Dtype.f64, "This will only be accurate if 64 bit datatype is used!"
        shape     = self.phase_obj_3d.shape
        point     = (np.random.randint(shape[0]), np.random.randint(shape[1]), np.random.randint(shape[2])) 
        illu_idx  = np.random.randint(len(self.fx_illu_list))
        fx_illu   = self.fx_illu_list[illu_idx]
        fy_illu   = self.fy_illu_list[illu_idx]
        x         = np.ones(shape, dtype = np_complex_datatype)
        if self._defocus_obj.pad:
            amplitude = af.randu(shape[0]//2, shape[1]//2, dtype = af_float_datatype)
        else:
            amplitude = af.randu(shape[0], shape[1], dtype = af_float_datatype)
        print("testing the gradient at point : ", point)

        def func(x0):
            fields              = self._scattering_obj.forward(x0, fx_illu, fy_illu)
            field_scattered     = self._aberration_obj.forward(fields["forward_scattered_field"])
            field_measure       = self._defocus_obj.forward(field_scattered, self.prop_distances)
            residual            = af.abs(field_measure) - amplitude
            function_value      = af.sum(residual*af.conjg(residual)).real
            return function_value

        numerical_gradient      = calculateNumericalGradient(func, x, point, delta = delta)

        fields                  = self._scattering_obj.forward(x, fx_illu, fy_illu)
        forward_scattered_field = fields["forward_scattered_field"]
        cache                   = fields["cache"]
        forward_scattered_field = self._aberration_obj.forward(forward_scattered_field)
        field_measure           = self._defocus_obj.forward(forward_scattered_field, self.prop_distances)
        analytical_gradient     = self._computeGradient(field_measure, amplitude, cache)[point]

        print("numerical gradient:  %5.5e + %5.5e j" %(numerical_gradient.real, numerical_gradient.imag))
        print("analytical gradient: %5.5e + %5.5e j" %(analytical_gradient.real, analytical_gradient.imag))

    def _forwardMeasure(self, fx_illu, fy_illu, fz_illu_layer, obj = None):
    #def forwardMeasure(self, fx_illu, fy_illu, obj = None):
        """
        From an illumination position, this function computes the exit wave.
        fx_illu, fy_illu, fz_illu:       illumination coordinates in x, y and z (scalars)
        obj:                    object to be passed through (Optional, default pick from phase_obj_3d)
        """
        if obj is None:
            #not this case
            fields = self._scattering_obj.forward(self._x, fx_illu, fy_illu, fz_illu_layer) #self._x = self.phase_obj_3d.contrast_obj=self.RI_obj - self.RI
        else:
            #execute this case, fields is a dictionary contains a complex 2D array "field" and a tuple "cache"
            fields = self._scattering_obj.forward(obj, fx_illu, fy_illu, fz_illu_layer) 
       
        field_scattered                   = self._aberration_obj.forward(fields["forward_scattered_field"])
        field_scattered                   = self._defocus_obj.forward(field_scattered, self.prop_distances)
        # Yi modified---------------------------------------------------------------
#         fieldPhase = af.exp(1.0j*af.arg(field_scattered))
#         field_temp  = af.to_array(np.abs(field_scattered))
#         for a in range(5):
#             field_temp = field_temp + af.shift(field_temp, a+1, 0) +\
#                       af.shift(field_temp, -1*(a+1), 0) +\
#                       af.shift(field_temp, 0, (a+1)) +\
#                       af.shift(field_temp, 0, -1*(a+1))
#         fields["forward_scattered_field"] = field_temp / 25 * fieldPhase
#         #change cache as well
#         phasecontrast_obj_af, field_layer_conj_or_grad, flag_gpu_inout = fields["cache"]
#         fieldPhase_conjg = af.exp(1.0j*af.arg(field_layer_conj_or_grad))
#         fieldAmp_conjg = af.to_array(np.abs(field_layer_conj_or_grad))
#         for a in range(5):
#             fieldAmp_conjg = fieldAmp_conjg + af.shift(fieldAmp_conjg, a+1, 0, 0) +\
#                       af.shift(fieldAmp_conjg, -1*(a+1), 0, 0) +\
#                       af.shift(fieldAmp_conjg, 0, (a+1), 0) +\
#                       af.shift(fieldAmp_conjg, 0, -1*(a+1), 0)
#         field_layer_conj_or_grad = fieldAmp_conjg / 25 * fieldPhase_conjg
#         fields["cache"] = (phasecontrast_obj_af, field_layer_conj_or_grad, flag_gpu_inout)
        #----------------------------------------------------------------------
#         original code:
        fields["forward_scattered_field"] = field_scattered
       
        if self._scattering_obj.back_scatter:
            field_scattered                   = self._aberration_obj.forward(fields["back_scattered_field"])
            field_scattered                   = self._defocus_obj.forward(field_scattered, self.prop_distances)
            fields["back_scattered_field"]    = field_scattered
        return fields

    def _computeGradient(self, configs, field_measure, amplitude, cache, field_bp_init=None):
        """
        Error backpropagation to return a gradient
        field_measure:  exit wave computed in forward model
        amplitude:      amplitude measured
        cache:          exit wave at each layer, saved previously
        """
        if configs.multiillu or configs.groupillu or configs.spot:
            field_bp  = ((field_bp_init)**(-1) - amplitude)*field_measure*field_bp_init
        else:
#             field_bp  = field_measure - amplitude*af.exp(1.0j*af.arg(field_measure))
            field_bp = 2* field_measure * (af.abs(field_measure * af.conjg(field_measure))-amplitude) 
        
        field_bp  = self._defocus_obj.adjoint(field_bp, self.prop_distances)
        field_bp  = self._aberration_obj.adjoint(field_bp)
        gradient  = self._scattering_obj.adjoint(field_bp, cache)
        return gradient["gradient"]

    def _initialization(self,configs, x_init = None):
        """
        Initialize algorithm
        configs:         configs object from class AlgorithmConfigs
        x_init:          initial guess of object
        """
        if x_init is None:
            if self.scat_model is "MultiTrans":
                self._x[:, :, :] = 1.0
            else:
                self._x[:, :, :] = 0.0
        else:
            self._x[:, :, :] = x_init

    def _solveFirstOrderGradient(self, configs, amplitudes, verbose):
        """
        MAIN part of the solver, runs the FISTA algorithm
        configs:        configs object from class AlgorithmConfigs
        amplitudes:     all measurements
        verbose:        boolean variable to print verbosely
        """
        flag_FISTA    = False
        if configs.method == "FISTA":
            flag_FISTA = True
 
        #-------------------------------------------------------------------------------------------------   
        # update multiple angles at a time
        batch_update = False
#         if configs.fista_global_update or configs.batch_size != 1:
        if configs.fista_global_update or configs.batch_size > 1:
            gradient_batch    = af.constant(0.0, self.phase_obj_3d.shape[0],\
                                                 self.phase_obj_3d.shape[1],\
                                                 self.phase_obj_3d.shape[2], dtype = af_complex_datatype)
            batch_update = True
            if configs.fista_global_update:
                configs.batch_size = 0

        #TODO: what if num_batch is not an integer
        if configs.batch_size == 0:
            num_batch = 1
        else:
            if self.number_rot < 2:
                num_batch = self.number_illum // configs.batch_size
            else:
                num_batch = self.number_rot // configs.batch_size
        #-----------------------------------------------------------------------------------------------
        stepsize      = configs.stepsize
        max_iter      = configs.max_iter
        reg_term      = configs.reg_term
        configs.error = []        
        obj_gpu       = af.constant(0.0, self.phase_obj_3d.shape[0],\
                                         self.phase_obj_3d.shape[1],\
                                         self.phase_obj_3d.shape[2], dtype = af_complex_datatype)
        field_measure_total = af.constant(0.0, amplitudes.shape[0],amplitudes.shape[1],self.number_illum, dtype = af_complex_datatype)
        
        #field_measure_total = np.zeros((amplitudes.shape[0],amplitudes.shape[1],len(configs.group)), dtype = complex )
        #Initialization for FISTA update
        if flag_FISTA:
            restart       = configs.restart
            y_k           = self._x.copy() 
            t_k           = 1.0

        #Start of iterative algorithm
        with contexttimer.Timer() as timer:
            if verbose:
                print("---- Start of the %5s algorithm ----" %(self.scat_model))
            for iteration in range(max_iter):#--------------------------------------------------------
                cost                   = 0.0
                obj_gpu[:]             = af.to_array(self._x)

                if configs.random_order:
                    rot_order  = np.random.permutation(range(self.number_rot))
                    illu_order = np.random.permutation(range(self.number_illum))
                else:
                    rot_order  = range(self.number_rot)
                    illu_order = range(self.number_illum)

                for batch_idx in range(num_batch):#num_batch=number_illum//batchsize=2
                    if batch_update:
                        gradient_batch[:,:,:] = 0.0

                    if configs.batch_size == 0:
                        rot_indices  = rot_order
                        illu_indices = illu_order
                    else:
                        if self.number_rot < 2:#number_rot=1
                            rot_indices = rot_order
                            illu_indices = illu_order[batch_idx * configs.batch_size : (batch_idx+1) * configs.batch_size]
                        else:
                            illu_indices = illu_order
                            rot_indices = rot_order[batch_idx * configs.batch_size : (batch_idx+1) * configs.batch_size]    
                    for rot_idx in rot_indices:
                        # Rotate the object
                        if self.rot_angles[rot_idx] != 0:
                            self._rot_obj.rotate(obj_gpu, self.rot_angles[rot_idx])
                            if batch_update:
                                self._rot_obj.rotate(gradient_batch, self.rot_angles[rot_idx])
                        
                        #for illu_idx in illu_indices: #illu_indices=(illu_idx, illu_idx+1) 
                        for illu_idx in range(self.number_illum):
                            #forward scattering
                            fx_illu                       = self.fx_illu_list[illu_idx]
                            fy_illu                       = self.fy_illu_list[illu_idx]
                            fz_illu_layer                 = self.fz_illu_list[illu_idx]
                            fields                        = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj = obj_gpu)
                            field_measure                 = fields["forward_scattered_field"]
                            cache                         = fields["cache"]
                            #calculate error
                            if configs.multiillu:
                                amplitude                           = af.to_array(amplitudes[:,:,:,:, rot_idx])
                                field_measure_total[:, :, illu_idx] = field_measure 
                            elif configs.groupillu:
                                field_measure_total[:, :, illu_idx] = field_measure
                            elif configs.spot:
                                field_measure_total[:, :, illu_idx] = field_measure  
                            else:
                                amplitude       = af.to_array(amplitudes[:,:,:,illu_idx, rot_idx])
#                                 residual        = af.abs(field_measure)-amplitude
                                residual        = af.abs(field_measure * af.conjg(field_measure)) - amplitude
                                cost           += af.sum(residual*af.conjg(residual)).real
                                if batch_update:
                                    gradient_batch[:, :, :]  += self._computeGradient(configs, field_measure, amplitude, cache)
                                else:
                                    #execute this case
                                    obj_gpu[:, :, :]   -= stepsize * self._computeGradient(configs, field_measure, amplitude, cache)
                                    
                            amplitude       = None
                            field_measure   = None
                            cache           = None
                        
                        if configs.spot: 
                            for illu_idx in range(self.number_illum):
                                amplitude       = af.to_array(amplitudes[:,:,:,illu_idx, rot_idx])
                                field_temp  = af.to_array(af.abs(field_measure_total[:, :, illu_idx]))
                                for a in range(configs.spotpixel):
                                    field_temp = field_temp + af.shift(field_temp, a+1, 0) +\
                                              af.shift(field_temp, -1*(a+1), 0) +\
                                              af.shift(field_temp, 0, (a+1)) +\
                                              af.shift(field_temp, 0, -1*(a+1))
                                field_bp_inits  = field_temp / (5 * configs.spotpixel)
                                field_bp_inits  = np.array(field_bp_inits)
                                field_bp_inits  = field_bp_inits[:,:, np.newaxis]
                                field_bp_inits  = field_bp_inits[:,:,:, np.newaxis]
                                field_bp_inits  = field_bp_inits[:,:,:,:, np.newaxis]
                                field_bp_init   = af.to_array(field_bp_inits[:,:,:,:, 0])
                                #forward scattering
                                fx_illu                       = self.fx_illu_list[illu_idx]
                                fy_illu                       = self.fy_illu_list[illu_idx]
                                fz_illu_layer                 = self.fz_illu_list[illu_idx]
                                fields                        = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj = obj_gpu)
                                field_measure                 = fields["forward_scattered_field"]
                                cache                         = fields["cache"]

                                #calculate gradient
                                if batch_update:
                                    gradient_batch[:, :, :]  += self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init)
                                else:
                                    #execute this case
                                    obj_gpu[:, :, :]   -= stepsize * self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init) 
                           
                                field_measure_total[:, :, illu_idx] = field_measure
                                residual        = field_temp / (5 * configs.spotpixel) - amplitude
                                cost           += af.sum(residual*af.conjg(residual)).real
                                
                                field_measure   = None
                                cache           = None
                                amplitude       = None
                                
                        if configs.multiillu:
                            amplitude       = af.to_array(amplitudes[:,:,:,:, rot_idx])
                            for illu_idx in range(self.number_illum):
                                field_bp_inits  = np.array(af.sum(af.abs(field_measure_total*af.conjg(field_measure_total)), 2)**(-0.5))
                                field_bp_inits  = field_bp_inits[:,:, np.newaxis]
                                field_bp_inits  = field_bp_inits[:,:,:, np.newaxis]
                                field_bp_inits  = field_bp_inits[:,:,:,:, np.newaxis]
                                field_bp_init   = af.to_array(field_bp_inits[:,:,:,:, 0])

                                #forward scattering
                                fx_illu                       = self.fx_illu_list[illu_idx]
                                fy_illu                       = self.fy_illu_list[illu_idx]
                                fz_illu_layer                 = self.fz_illu_list[illu_idx]
                                fields                        = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj = obj_gpu)
                                field_measure                 = fields["forward_scattered_field"]
                                cache                         = fields["cache"]

                                #calculate gradient
                                if batch_update:
                                    gradient_batch[:, :, :]  += self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init)
                                else:
                                    obj_gpu[:, :, :]   -= stepsize * self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init)
                                #updates field_bp_init with new layers
                                field_measure_total[:, :, illu_idx] = field_measure
                                residual        = (af.sum(af.abs(field_measure_total*af.conjg(field_measure_total)), 2))**(0.5) - amplitude
                                cost           += af.sum(residual*af.conjg(residual)).real

                                field_measure   = None
                                cache           = None
                        
                        if configs.groupillu:
                            for illu_group_idx in range(len(configs.group)):
                                current_field_list = configs.group[illu_group_idx]
                                sub_fields         = field_measure_total[:, :, af.to_array(current_field_list)]
                                amplitude          = af.to_array(amplitudes[:, :, illu_group_idx, :, rot_idx])

                                for sub_idx in range(len(current_field_list)):
                                    #calculate residue and cost of this group
                                    field_num       = current_field_list[sub_idx]
                                    field_bp_inits  = np.array(af.sum(af.abs(sub_fields*af.conjg(sub_fields)), 2)**(-0.5))
                                    field_bp_inits  = field_bp_inits[:,:, np.newaxis]
                                    field_bp_inits  = field_bp_inits[:,:,:, np.newaxis]
                                    field_bp_inits  = field_bp_inits[:,:,:,:, np.newaxis]
                                    field_bp_init   = af.to_array(field_bp_inits[:,:,:,:, 0])
                                        
                                    #forward scattering
                                    fx_illu                       = self.fx_illu_list[field_num]
                                    fy_illu                       = self.fy_illu_list[field_num]
                                    fz_illu_layer                 = self.fz_illu_list[field_num]
                                    fields                        = self._forwardMeasure(fx_illu, fy_illu, fz_illu_layer, obj = obj_gpu)
                                    field_measure                 = fields["forward_scattered_field"]
                                    cache                         = fields["cache"]
                                    
                                    #calculate gradient
                                    if batch_update:
                                        gradient_batch[:, :, :]  += self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init)
                                    else:
                                        obj_gpu[:, :, :]  -= stepsize * self._computeGradient(configs, field_measure, amplitude, cache, field_bp_init)
                                    #updates field_bp_init with new layers
                                    field_measure_total[:, :, field_num] = field_measure
                                    sub_fields[:, :, sub_idx]            = field_measure
                                    field_measure   = None
                                    cache           = None
                                residual     = (af.sum(af.abs(sub_fields*af.conjg(sub_fields)), 2))**(0.5) - amplitude
                                cost        += af.sum(residual*af.conjg(residual)).real    
                                amplitude    = None
                                
                #------------------------------------------------------------------------------------------------------------------------   

                    if self.rot_angles[rot_idx] != 0:
                            self._rot_obj.rotate(obj_gpu, -1.0*self.rot_angles[rot_idx])
                            if batch_update:
                                self._rot_obj.rotate_adj(gradient_batch, self.rot_angles[rot_idx])
                    if batch_update:
                        obj_gpu[:, :, :] -= stepsize * gradient_batch

                if np.isnan(obj_gpu).sum() > 0:
                    stepsize     *= 0.5 
                    print("WARNING: Gradient update diverges! Resetting stepsize to %3.2f" %(stepsize))
                    return obj_gpu
#-----------------------------------------------------------------------------------------------------------------------------------------
                # L2 regularizer
                obj_gpu[:, :, :] -= stepsize * reg_term * obj_gpu
                

                #record total error   
                configs.error.append(cost + reg_term * af.sum(obj_gpu*af.conjg(obj_gpu)).real)  
#                 configs.error.append(cost)
                if flag_FISTA:
                    #check convergence
                    if iteration > 0:
                        if configs.error[-1] > configs.error[-2]:
                            if restart:
                                t_k              = 1.0
                                self._x[:, :, :] = y_k
                                print("WARNING: FISTA Restart! Error: %5.5f" %(np.log10(configs.error[-1])))
                                continue
                            else:
                                print("WARNING: Error increased! Error: %5.5f" %(np.log10(configs.error[-1])))                     

                    #FISTA auxiliary variable
                    y_k1 = np.array(self._regularizer_obj.applyRegularizer(obj_gpu))
                    
                    #FISTA update
                    t_k1                 = 0.5*(1.0 + (1.0 + 4.0*t_k**2)**0.5)
                    beta                 = (t_k - 1.0) / t_k1
                    self._x[:, :, :]     = y_k1 + beta * (y_k1 - y_k)
                    t_k                  = t_k1
                    y_k                  = y_k1.copy()
                else:
                    #check convergence
                    self._x[:, :, :]  = np.array(obj_gpu)
                    if iteration > 0:
                        if configs.error[-1] > configs.error[-2]:
                            print("WARNING: Error increased! Error: %5.5f" %(np.log10(configs.error[-1]))) 
                            stepsize     *= 0.8
                if verbose:
                    print("iteration: %d/%d, error: %5.5f, elapsed time: %5.2f seconds" %(iteration+1, max_iter, np.log10(configs.error[-1]), timer.elapsed))
                if iteration % 10 == 0:
                    np.save("reconstructx",self._x)
                    np.save("cost",configs.error)

        return self._x

    def solve(self, configs, amplitudes, x_init = None, verbose = True):
        """
        function to solve for the tomography problem

        configs:        configs object from class AlgorithmConfigs
        amplitudes:     measurements in amplitude not INTENSITY, ordered by (x,y,illumination,defocus,rotation)
        x_init:         initial guess for object
        verbose:        boolean variable to print verbosely
        """
        self._initialization(configs, x_init) #after this line, solver_obj._x=phantom is cleared up. solver_obj._x=0
        #self._aberration_obj = Aberration(phase_obj_3d.shape[:2], phase_obj_3d.pixel_size, self.wavelength, kwargs["na"], pad = False)
        self._aberration_obj.setUpdateParams(flag_update = configs.pupil_update, pupil_step_size = configs.pupil_step_size, update_method = configs.pupil_update_method, global_update = configs.pupil_global_update, measurement_num = self.number_illum*self.number_rot)
        
        self._regularizer_obj = Regularizer(configs, verbose)#Regularizer.py
        
        if configs.multiillu:
            amplitudes = (np.sum(amplitudes*np.conj(amplitudes),2))**0.5
            if self.number_defocus < 2:
                amplitudes = amplitudes[:,:, np.newaxis]   
            amplitudes = amplitudes[:,:,:, np.newaxis]
        elif configs.groupillu:
            amplitude_group = np.zeros((amplitudes.shape[0],amplitudes.shape[1],len(configs.group)), dtype = float )
            for illu_group_idx in range(len(configs.group)):
                current_field_list = configs.group[illu_group_idx]
                amp = amplitudes[:, :, current_field_list]
                amplitude_group[:,:,illu_group_idx] = (np.sum(amp*np.conj(amp),axis=2))**0.5
            amplitudes = amplitude_group[:,:,:, np.newaxis]
        elif configs.spot:
            amplitudes_temp  = af.to_array(amplitudes)
            for a in range(configs.spotpixel):
                amplitudes_temp = amplitudes_temp + af.shift(amplitudes_temp, a+1, 0, 0) +\
                          af.shift(amplitudes_temp, -1*(a+1), 0, 0) +\
                          af.shift(amplitudes_temp, 0, (a+1), 0) +\
                          af.shift(amplitudes_temp, 0, -1*(a+1), 0)
            amplitudes        = amplitudes_temp / (5 * configs.spotpixel)
            amplitudes        = np.array(amplitudes)
            if self.number_defocus < 2:
                amplitudes = amplitudes[:,:, np.newaxis]
            if self.number_illum < 2:
                amplitudes = amplitudes[:,:,:, np.newaxis]
        else:   
            if self.number_defocus < 2:
                amplitudes = amplitudes[:,:, np.newaxis]
            if self.number_illum < 2:
                amplitudes = amplitudes[:,:,:, np.newaxis]
        
        if self.number_rot < 2:
            amplitudes = amplitudes[:,:,:,:, np.newaxis]
        return self._algorithms[configs.method](configs, amplitudes, verbose)
    
    def flatPSF(self, psf, ds, x_0 = None, roi = False):
        """
        calculate H matrix in the optimization, flatten PSF
        create by Yi
        psf:            3D psf calculated from fieldPredict
        ds:             downsample rate along one dimention
        amplitudes:     measurements in amplitude not INTENSITY, ordered by (x,y,illumination,defocus,rotation)
        x_0:            mask for roi
        roi:            select the region of interest from the input amplitude image
        
        """
        #crop the raw PSF to mimic shift
        Nx = int(psf.shape[0]/2)
        Ny = int(psf.shape[1]/2)
        Nz = psf.shape[2]
        if roi == False:
            PSF = np.zeros([Nx*Ny, int(Nz * Nx/ds * Ny /ds)])
            for iy in range(0, Ny, ds):
                for ix in range(0, Nx, ds):
                    temp = np.reshape(psf[ix:ix+Nx, iy:iy+Ny, :], (Nx * Ny, Nz))
                    m = int((ix/ds * Ny/ds + iy/ds) * Nz)
                    PSF[:, m:m+Nz] = temp
        else:
            PSF = np.zeros([Nx*Ny, np.count_nonzero(x_0)])
            m = 0
            for ix in range(0, Nx):
                for iy in range(0, Ny):
                    for iz in range(0, Nz):
                        if x_0[ix, iy, iz] == 1:
                            temp       = np.reshape(psf[Nx-ix:2*Nx-ix, Ny-iy:2*Ny-iy, iz], (Nx * Ny))
                            PSF[:, m] = temp
                            m         += 1
            
        return PSF
    
    def deconv3D(self, configs, PSF, amplitudes, x_init, verbose = True):
        """
        function to find the position of light sources in 3D from a 2D image, known 3D PSF
        create by Yi
        configs:        configs object from class AlgorithmConfigs
        PSF:            2D flat PSF, H matrix
        amplitudes:     measurements in amplitude not INTENSITY, ordered by (x,y,illumination,defocus,rotation)
        x_init:         initial guess for object
        verbose:        boolean variable to print verbosely
        """
        xk = x_init.copy()
        tk = 1
        Uk = xk.copy()
#         Nx = amplitudes.shape[0]
#         Ny = amplitudes.shape[1]
#         Nz = 20
#         amp= amplitudes[int(Nx/4):int(Nx*0.75), int(Ny/4):int(Ny*0.75), :]
        amp = amplitudes
        Nx = amp.shape[0]*2
        Ny = amp.shape[1]*2
        Nz = 20
#         Yc = np.reshape(np.sum(amp,2), (PSF.shape[0], 1))
        Yc = np.reshape(amp, (PSF.shape[0], 1))
        stepsize      = configs.stepsize
        max_iter      = configs.max_iter
        reg_tv        = configs.reg_tv
        thres         = configs.thres
        L2reg         = configs. L2reg
        configs.error = []
        
        with contexttimer.Timer() as timer:
            for iteration in range(max_iter):
                residue = np.matmul(PSF, Uk) - Yc
                gradf   = np.matmul(np.conj(PSF.T), residue)
                if iteration > 3000:
                    Uk1     = Uk - 2**(iteration/650) * stepsize * gradf
                else:
                    Uk1     = Uk - stepsize*gradf
                    
                if configs.reg == True:
                    Utemp=np.reshape(Uk1, (int((PSF.shape[1]/Nz)**0.5),int((PSF.shape[1]/Nz)**0.5),Nz))
                    K=Utemp*6-np.roll(Utemp, 1, axis=0) - np.roll(Utemp, -1, axis=0) - \
                    np.roll(Utemp, 1, axis=1) - np.roll(Utemp, -1, axis=1) - \
                    np.roll(Utemp, 1, axis=2)-np.roll(Utemp, -1, axis=2)
                    K=np.reshape(K, (PSF.shape[1],1))
                    #soft thresholding
    #                 thres          = np.min(Uk1) * 1.2
                    temp           = np.abs(Uk1 - reg_tv * K) - thres
                    temp[temp < 0] = 0
                    xk1            = np.sign(Uk1 - reg_tv * K) * temp
#                 xk1            = (xk1-np.min(xk1))/np.max(xk1-np.min(xk1))
                else:
                    temp           = np.abs(Uk1) - thres
                    temp[temp < 0] = 0
                    xk1            = np.sign(Uk1) * temp
#                     xk1[xk1 >= 0.5 * np.max(xk1)] = np.max(xk1)
#                     xk1[xk1 < 0.5 * np.max(xk1)] = np.min(xk1)
                    xk1[xk1 < 0]   = 0
                # FISTA update
                tk1  = 1 + ((1 + 4 * tk**2)**0.5) / 2
                Uk   = xk1 + (tk - 1) / tk1 * (xk1 - xk)
                xk   = xk1.copy()
#                 cost = (np.sum(np.abs(np.matmul(PSF, xk) - Yc)**2))**0.5 + thres / stepsize * np.sum(np.abs(xk))
                cost = (np.sum(np.abs(np.matmul(PSF, xk) - Yc)**2))**0.5 + L2reg * np.sum(np.abs(xk))
                configs.error.append(cost)
                if iteration > 0:
                    if configs.error[-1] >= configs.error[-2]:
                        print("WARNING: Error increased!")
                        break
                    else:
                        if verbose == True:
                            print("iteration: %d/%d, error: %5.5f, elapsed time: %5.2f seconds" %(iteration+1, max_iter, np.log10(configs.error[-1]), timer.elapsed))
                        if iteration % 100 == 0:
                            print("iteration: %d/%d, error: %5.5f, elapsed time: %5.2f seconds" %(iteration+1, max_iter, np.log10(configs.error[-1]), timer.elapsed))
                        if iteration % 1000 == 0:
                            np.save("findlocationx",xk)
        
        return xk
                    