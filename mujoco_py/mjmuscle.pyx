from libc.math cimport fabs, fmax, fmin
from mujoco_py.generated import const

cdef struct muscle_input:
    mjtNum dt_seconds  # PID sampling time.
    mjtNum length
    mjtNum velocity
    mjtNum actuator_acc0
    mjtNum activation
    mjtNum control
    mjtNum act_length_range_0
    mjtNum act_length_range_1
    mjtNum range_0
    mjtNum range_1
    mjtNum force
    mjtNum scale
    mjtNum lmin
    mjtNum lmax
    mjtNum vmax
    mjtNum fpmax
    mjtNum fvmax


cdef struct muscle_output:
    mjtNum force


cdef mjtNum c_zero_gains(const mjModel*m, const mjData*d, int id) with gil:
    '''Actuator is implemented as a bias function because it sidesteps the issue
    of a linear controller. Normally the gain output is multiplied with the 
    ctrl signal, we do this manually in the bias now.'''
    return 0.0


cdef muscle_output c_muscle_function(muscle_input parameters):
    """
    :param parameters: PID parameters
    :return: A PID output struct containing the control output and the error state
    """
    
    cdef mjtNum a = 0.5 * (parameters.lmin + 1)
    cdef mjtNum b = 0.5 * (parameters.lmax  + 1)
    cdef mjtNum c = parameters.fvmax - 1
    

    cdef mjtNum rescaled_length = rescale_length(parameters.length, parameters.range_0, parameters.range_1, parameters.act_length_range_0, parameters.act_length_range_1)
    cdef mjtNum rescaled_velocity = rescale_velocity(parameters.velocity, parameters.range_0, parameters.range_1, parameters.act_length_range_0, parameters.act_length_range_1)
    cdef mjtNum FL = bump(rescaled_length, parameters.lmin, 1, parameters.lmax) + 0.15 * bump(rescaled_length, parameters.lmin, 0.5 * (parameters.lmin + 0.95), 0.95)
    cdef mjtNum FV = active_velocity(rescaled_velocity, c, parameters.vmax, parameters.fvmax)
    cdef mjtNum PF = passive_force(rescaled_length, parameters.fpmax, b)
    cdef mjtNum peak_force = get_peak_force(parameters.force, parameters.scale, parameters.actuator_acc0)
    force = force_output(parameters.activation, FL, FV, PF, peak_force)
    return muscle_output(force=force)


cdef force_output(mjtNum activation, mjtNum FL, mjtNum FV, mjtNum PF, mjtNum peak_force):
    # TODO implement a way of just activating additional components without having to go through every if-case
    return - (FL * FV * activation + PF) * peak_force


cdef get_peak_force(mjtNum force, mjtNum scale, mjtNum actuator_acc0):
    if (force + 1) < 0.01:
        return scale / actuator_acc0
    else: 
        return force


cdef bump(mjtNum length, mjtNum A, mjtNum mid, mjtNum B):
    cdef mjtNum left = 0.5 * (A + mid)
    cdef mjtNum right = 0.5 * (mid + B)
    cdef mjtNum temp = 0

    if ((length <= A) or (length >= B)):
            return 0
    elif (length < left):
        temp = (length - A) / (left - A)
        return 0.5 * temp * temp
    elif (length < mid):
        temp = (mid - length) / (mid - left)
        return 1 - 0.5 * temp * temp
    elif (length < right):
        temp = (length - mid) / (right - mid)
        return 1 - 0.5 * temp * temp
    else:
        temp = (B - length) / (B - right)
        return 0.5 * temp * temp


cdef passive_force(mjtNum length, mjtNum fpmax, mjtNum b):
    cdef mjtNum temp = 0

    if (length <= 1):
        return  0
    elif (length <= b):
        temp = (length -1) / (b - 1)
        return 0.25 * fpmax * temp * temp * temp
    else:
        temp = (length - b) / (b - 1)
        return 0.25 * fpmax * (1 + 3 * temp)


cdef active_velocity(mjtNum velocity, mjtNum c, mjtNum vmax, mjtNum fvmax):
    cdef mjtNum eff_vel = velocity / vmax
    if (eff_vel < -1):
        return 0
    elif (eff_vel <= 0):
        return (eff_vel + 1) * (eff_vel + 1)
    elif (eff_vel <= c):
        return fvmax - (c - eff_vel) * (c - eff_vel) / c
    else:
        return fvmax


cdef rescale_length(mjtNum length, mjtNum range_0, mjtNum range_1, mjtNum act_length_range_0, mjtNum act_length_range_1):
    cdef mjtNum L0  = (act_length_range_1 - act_length_range_0) / (range_1 * (1  - (range_0/range_1)))
    cdef mjtNum LT = act_length_range_0 - range_0 * L0
    return (length - LT) / L0


cdef rescale_velocity(mjtNum velocity, mjtNum range_0, mjtNum range_1, mjtNum act_length_range_0, mjtNum act_length_range_1):
    cdef mjtNum L0  = (act_length_range_1 - act_length_range_0) / (range_1 - range_0)
    return velocity / L0


cdef enum USER_DEFINED_ACTUATOR_PARAMS_MUSCLE:
    IDX_RANGE_0 = 0,
    IDX_RANGE_1 = 1,
    IDX_FORCE = 2,
    IDX_SCALE = 3,
    IDX_LMIN = 4,
    IDX_LMAX = 5,
    IDX_VMAX= 6,
    IDX_FPMAX = 7,
    IDX_FVMAX= 8,


cdef enum USER_DEFINED_CONTROLLER_DATA_MUSCLE:
    NUM_USER_DATA_PER_ACT = 1,


cdef mjtNum c_muscle_bias(const mjModel*m, const mjData*d, int id):
    """
    To activate PID, set gainprm="Kp Ti Td iClamp errBand iSmooth" in a general type actuator in mujoco xml
    """
    cdef mjtNum dt_in_sec = m.opt.timestep
    result = c_muscle_function(parameters=muscle_input(
        dt_seconds=dt_in_sec,
        length=d.actuator_length[id],
        velocity=d.actuator_velocity[id],
        actuator_acc0=m.actuator_acc0[id],
        activation=d.act[id],
        control=d.ctrl[id],
        act_length_range_0=m.actuator_lengthrange[2 * id],
        act_length_range_1=m.actuator_lengthrange[2 * id + 1],
        range_0=m.actuator_gainprm[IDX_RANGE_0],
        range_1=m.actuator_gainprm[IDX_RANGE_1],
        force=m.actuator_gainprm[IDX_FORCE],
        scale=m.actuator_gainprm[IDX_SCALE],
        lmin=m.actuator_gainprm[IDX_LMIN],
        lmax=m.actuator_gainprm[IDX_LMAX],
        vmax=m.actuator_gainprm[IDX_VMAX],
        fpmax=m.actuator_gainprm[IDX_FPMAX],
        fvmax=m.actuator_gainprm[IDX_FVMAX]))
    #print(f' Vmax is {m.actuator_gainprm[IDX_VMAX]} for actuator')
    return result.force 


cdef enum USER_DEFINED_ACTUATOR_DATA:
    IDX_CONTROLLER_TYPE = 0
    NUM_ACTUATOR_DATA = 1


cdef mjtNum c_custom_bias(const mjModel*m, const mjData*d, int id) with gil:
    """
    Switches between PID and Cascaded PID-PI type custom bias computation based on the
    defined actuator's actuator_user field.
    user="1": Cascade PID-PI
    default: PID
    :param m: mjModel
    :param d:  mjData
    :param id: actuator ID
    :return: Custom actuator force
    """
    #controller_type = int(m.actuator_user[id * m.nuser_actuator + IDX_CONTROLLER_TYPE])
    return c_muscle_bias(m, d, id)

def set_muscle_control(m, d):
    global mjcb_act_gain
    global mjcb_act_bias

    if m.nuserdata < m.nu * NUM_USER_DATA_PER_ACT:
        raise Exception('nuserdata is not set large enough to store PID internal states.')

    if m.nuser_actuator < m.nu * NUM_ACTUATOR_DATA:
        raise Exception(f'nuser_actuator is not set large enough to store controller types. It is {m.nuser_actuator} but should be {m.nu * NUM_ACTUATOR_DATA}')

    for i in range(m.nuserdata):
        d.userdata[i] = 0.0

    mjcb_act_gain = c_zero_gains
    mjcb_act_bias = c_custom_bias
