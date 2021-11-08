from libc.math cimport fabs, fmax, fmin
from mujoco_py.generated import const

cdef struct muscle_input:
    mjtNum dt_seconds  # PID sampling time.
    mjtNum length
    mjtNum velocity
    mjtNum actuator_acc0
    mjtNum activation
    mjtNum control
    mjtNum actuator_length_range_0
    mjtNum actuator_length_range_1
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
    cdef mjtNum force = 1 * parameters.control+ 0
    force = bump(parameters.length, parameters.velocity, parameters.length, parameters.length)
    return muscle_output(force=force)

cdef bump(mjtNum length, mjtNum A, mjtNum mid, mjtNum B):
    return 1

cdef passive_force(mjtNum length, mjtNum fpmax, mjtNum b):
    return 1

cdef active_velocity(mjtNum velocity, mjtNum c, mjtNum vmax, mjtNum fvmax):
    return 1

cdef rescale_length(mjtNum length, int id):
    return length

cdef rescale_velocity(mjtNum velocity, int id):
    return velocity

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
        actuator_length_range_0=m.actuator_lengthrange[2 * id],
        actuator_length_range_1=m.actuator_lengthrange[2 * id + 1],
        range_0=m.actuator_gainprm[id + IDX_RANGE_0],
        range_1=m.actuator_gainprm[id + IDX_RANGE_1],
        force=m.actuator_gainprm[id + IDX_FORCE],
        scale=m.actuator_gainprm[id + IDX_SCALE],
        lmin=m.actuator_gainprm[id + IDX_LMIN],
        lmax=m.actuator_gainprm[id + IDX_LMAX],
        vmax=m.actuator_gainprm[id + IDX_VMAX],
        fpmax=m.actuator_gainprm[id + IDX_FPMAX],
        fvmax=m.actuator_gainprm[id + IDX_FVMAX]))
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
