from XCM.msg import JointStateAdvr
from XCM.msg import CommandAdvr

def config_advr_command(joint_names, cmd_type, init_cmd_vals):
    advr_cmd_msg = CommandAdvr()
    advr_cmd_msg.name = joint_names
    advr_cmd_msg = update_advr_command(advr_cmd_msg, cmd_type, init_cmd_vals)
    #if hasattr(advr_cmd_msg, cmd_type):
    #    setattr(advr_cmd_msg, cmd_type, init_cmd_vals)
    #else:
    #    raise ValueError("Wrong ADVR command type option")
    return advr_cmd_msg


def update_advr_command(cmd_msg, cmd_type, cmd_vals):
    if hasattr(cmd_msg, cmd_type):
        setattr(cmd_msg, cmd_type, cmd_vals)
    else:
        raise ValueError("Wrong ADVR command type option")
    return cmd_msg

def get_advr_sensor_data(obs_msg, state_type):
    if state_type in obs_msg:
        data_field = getattr(obs_msg, state_type)
    else:
        raise ValueError("Wrong ADVR field type option")
    return data_field

