#! /usr/bin/env python3
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

ROBOT_IP = "192.168.200.10"
PORT = 30004

conf = rtde_config.ConfigFile("./RTDE_Python_Client_Library/examples/record_configuration.xml")  # XML defines what to read
output_names, output_types = conf.get_recipe("out")

con = rtde.RTDE(ROBOT_IP, PORT)
con.connect()

# Setup data output
con.send_output_setup(output_names, output_types)

# Start receiving data
con.send_start()

try:

    state = con.receive()
    if state:
        print("Joint positions:", state.actual_q)
        print("Joint velocities:", state.actual_qd)
        print("Joint currents:", state.actual_current)
        print("Tool Accelerometer:", state.actual_TCP_force)
        print("Tool Accelerometer:", state.actual_TCP_pose)
except KeyboardInterrupt:
    print("Terminating connection...")
finally:
    con.send_pause()
    con.disconnect()
    print("Connection closed.")
