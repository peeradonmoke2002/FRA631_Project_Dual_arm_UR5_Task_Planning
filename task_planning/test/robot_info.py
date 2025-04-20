import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import socket
import time

def get_dashboard_info(robot_ip, command, port=29999):
    """
    Query the UR Dashboard Server for information.
    
    Parameters:
        robot_ip (str): The IP address of the robot.
        command (str): The dashboard command to send (e.g., 'get model').
        port (int): The port for the dashboard server (default is 29999).
        
    Returns:
        str: The response from the robot.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect((robot_ip, port))
        # Send the command followed by newline (UR requires newline-terminated commands)
        s.sendall((command + "\n").encode("utf-8"))
        response = s.recv(1024).decode("utf-8").strip()
    return response

def main():
    ROBOT_IP = "192.168.200.10"  # Change to your UR5's IP
    RTDE_PORT = 30004            # Default RTDE port

    # Load RTDE configuration (ensure record_configuration.xml is set to output desired fields)
    conf = rtde_config.ConfigFile("./RTDE_Python_Client_Library/examples/record_configuration.xml")
    output_names, output_types = conf.get_recipe("out")

    # Connect to the robot using RTDE
    con = rtde.RTDE(ROBOT_IP, RTDE_PORT)
    con.connect()
    con.send_output_setup(output_names, output_types)
    con.send_start()

    # Get high-level metadata using the Dashboard Server
    try:
        model_info = get_dashboard_info(ROBOT_IP, "get model")
        print("Robot Model (from Dashboard Server):", model_info)
    except Exception as e:
        print("Error querying Dashboard Server for model:", e)

    print("\nConnected to UR5 via RTDE. Retrieving real-time data...\n")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            state = con.receive()
            if state:
                # Print all available RTDE fields (for debugging/inspection)
                print("Available fields:", list(state.__dict__.keys()))
                
                # Print joint positions and robot mode code
                print("Joint positions:", state.actual_q)
                print("Tool pose:", state.actual_TCP_pose)
                print("Robot mode (code):", state.robot_mode)
                print("Joint Limit", )
                
                # Optionally, decode the robot_mode code using a lookup table if available.
                # For example:
                robot_mode_map = {
                    0: "NO_CONTROLLER",
                    1: "INIT",
                    2: "BOOTING",
                    3: "POWER_OFF",
                    4: "POWER_ON",
                    5: "IDLE",
                    6: "BACKDRIVE",
                    7: "RUNNING",
                    8: "UPDATING_FIRMWARE"
                }
                mode_str = robot_mode_map.get(state.robot_mode, "UNKNOWN")
                print("Robot mode (interpreted):", mode_str)
                
                print("-" * 50)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        con.disconnect()

if __name__ == "__main__":
    main()
