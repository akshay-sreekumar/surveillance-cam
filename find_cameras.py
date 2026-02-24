import cv2

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    
    # Check up to 10 ports
    while len(non_working_ports) < 6 and dev_port < 10:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print(f"Port {dev_port}: Not Working")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"Port {dev_port}: Working - Resolution: {int(w)}x{int(h)}")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port}: Present but cannot read frames")
                available_ports.append(dev_port)
            camera.release()
        dev_port += 1
        
    return available_ports, working_ports, non_working_ports

print("Checking available camera indices...")
list_ports()
print("Done checking cameras.")
