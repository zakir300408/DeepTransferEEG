import time
import serial
import serial.tools.list_ports


UP= "AA 0D 0A"
STOP = "BB 0D 0A"
DOWN = "CC 0D 0A"

# Control Exoskeleton using Serial Communication
class ControlExoskeleton:
    def __init__(self, port):
        self.serial = serial.Serial(port, baudrate=115200, timeout=1)
        print(f"Connected to {port}")

    # List all available ports and print them
    @staticmethod
    def list_ports():
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"Port: {port.device}, Description: {port.description}")
        return ports

    # Find the port with CH340 in the description
    @staticmethod
    def find_ch340_port():
        ports = ControlExoskeleton.list_ports()
        for port in ports:
            if "CH340" in port.description:
                return port.device
        return None

    def send_hex(self, hex_string):
        """Send a space-separated HEX string over serial."""
        cleaned = hex_string.replace(" ", "")
        data = bytes.fromhex(cleaned)
        self.serial.write(data)
        print(f"Sent: {hex_string}")
        # #read response
        # response = self.serial.read(100)  # Read up to 100 bytes
        # if response:
        #     print(f"Received: {response.hex()}")
        # else:
        #     print("No response received.")

if __name__ == "__main__":
    ch340_port = ControlExoskeleton.find_ch340_port()
    if ch340_port:
        exo = ControlExoskeleton(ch340_port)
        for _ in range(10):
            exo.send_hex(UP)
            time.sleep(12)
            exo.send_hex(DOWN)
            time.sleep(12)
    else:
        print("No CH340 device found. Please connect the device or check available ports.")
