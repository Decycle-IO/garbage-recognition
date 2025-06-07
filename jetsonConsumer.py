import requests

class JetsonConsumer:
    def __init__(self, hostname='192.168.57.211:3000'):
        self.hostname = hostname

    def open_door(self):
        requests.get(f"http://{self.hostname}/open_door")
    def close_door(self):
        requests.get(f"http://{self.hostname}/close_door")
    def plastic(self):
        requests.get(f"http://{self.hostname}/plastic")
    def metal(self):
        requests.get(f"http://{self.hostname}/metal")
    def other(self):
        requests.get(f"http://{self.hostname}/other")
