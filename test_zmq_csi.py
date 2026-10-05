import json
import time
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "")

print("Connected to tcp://127.0.0.1:5555")
print("Waiting up to 5 seconds for CSI message...")

poller = zmq.Poller()
poller.register(sock, zmq.POLLIN)

events = dict(poller.poll(5000))

if sock not in events:
    print("FAIL: no ZMQ message received within 5 seconds")
else:
    raw = sock.recv()
    data = json.loads(raw.decode("utf-8"))

    print("ZMQ message received: PASS")
    print("seq =", data["seq"])
    print("number of magnitude values =", len(data["csi_mag"]))
    print("number of phase values =", len(data["csi_phase"]))
    print("mag first/last =", data["csi_mag"][0], data["csi_mag"][-1])
    print(
        "phase first/middle/last =",
        data["csi_phase"][0],
        data["csi_phase"][53],
        data["csi_phase"][-1],
    )

sock.close()
ctx.term()