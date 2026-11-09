
# so the server never renders or encodes video ik u get it cuz i didnt
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
os.chdir(ROOT)  
sys.path[:0] = [ROOT, os.path.join(ROOT, "car_ai")]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")   # else SDL swallows SIGTERM and kill does nothing

from arena import Rusil, ChatGPT, Gemini, Claude, SAVE_DIR
from game import *

# yes i am rusil
AIS = [Rusil(), ChatGPT(), Gemini(), Claude()]
TICK = 1 / 30    # updates sent per second; the sim itself still runs at 60 fps
SPEEDS = (1, 2, 4, 8)


class Race:
    def __init__(self):
        self.track = load_track()
        self.cond = threading.Condition()
        self.version = 0
        self.payload = json.dumps({"status": "idle", "cars": []})
        self.lock = threading.Lock()
        self.thread = None
        self.stop = threading.Event()

    def start(self, speed):
        with self.lock:
            self.stop.set()
            if self.thread:
                self.thread.join()
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self.run, args=(speed, self.stop), daemon=True)
            self.thread.start()

    def spawn(self):
       
        x, y, angle = START_GRID[0]
        cars = []
        for ai in AIS:
            path = os.path.join(SAVE_DIR, ai.file)
            if not os.path.exists(path):
                print(f"{ai.name}: no saved brain at {path}, skipping")
                continue
            car = SpaceCar(x, y, create_car_surface(ai.color), ai.name, angle)
            car.brain = ai.load(path)[0]
            car.ai = ai
            cars.append(car)
        return cars

    def run(self, speed, stop):
        cars = self.spawn()
        frame = 0
        next_tick = time.perf_counter()
        while not stop.is_set():
            for _ in range(2 * speed):
                for car in cars:
                    if car.alive:
                        car.ai.drive(car, self.track)
                        car.step(self.track)
                frame += 1
                done = frame >= ROUND_FRAMES or not any(car.alive for car in cars)
                if done:
                    break
            self.publish(self.snapshot(cars, frame, "done" if done else "running", speed))
            if done:
                return
            next_tick += TICK
            time.sleep(max(0, next_tick - time.perf_counter()))

    def snapshot(self, cars, frame, status, speed):
        lap = self.track.lap_length
        out = []
        for car in cars:
            rays = []
            if car.alive:
                for offset in OFFSETS:
                    px, py, _ = car.cast_ray(self.track, offset)
                    rays.append([round(px, 1), round(py, 1)])
            out.append({
                "name": car.name,
                "color": "rgb(%d,%d,%d)" % car.ai.color,
                "x": round(car.x, 1), "y": round(car.y, 1), "angle": round(car.angle, 1),
                "alive": car.alive, "finished": car.finished, "crashed": car.crashed,
                "time": round(car.finish_frame / FPS, 2) if car.finished else None,
                "progress": round(min(max(car.progress, 0) / lap, 1), 3),
                "rays": rays,
            })
        return {"status": status, "seconds": round(frame / FPS, 2), "speed": speed,
                "size": [CAR_SIZE[0], CAR_SIZE[1]], "cars": out}

    def publish(self, state):
        with self.cond:
            self.payload = json.dumps(state)
            self.version += 1
            self.cond.notify_all()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_file(os.path.join(WEB, "index.html"), "text/html; charset=utf-8")

        elif self.path == "/style.css":
            self.send_file(os.path.join(WEB, "style.css"), "text/css; charset=utf-8")

        elif self.path == "/app.js":
            self.send_file(os.path.join(WEB, "app.js"), "text/javascript; charset=utf-8")

        elif self.path == "/track.png":
            self.send_file(os.path.join(ROOT, "assets", "track.png"), "image/png")
        elif self.path == "/events":
            self.events()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/start":
            return self.send_error(404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            speed = int(body.get("speed", 1))
        except ValueError:
            speed = 1
        race.start(speed if speed in SPEEDS else 1)
        self.send_response(204)
        self.end_headers()

    def send_file(self, path, kind):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def events(self):
        # server-sent events: one JSON message per race update
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seen = -1
        try:
            while True:
                with race.cond:
                    race.cond.wait_for(lambda: race.version != seen, timeout=15)
                    version, payload = race.version, race.payload
                if version == seen:
                    self.wfile.write(b": ping\n\n")   # keeps proxies from closing an idle stream
                else:
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    seen = version
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def main():
    global race
    parser = argparse.ArgumentParser(description="Web page to watch the trained AIs race")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    race = Race()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
