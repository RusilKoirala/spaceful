const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("start");
const speedSelect = document.getElementById("speed");
const raysToggle = document.getElementById("rays");
const statusText = document.getElementById("status");
const boardBody = document.getElementById("board");

const track = new Image();
track.src = "track.png";

let state = { status: "idle", cars: [] };



// ---- controls ----

startButton.onclick = () => {
  const speed = Number(speedSelect.value);

  fetch("start", {
    method: "POST",
    body: JSON.stringify({ speed }),
  });
};

raysToggle.onchange = draw;
track.onload = draw;



// ---- live updates from the server ----

const events = new EventSource("events");

events.onmessage = (e) => {
  state = JSON.parse(e.data);
  draw();
  board();
};

events.onerror = () => {
  statusText.textContent = "Disconnected, retrying…";
};



// ---- drawing ----

function draw() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (track.complete) {
    ctx.drawImage(track, 0, 0);
  }

  for (const car of state.cars) {
    if (raysToggle.checked) {
      drawRays(car);
    }
    drawCar(car);
  }

  ctx.globalAlpha = 1;
}


function drawRays(car) {
  ctx.strokeStyle = car.color;
  ctx.fillStyle = car.color;
  ctx.globalAlpha = 0.5;

  for (const [px, py] of car.rays) {
    ctx.beginPath();
    ctx.moveTo(car.x, car.y);
    ctx.lineTo(px, py);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}


function drawCar(car) {
  const [w, h] = state.size || [14, 7];

  ctx.globalAlpha = car.alive || car.finished ? 1 : 0.3;

  ctx.save();
  ctx.translate(car.x, car.y);
  ctx.rotate(-car.angle * Math.PI / 180);   // pygame angles turn counter-clockwise
  ctx.fillStyle = car.color;
  ctx.fillRect(-w / 2, -h / 2, w, h);
  ctx.restore();
}



// ---- leaderboard ----

function label(car) {
  if (car.finished) return `Finished ${car.time.toFixed(2)}s`;
  if (car.crashed) return "Crashed";
  if (!car.alive) return "Stalled";

  return state.status === "done" ? "Out of time" : "Racing";
}


function statusLabel() {
  if (state.status === "idle") return "Press start";
  if (state.status === "running") return `Racing ${state.seconds.toFixed(1)}s  x${state.speed}`;

  return `Done at ${state.seconds.toFixed(1)}s`;
}


function ranked(cars) {
  // finished first (fastest lap on top), then whoever is furthest along
  return [...cars].sort((a, b) => {
    if (a.finished !== b.finished) return b.finished - a.finished;
    if (a.finished) return a.time - b.time;
    return b.progress - a.progress;
  });
}


function row(car) {
  const percent = Math.round(car.progress * 100);

  return `
    <tr>
      <td class="name">
        <span class="dot" style="background:${car.color}"></span>${car.name}
      </td>
      <td class="state">${label(car)}</td>
      <td>
        <div class="bar">
          <div style="width:${percent}%; background:${car.color}"></div>
        </div>
      </td>
      <td class="percent">${percent}%</td>
    </tr>`;
}


function board() {
  statusText.textContent = statusLabel();
  boardBody.innerHTML = ranked(state.cars).map(row).join("");
}
