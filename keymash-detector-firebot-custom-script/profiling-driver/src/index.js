import * as wasmObj from "../../core/pkg"

const input = /** @type {HTMLInputElement} */ (document.getElementById('input'));
const output = document.getElementById('output');
const timing = document.getElementById('timing');
document.querySelector('form').addEventListener('submit', (e) => {
  e.preventDefault();
  const results = [];
  const warmupStart = performance.now();
  while (performance.now() < warmupStart + 5000) {
    results.push(wasmObj.test_fragment(input.value));
  }
  const timings = [];
  for (let i = 0; i < 40; i++) {
    const startTime = performance.mark(`${i}`);
    results.push(wasmObj.test_fragment(input.value));
    const duration = performance.measure(`d${i}`, startTime.name).duration;
    timings.push(duration);
  }
  output.innerText = String(results[results.length - 1]);
  timing.innerText = String(timings);
});
