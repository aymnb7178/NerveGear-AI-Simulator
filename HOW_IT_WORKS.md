# How NerveGear AI Simulator works

```
generate() -> band_powers() -> relative / engagement -> classify_state() -> dashboard
signal.py       analysis.py          analysis.py          analysis.py     dashboard.py
```

## Background in two lines
EEG is a voltage over time. Neuroscience groups its frequencies into bands: delta (0.5-4 Hz, deep sleep), theta (4-8, drowsy), alpha (8-13, relaxed/eyes closed), beta (13-30, active thinking), gamma (30-45). Which band dominates is a rough hint of mental state.

## 1. Synthetic signal (`signal.py`)
For each band, add three sine waves at random frequencies inside the band. Their amplitude comes from a table per scenario (relaxed = big alpha, focused = big beta, drowsy = big delta+theta). Add Gaussian noise. A fixed `seed` gives the same signal every time.

## 2. Band power (`analysis.py`)
Remove the mean, multiply by a Hann window (reduces spectral leakage), take the FFT, square it to get power per frequency, then sum the power inside each band's frequency range.

## 3. Engagement index
`beta / (alpha + theta)` - a ratio used in published EEG-engagement work. High = more beta relative to slow waves. Here it is only computed on simulated data.

## 4. State label
Rules on relative power (band / total): beta dominant -> focused; delta+theta over half -> drowsy; otherwise relaxed.

## 5. Dashboard
OpenCV draws the signal trace, stacked relative-power bars per 2-second window, a state colour strip and the engagement number.

## Interview explanation
"It's a simulation of the signal-processing side of EEG: I generate synthetic band-limited signals, compute band power with an FFT, derive an engagement ratio and a simple rule-based state, and plot it. There's no hardware and no real data, so it demonstrates the pipeline, not any medical or BCI capability. Tests check it with known sine waves. I built it with AI assistance and can explain each step."

## Limits
Real EEG has artifacts (blinks, muscle, mains hum), multiple channels and large person-to-person differences. None of that is simulated.
