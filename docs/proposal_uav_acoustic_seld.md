---
title: "Hearing Drones in 3D: Convolutional Detection and Direction-Finding from a Microphone Array"
author: "Shai Livshits, Dan Masad"
date: ""
geometry: margin=0.8in
fontsize: 10pt
colorlinks: true
header-includes: |
  \usepackage{titlesec}
  \usepackage{titling}
  \usepackage{enumitem}
  \setlength{\droptitle}{-5em}
  \posttitle{\par\end{center}}
  \titlespacing*{\section}{0pt}{5pt}{1pt}
  \setlength{\parskip}{2pt}
  \setlist[enumerate]{itemsep=0pt,topsep=2pt,parsep=0pt,leftmargin=1.4em}
---

**Project Category:** Applied -- Audio / Convolutional Networks / Defense & Security

## Problem Definition

Drones are hard to see and often fly with no radio link, defeating cameras and radio
detectors. Sound is a cheap fallback: rotors are loud and distinctive. From a ground microphone array we
want a network reporting ten times per second
whether a drone is audible and where: azimuth (compass bearing) and
elevation (angle above the horizon).

## Key Challenges

A distant drone is quiet and easily confused with wind, traffic, and aircraft. It
fills a tiny fraction of the search space, so "nothing" scores well -- severe class imbalance. Recordings are continuous: neighbouring
windows look almost identical, so a random split would leak training data into test.

## Data

Western Sydney University's *UAV Acoustic Localization Dataset* (CC BY-SA 4.0) downloads
freely: ~75 minutes of a DJI Air 3
at two sites, ~10 minutes of drone-free background, with GPS
logs every 100 ms giving direction and speed: labels come free. The 24
microphones sit on a 1.3 m tripod wired as four synchronised six-channel
recorders. A downloaded slice confirmed labels and geometry agree; we train on one
site, test on another.

## Proposed Method

**From sound to images:** Sound from one direction reaches the microphones at slightly
different times. Delaying each by those offsets and summing
reinforces a correct guess and partly cancels a wrong one. Sweeping the
dataset's direction grid turns each 100 ms of audio into a $91\times23$ image[^grid] of
energy per direction; the drone is a bright blob. Fixed signal processing
makes this an image problem.

[^grid]: Azimuth $-180^\circ$ to $180^\circ$ and elevation $0^\circ$ to $88^\circ$, both
in $4^\circ$ steps, giving $91\times23$ directions. $4^\circ$ is finer than the array's
$\approx7^\circ$ beamwidth at 2200 Hz, so the main lobe is oversampled rather than
missed.

**Models:** The baseline picks the brightest pixel, as classical systems do, failing
on noise and at range. We train, from scratch, an encoder--decoder CNN outputting
a per-pixel drone probability plus one "drone present" score, with class-weighted cross-entropy and Adam. Consecutive images enter as extra input
channels.

## Evaluation

On the unseen test site: precision--recall and false alarms per hour for detection; median
angular error and hit rates at the 10$^\circ$ the dataset calls
correct. Everything is repeated on six-microphone subsets -- a single
recorder, then six spanning the full 1.3 m -- and for several numbers of stacked
images.

## Related Work and Contribution

Rodríguez et al. (2026) released this dataset and a single-frame model, naming temporal
context as future work and publishing no code; Al-Emadi et al. (2019) detect from one
microphone, without direction. Our question is practical: how little array does this
task need? Six of the 24 microphones already span the whole 1.3 m, so the other
18 buy no angular resolution -- only suppression of the false peaks a sparse array
invents. We hypothesise that temporal context supplies it instead:
a real drone persists while false peaks flicker -- making cheap, widely
spaced arrays operationally practical. We also re-test their reported drop in false
alarms over classical beamforming: they scored it by output stability; our
drone/no-drone score gives a true false-alarm rate. As an extra we try speed from an
approaching drone's Doppler pitch shift, though throttle moves rotor pitch more than
motion.

## References

\small

1. B. J. Rodríguez, S. F. Chevtchenko, M. Herrera Martínez, Y. Bethi, S. Afshar.
   *Acoustic Imaging for UAV Detection: Dense Beamformed Energy Maps and U-Net SELD.*
   [arXiv:2508.00307](https://arxiv.org/abs/2508.00307), Aug 2025 (rev. Jul 2026);
   IWAENC 2026.
2. B. J. Rodríguez, S. F. Chevtchenko, S. Afshar. *UAV Acoustic Localization Dataset:
   24-Channel Beamformed Recordings of a DJI Air 3 Drone with Synchronized GPS Flight
   Logs.* Western Sydney University, published 15 Jul 2026.
   [DOI 10.26183/07nm-4p35](https://doi.org/10.26183/07nm-4p35).
3. S. Al-Emadi, A. Al-Ali, A. Mohammad, A. Al-Ali. *Audio Based Drone Detection and
   Identification using Deep Learning.* IWCMC, 2019.
   [DOI 10.1109/IWCMC.2019.8766732](https://doi.org/10.1109/IWCMC.2019.8766732).
