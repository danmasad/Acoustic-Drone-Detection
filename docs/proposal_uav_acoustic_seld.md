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

Drones are hard to see and can fly with their radio link off, defeating cameras and
radio detectors. Sound is a cheap fallback: rotors are loud and distinctive. From 24 ground
microphones recording simultaneously, we want a network that reports ten times per
second whether a drone is audible and in which direction -- azimuth (compass bearing)
and elevation (angle above the horizon).

## Key Challenges

A distant drone is quiet and easily confused with wind, traffic, and aircraft. It
fills a tiny fraction of the search space, so always answering "nothing" already scores
well -- severe class imbalance. Recordings are continuous, so neighbouring 100 ms
windows look almost identical: a random split would leak training data into the test
set.

## Data

Western Sydney University's *UAV Acoustic Localization Dataset* (CC BY-SA 4.0, DOI
10.26183/07nm-4p35) downloads freely over HTTP, no registration: ~75 minutes of
24-channel outdoor recordings of a DJI Air 3 at two sites, ~10 minutes of drone-free
background, and GPS logs every 100 ms giving true direction and speed -- labels come
for free. We downloaded a slice and confirmed audio, labels, and geometry agree. We
train on one site and test on a second, unseen one.

## Proposed Method

**From sound to images:** Sound from a given direction reaches the microphones at
slightly different times. Delaying each microphone by exactly those offsets and adding
the signals makes a correct guess reinforce and a wrong guess partly cancel. Sweeping
a grid of 91 bearings $\times$ 23 elevations turns each 100 ms of audio into a
$91\times23$ image whose pixels hold the energy arriving from each direction; the
drone is a bright blob. This is fixed signal processing with nothing to
learn, and it turns the task into an image problem.

**Models:** The baseline picks the brightest pixel, as classical systems do; it fails
on noise and at range. Against it we train a small CNN outputting a per-pixel
probability that the drone is there plus one overall "drone present" score, with
class-weighted cross-entropy and Adam. Our main experiment feeds several consecutive
images as extra input channels, letting the network use the fact that a real drone
drifts smoothly while noise flickers. A secondary output regresses speed, testing
whether the network picks up the Doppler pitch shift of an approaching drone.

## Evaluation

On the unseen site: precision--recall curves and false alarms per hour (detection),
median angular error and frames within 5/10/20$^\circ$ (direction), mean absolute error
(speed). Ablations: number of stacked frames; accuracy with 12, 6, or 4 of the 24 microphones.

## Related Work and Contribution

Rodríguez et al. (arXiv:2508.00307) published this dataset with a convolutional model
that sees one frame at a time, named temporal context as future work, and released no
training code. Al-Emadi et al. (2019) detect drones from one microphone, without
direction. We contribute an open
PyTorch implementation, the first use of consecutive frames on this dataset, a proper
detection evaluation (the original reports only a five-minute false-alarm check), and
a microphone-count study for cheaper arrays.

## References

\small

1. B. J. Rodríguez, S. F. Chevtchenko, M. Herrera Martínez, Y. Bethi, S. Afshar.
   *Acoustic Imaging for UAV Detection: Dense Beamformed Energy Maps and U-Net SELD.*
   arXiv:2508.00307. Dataset: DOI 10.26183/07nm-4p35.
2. S. Al-Emadi, A. Al-Ali, A. Mohammad, A. Al-Ali. *Audio Based Drone Detection and
   Identification using Deep Learning.* IWCMC, 2019.
