# Vendored TabDiff research dependency

This directory contains the TabDiff runtime used by MTAM-HG. The upstream
project is [MinkaiXu/TabDiff](https://github.com/MinkaiXu/TabDiff), the official
implementation of *TabDiff: a Mixed-type Diffusion Model for Tabular Data
Generation* (ICLR 2025).

MTAM-HG adds a differentiable CAPL mechanism module and CLI hooks for the three
paper energies, fine-tuning steps, diffusion timesteps, and sampling guidance.
The upstream MIT license is preserved in [`LICENSE`](LICENSE).

To protect the industrial partner, this vendored tree deliberately omits every
dataset, generated table, checkpoint, result, debug artifact, demo media file,
and Python cache. Those paths are also blocked by the root release audit.

Use the root MTAM-HG runners instead of invoking this directory directly; they
prepare only the authorized training split and keep all runtime artifacts in
ignored paths.
