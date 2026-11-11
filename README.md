# nav-imitation-learning

Portfolio demo by **Arpan Bom** (`applycoding@gmail.com`). A CPU-only
**simulation** lab for imitation learning in planar navigation. A scripted
expert (grid A* plus a lookahead heading tracker) records demonstrations on
synthetic occupancy maps. A numpy MLP is trained by behavior cloning, and a
DAgger-style loop aggregates labels at policy-visited states. A static GitHub
Pages dashboard shows **recorded CI output** only; the page does not train and
does not simulate.

**Simulation only.** Synthetic maps and a scripted expert. Not a production
driving policy. Not a foundation vision-language-action model. Not Nav2. Not
LeRobot. Not on-robot data. Not
[applycoding/robot-learning-vla-demo](https://github.com/applycoding/robot-learning-vla-demo)
(that repository is a language-conditioned 2-link arm). The tracker is a
lookahead heading controller, not Coulter's curvature law.

Licensed under the [Apache License, Version 2.0](LICENSE). SPDX headers will
appear in authored sources.

## Status

This initial commit contains the planning files (`PLAN.md`, `TASKS.md`,
`REFERENCES.md`). Implementation follows those documents.

## Author

Arpan Bom <applycoding@gmail.com>
