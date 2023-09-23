# motivated-predators-in-pursuit-domain

This repository is the codebase pertaining to our work on proposing a mathematical framework to model goal-driven agents with an inherent motive that also employ other goal prioritization strategies. We adopted a prey-predator domain which contains a 2D toroidal world with 3 types of preys differentiating based on their resting likelihood. The predators are of three types, based on their inherent motivation. Following a well-established human motivation psychology theory called Three Needs Theory, we modelled three types of predators with Affiliation, Achievement and Power motives. We analyzed how a team of 12 motivated predators perform in the face of different number of preys and analysed the team performance using 4 metrics; total steps, yield per step, tension per step and yield per unit tension.

The main branch contains the results obtained for the four metrics as CSV files. Running `./summary.py` will generate the ternary plots that depict the distribution of values of the metrics for each configuration.

## Run Code
- Switch the branch to `heterogeneous-predators`.
- Execute the script path `motivated-predators-in-pursuit-domain\seekAndTrack\prepareEnv.py` with parameters similar to the follwoing example.
- `--grid-size 16 --motive-profile-ratio 4 4 4 --predator-count 12 --prey-count 6 --difficulty-level 0` will configure a 16x16 arena with a team of 12 predators equally distributed across the 3 motives. The arena will have 6 preys where 80% of the preys will be extremely slow as the difficulry level is set to 'Easy' (0).

## Cite
Please cite our paper if you use this code in your work.
