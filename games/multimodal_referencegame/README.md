# Multimodal Drawing Instruction Giving and Following: multimodal reference

Implemented by: Sherzod Hakimov

Each instance of this game includes three grids where one is the target and the remaining two are distractor grids, which Player A (Instruction Giver) needs to generate a referring expression that describes only the target grid and Player B (Instruction Follower) takes into account the given referring expression and expected to guess which of the given three grids is the target grid. The grids are given as images to a model (model that support images and text data).

The Game Master selects a target and two distractor grids and instructs the Player A to generate a referring expression that uniquely describes the target grid and differentiates it from the distractors. There is a history of work on referring expression generation and the topic has recently received new attention in the context of neural learners. The Game Master then provides the same three grids and the referring expression from Player A to Player B. The three grids are numbered such as *first*, *second*, and *third* and the order of grids are randomly shuffled for Player B. Player B generates a single expression that should refer to the number of the target grid that matches the given expression. The game is played for a single turn.

### Instantiation
We used existing grids for the text-based (ASCII representation) of grids and converted into images using `matplotlib` Python library. Then we used existing photo datasets (DOCCI, ADE20K, CLEVR) and selected some instances from them. Finally, we used pentomino pieces and created instances from them.

1. **Edit distance of two**: We apply one or two edits to the target grid to obtain a distractor grid. We created 18 such tuples of a target and two distractor grids using two edits.
2. **Edit distance of four**: We apply the same idea explained above but create 18 grids with four edits.

We want to to measure whether the tested language models are able to differentiate between grids that look a like (two edit distances) and whether it is simpler compared to grids that somewhat look slightly different (four edit distances).

### Evaluation
The evaluation of each episode is done by checking whether the Player B guesses the target grid correctly. It is simply "successful" when the generated expression matches the number of the target grid and "failed" otherwise. Additionally, we also measure the number of characters and the token size in the referring expression generated by the Player A.



## Used Datasets:

DOCCI dataset: [link](https://google.github.io/docci/#downloads)

ADE20K dataset: [link](https://datasetninja.com/ade20k#download)

CLEVR dataset: [link](https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip)