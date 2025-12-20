# Rethinking Fusion and Orthogonal Projection for Face-Voice Association (FAME 2026)
Paper Link: [arxiv](https://arxiv.org/abs/2512.02860) &nbsp; 


## Overview
RFOP revisits the fusion and orthogonal projection for face-voice association by effectively focusing on the relevant semantic information within the two modalities.

<img width="480" alt="image" src="images/rfop.png">

## Installation
Please follow the instructions [here](https://github.com/msaadsaeed/FOP) to make the environment and install the libraries.

## Training
Use following command to train the model
```
python main.py --batch_size 64 --epochs 50 --dim_embed 256
```

## Score Computation
Use following command to compute score for the trained model
```
python computeScore.py --ckpt <path to checkpoint.pth.tar> --dim_embed 256 
```

## Acknowledgements
The codebase is inspired from the [FOP](https://github.com/msaadsaeed/FOP) repository. We thank them for releasing their valuable codebase. 

## Citation
```
@misc{rfop2025,
      title={RFOP: Rethinking Fusion and Orthogonal Projection for Face-Voice Association}, 
      author={Abdul Hannan and Furqan Malik and Hina Jabbar and Syed Suleman Sadiq and Mubashir Noman},
      year={2025},
      eprint={2512.02860},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.02860}, 
}
```
