# Online Unsupervised Arm Posture Adaptation for sEMG-based Gesture Recognition on a Parallel Ultra-Low-Power Microcontroller



## Introduction
The repository contains the code developed for our paper M. Zanghieri _et al._, “Online Unsupervised Arm Posture Adaptation for sEMG-based Gesture Recognition on a Parallel Ultra-Low-Power Microcontroller,” _IEEE BioCAS_, 2023 [[1]](#1).

We also release our [**UniBo-INAIL dataset**](https://github.com/pulp-bio/unibo-inail-semg-dataset/tree/main) for research on multi-subject, multi-day, and multi-posture sEMG.



## Usage
1. Run ``experiment_multiposture.ipynb`` (or equivalently ``experiment_multiposture.py``) to perform the baseline experiments with multi-posture training. 
2. Run ``experiment_calibmodes.ipynb`` (or equivalently ``experiment_calibmodes.py``) for the online PCA posture adaptation experiments.
3. Run ``read_results.ipynb`` to get the results statistics.



## Authors
This work was realized mainly at the [**Energy-Efficient Embedded Systems Laboratory (EEES Lab)**](https://dei.unibo.it/it/ricerca/laboratori-di-ricerca/eees) of University of Bologna (Italy) by:
- [Marcello Zanghieri](https://scholar.google.com/citations?user=WnIqQj4AAAAJ&hl=en) - University of Bologna
- [Mattia Orlandi](https://scholar.google.com/citations?hl=en&user=It3fdrEAAAAJ) - University of Bologna
- [Dr. Elisa Donati](https://scholar.google.com/citations?hl=en&user=03ZYhbIAAAAJ) - INI institute (University of Zürich, ETH Zürich)
- [Prof. Emanuele Gruppioni](https://scholar.google.it/citations?user=PgLLxVsAAAAJ&hl=it) - University of Bologna, INAIL Prosthesis Centre in Vigorso di Budrio (Bologna)
- [Prof. Luca Benini](https://scholar.google.com/citations?hl=en&user=8riq3sYAAAAJ) - University of Bologna, ETH Zürich
- [Prof. Simone Benatti](https://scholar.google.com/citations?hl=en&user=8Fbi_kwAAAAJ) - University of Modena & Reggio Emilia, University of Bologna



## Citation
When referring to our paper or using our UniBo-INAIL dataset, please cite our work [[1]](#1):
```
@INPROCEEDINGS{zanghieri2023online,
  author={Zanghieri, Marcello and Orlandi, Mattia and Donati, Elisa and Gruppioni, Emanuele and Benini, Luca and Benatti, Simone},
  booktitle={2023 IEEE Biomedical Circuits and Systems Conference (BioCAS)}, 
  title={Online Unsupervised Arm Posture Adaptation for {sEMG}-based Gesture Recognition on a Parallel Ultra-Low-Power Microcontroller}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/BioCAS58349.2023.10388902}}
```


## References

<a id="1">[1]</a>
M. Zanghieri, M. Orlandi, E. Donati, E. Gruppioni, L. Benini, S. Benatti,
“Online unsupervised arm posture adaptation for sEMG-based gesture recognition on a parallel ultra-low-power microcontroller,”
in _2023 IEEE International Conference on Biomedical Circuits and Systems (BioCAS)_,
2023,
pp. 1-5.
DOI: [10.1109/BioCAS58349.2023.10388902](https://doi.org/10.1109/BioCAS58349.2023.10388902).


## License
All files are released under the LGPL-2.1 license (`LGPL-2.1`) (see `LICENSE`).
