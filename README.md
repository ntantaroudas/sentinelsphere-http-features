# A pragmatic reassessment of HTTP aware features for web attack detection on CIC-IDS2017 and CSIC 2010

Code and data for the Open Research Europe article "A pragmatic reassessment of HTTP aware features for web attack detection on CIC-IDS2017 and CSIC 2010" by Nikolaos D. Tantaroudas, Ilias Karachalios and Andrew J. McCracken, a follow on study of the SentinelSphere platform carried out under the Horizon Europe project ResilMesh, Grant Agreement No. 101119681.

Preprint of the short version, https://doi.org/10.20944/preprints202610.1120.v1.

## Contents and how to reproduce

Code and aggregated results for "Do HTTP features carry the signal? Shortcut learning and label errors in web attack detection on CIC-IDS2017 and CSIC 2010".

`code` holds the repaired feature extractor, the experiment drivers `exp_cic.py` and `exp_csic.py` and the analysis script `analyse.py`. `results/cic_results.json` and `results/csic_results.json` hold every metric of every configuration and split, the per class out of fold detection, the single feature ranking, the corrected label analysis and the CSIC ablation.

The datasets are not redistributed. Place the CIC-IDS2017 Thursday morning web attack CSV and the corrected Thursday CSV of Engelen, Rimmer and Joosen in `data`, or point the environment variable `PAPER_B_DATA` at them, and the raw CSIC 2010 files in `data/csic`, or point `PAPER_B_CSIC` at them. Then run `python3 code/exp_cic.py`, `python3 code/exp_csic.py` and `python3 code/analyse.py`.

The analysis scripts import `figstyle.py` from the folder `code`, which sets the figure style used in the article, drawn at a display width of at most 150 mm and saved as EPS, PDF and 600 dpi PNG.

## Licence

The code is released under the MIT License, see `LICENSE`. The aggregated results and figures are released under the Creative Commons Zero 1.0 dedication, see `DATA_LICENSE.md`.

## Citation

Please cite the article, and this repository through its Zenodo DOI once a release has been archived, see `CITATION.cff`.
