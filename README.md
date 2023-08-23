This repo contains useful tools for running MLPerf benchmarks.
- [eval.py](eval.py): For the tiny MLPerf visual wake word (vww) detection model,
  this script downloads the dataset from Silabs and runs both TFLite reference
  models (int8-model and float-model) with the 1000 images listed in y_labels.csv
  to measure their accuracy.
- [eval.ipynb](eval.ipynb): Jupyter notebook generated from [eval.py](eval.py),
  click  <a href="https://colab.research.google.com/github/OpenMachine-ai/mlperf-tools/blob/main/eval.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20"> </a>
  to run it in your browswer.
