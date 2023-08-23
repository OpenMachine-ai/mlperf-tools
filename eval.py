# For the tiny MLPerf visual wake word (vww) detection model, this script
# downloads the dataset from Silabs and runs both TFLite reference models
# (int8-model and float-model) with the 1000 images listed in y_labels.csv
# to measure their accuracy
#
# Usage:
#   python3 eval.py
#
# Alternatively, if you don't want to run this python script locally, you can
# run in the cloud by opening the Jupyter Notebook eval.ipynb in your browser
# https://colab.research.google.com/github/OpenMachine-ai/mlperf-tools/blob/main/eval.ipynb
#
# Whenever you change this script, make sure to regenerate the Jupyter Notebook
# as follows:
#   pip install p2j
#   p2j eval.py

import os, csv, PIL
import numpy as np
import tensorflow as tf

#-------------------------------------------------------------------------------
# download models, dataset, y_lables.csv, and unzip
#-------------------------------------------------------------------------------
url = 'https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/'
os.system('curl -O ' + url + 'evaluation/datasets/vww01/y_labels.csv')
os.system('curl -O ' + url + 'training/visual_wake_words/trained_models/vww_96_int8.tflite')
os.system('curl -O ' + url + 'training/visual_wake_words/trained_models/vww_96_float.tflite')
os.system('curl -O https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz')
os.system('tar -xf vw_coco2014_96.tar.gz')

#-------------------------------------------------------------------------------
# set up TFLite interpreters for both models int8 (i8) and floating-point (fp)
#-------------------------------------------------------------------------------
i8_intp = tf.lite.Interpreter(model_path='vww_96_int8.tflite')
fp_intp = tf.lite.Interpreter(model_path='vww_96_float.tflite')
i8_intp.allocate_tensors()
fp_intp.allocate_tensors()

#-------------------------------------------------------------------------------
# run inference for each file listed in y_labels.csv
#-------------------------------------------------------------------------------
def inference(intp, img):
  """run inference for image 'img' using interpreter 'intp' and return the
  probability (0.0 ... 1.0) of the image being a person"""
  intp.set_tensor(intp.get_input_details()[0]['index'], img)
  intp.invoke()
  out = intp.get_tensor(intp.get_output_details()[0]['index'])[0][1]
  # for int8-model, convert int8 output (-128..127) to float 0.0 ... 1.0
  if np.issubdtype(out.dtype, np.integer):
    out = (out + 128) / 256
    # TODO: note that the maximum output here is 255/256 = 0.996 = 99.6%, so we
    # can never reach exactly 100%. This is a strange thing in TFLite, scaling
    # by 1/255 instead of 1/256 would support full range 0% to 100%
  return out

i8_pass_cnt, fp_pass_cnt, img_cnt = 0, 0, 0
with open('y_labels.csv') as f:
  for row in csv.reader(f):
    # get image 'img' from a file listed in y_labels: row[0] = filename;
    # row[2] = '1' or '0' for person/non-person; ignore row[1]
    fname = 'COCO_val2014_' + os.path.splitext(row[0])[0] + '.jpg'
    dir = 'person/' if row[2] == '1' else 'non_person/'
    img = PIL.Image.open('vw_coco2014_96/' + dir + fname)

    # convert to numpy array and reshape from (96, 96, 3) to (1, 96, 96, 3)
    img = np.asarray(img).reshape(1, 96, 96, 3)  # values are 0..255

    # for int8-model, convert from uint8 (0..255) to int8 (-128..127)
    i8_img = (img - 128).astype(np.int8)

    # for float-model, convert uint8 to float32
    fp_img = (img / 256).astype(np.float32)  # TODO, is this correct???

    # run inference for both models
    i8_out = inference(i8_intp, i8_img)
    fp_out = inference(fp_intp, fp_img)

    # for debug only: print absolute error of percent-values
    #print(int(np.abs(i8_out - fp_out) * 100))

    # check if the models were correct and update their pass-counters
    img_cnt += 1
    if row[2] == '1':
      if i8_out >= 0.5: i8_pass_cnt += 1
      if fp_out >= 0.5: fp_pass_cnt += 1
    else:
      if i8_out < 0.5: i8_pass_cnt += 1
      if fp_out < 0.5: fp_pass_cnt += 1
    # TODO: here we assume probability_of_person >= 0.5 indicates a person
    # (as opposed to '> 0.5'). Changing it to '> 0.5' reduces the accuracy by
    # 0.3 for the int8-model (no impact for float-model because it almost
    # never hits exactly 0.5 float value).  For the int8-model, using >= 0.5
    # makes sense because:
    #   - there are 256 possible output values for probability_of_person
    #   - to be unbiased, map exactly half of these 256 values to 'person',
    #     and the other half to 'non-person': values 128..255 -> 'person';
    #     values 0..127 -> 'non-person'
    #   - so value 128 maps onto 'person', which is exactly 128/256 = 0.5
    #     after conversion to float

print('float accuracy:', 100 * fp_pass_cnt / img_cnt, '  <--- this should be 80% or higher')
print('int8 accuracy :', 100 * i8_pass_cnt / img_cnt, '  <--- this should be 80% or higher')
print('image count   :', img_cnt, '  <--- this should be 1000')

# TODO: it's strange that running this script in colab produces 85.9%
# int8-accuracy, but 86.0% on my MacBook M1. Perhaps the rounding
# done by the TFLite interpreter depends on the machine architecture.

#-------------------------------------------------------------------------------
# clean up
#-------------------------------------------------------------------------------
os.system('rm -Rf vw_coco2014_96.tar.gz vw_coco2014_96 y_labels.csv')
os.system('rm -Rf vww_96_float.tflite vww_96_int8.tflite')

#-------------------------------------------------------------------------------
# references for VWW of MLPerf tiny benchmark
#-------------------------------------------------------------------------------
#  - benchmark: https://github.com/mlcommons/tiny/tree/master/benchmark
#  - VWW model: https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words
#  - Issue #135: https://github.com/mlcommons/tiny/issues/135
#  - Paper 'MLPerf tiny benchmark' https://arxiv.org/pdf/2106.07597.pdf
#  - Paper 'Visual Wake Words Dataset' https://arxiv.org/abs/1906.05721
#  - Source of the data set:
#    https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/download_and_train_vww.sh
#  - Repo with v1.1 benchmark results:
#    https://github.com/mlcommons/tiny_results_v1.1/tree/main
#  - Book TinyML https://www.abebooks.com/servlet/SearchResults?kn=TinyML
