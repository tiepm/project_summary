Use R packages tm and RKEA to find key words in documents. Accuracy depends on label data volume

Example usage
- Extract data archives:
  - data02-KEA2-train.zip: Training inputs
  - data02-KEA2-PP-train-label.zip: Training labels
  - data02.zip: New data inputs
- Clean data
  - python 004-04A-KEA2-PP.py data02-KEA2-train/ data02-KEA2-PP-train
  - python 004-04A-KEA2-PP.py data02/ data02-KEA2-PP/
- Train
  - Rscript 004-04B-KEA2-training.R data02-KEA2-PP-train/ data02-KEA2-PP-train-label/ 004-04B-KEA2-model
- Run on new data
  - python 004-04D-KEA2-KExtractL.py data02/ data02-KEA2-PP/ data02-KEA2-PP-output-label/ 004-04B-KEA2-model
- Clean up temp db files
  - rm *.db
  - rm *.db___LOCK



