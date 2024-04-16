#!/bin/bash

# for ((i=1; i<=29; i++));do
#     python bin/recursive_feature_addition.py -data data/Breast_cancer/LFQ.tsv -label data/Breast_cancer/patient_label.tsv -data_split data_split/Breast_cancer_LOOCV_$i.tsv -outDir result_Breast_cancer_LOOCV_$i
# done

for ((i=1; i<=10; i++));do
    python bin/recursive_feature_addition.py -data data/Asthma/MS_LFQ.tsv -label data/Asthma/patient_label.tsv -data_split data_split/Asthma_CV_$i.tsv -outDir result_Asthma_CV_$i
done