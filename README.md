# Improved Difference Images for Change Detection Classifiers in SAR Imagery Using Deep Learning

This repository includes code for the paper **Improved Difference Images for Change Detection Classifiers in SAR Imagery Using Deep Learning** (submitted to peer-review).


Preprint available: [https://arxiv.org/abs/2303.17835](https://arxiv.org/abs/2303.17835)

## Install dependencies

```
pip install -r requirements.txt
```

## Train the neural network

```
python main.py \
        --train_data "/path/to/train/records/*.tfrecord.GZIP" \
        --val_data "/path/to/val/records/*.tfrecord.GZIP" \
        --no_checkpoints \
        --epochs 50
```

The dataset that is used in the paper can be requested from the author: janne.alatalo(at)jamk.fi


The dataset must be preprocessed with `preprocess_tfrecords.py` script before training the model.

```
python preprocess_tfrecords.py --dataset_stats path/to/stats.json --output_dir /path/to/train/records/ input_file.tfrecord.GZIP
```

## Computing the classifier accuracies

### Use the trained model checkpoint to add predictions to the dataset

```
python generate_change_dataset_with_predictions.py \
        --model_checkpoint "logs/CHECK_THE_CORRECT_DIR/checkpoints/final" \
        --output_dir "/where/you/want/to/store/simulated-change-with-prediction/" \
        "/path/to/val/records/*.tfrecord.GZIP"
```

### Compute the results

For threshold classifier:

```
python threshold_classifier.py  --save_filename minus2_5dB-shift.png --simulated_change_shift -2.5 "/where/you/want/to/store/simulated-change-with-prediction/"
```

And for SVC classifier:

```
python svm_classifier.py "/where/you/want/to/store/simulated-change-with-prediction/"
```

## Simulated change

This repository includes the `generate_change_dataset.py` script that was used
to generate the simulated change dataset for the experiments. However, the
script is too tightly coupled with the database for it to be executable
anywhere. The script requires access to the PostgreSQL database that stores the
SAR image rasters, and the database is too large to be shareable. However, you
can request the dataset that was used to run the experiments. The dataset
includes the simulated changes for the validation samples.

## Citation (preprint version)

```
@misc{alatalo2023improved,
      title={Improved Difference Images for Change Detection Classifiers in SAR Imagery Using Deep Learning}, 
      author={Janne Alatalo and Tuomo Sipola and Mika Rantonen},
      year={2023},
      eprint={2303.17835},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This research was funded by the Regional Council of Central Finland/Council of Tampere Region and European Regional Development Fund as part of the
[*Data for Utilisation -- Leveraging digitalisation through modern artificial intelligence solutions and cybersecurity*](https://www.jamk.fi/en/research-and-development/rdi-projects/data-for-utilisation-leveraging-digitalisation-through-modern-artificial-intelligence-solutions-and) (grant number A76982),
and [*coADDVA - ADDing VAlue by Computing in Manufacturing*](https://www.jamk.fi/en/research-and-development/rdi-projects/coaddva-adding-value-by-computing-in-manufacturing) (grant number A77973) projects of Jamk University of Applied Sciences.

<p>
  <img src="figs/eu-logo.png" height="100" title="Co-funded by the European Union">
  <img src="figs/jamk.png" height="100" title="Jamk University of Applied Sciences">
</p>
