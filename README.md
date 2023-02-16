# Neural Network-Based Mapping Transformation Function for Improved Change Detection from SAR Satellite Images

This repository includes code for the paper **Neural Network-Based Mapping Transformation Function for Improved Change Detection from SAR Satellite Images**.

## Install dependencies

```
pip install -r requirements.txt
```

## Train the neural network

```
python main.py \
        --train_data "/path/to/train/records/*.tfrecord.GZIP" \
        --val_data "/path/to/val/records/*.tfrecord.GZIP" \
        --mixed_precision \
        --no_checkpoints \
        --epochs 50
```

The dataset that is used in the paper can be requested from the author: janne.alatalo(at)jamk.fi

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

## Acknowledgements

This project was developed in the [Data for Utilisation - Leveraging digitalisation through modern artificial intelligence solutions and cybersecurity project of Jamk University of Applied Sciences](https://www.jamk.fi/fi/tutkimus-ja-kehitys/tki-projektit/tieto-tuottamaan-digitalisaation-hyodyntaminen-modernien-tekoalyratkaisujen-ja-kyberturvallisuuden)
