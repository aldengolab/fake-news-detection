# Model learning pipeline

This code implements the model pipeline. 

## Files

- `model.py`: a class for models
- `model_loop.py`: a class for running a loop of classifiers; takes test-train data splits and various run params
- `run.py`: this code implements the model_loop class; it also implements our re-sampling of the data
- `run.sh`: this code runs the full model pipeline; it contains the specifications to alter the operation of the code
- `transform_features.py`: this file executes all feature generation

## To Run
Make any desired edits to the `run.sh` file and execute the following in bash:

        chmod +x run.sh
        ./run.sh
        

