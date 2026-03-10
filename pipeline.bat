echo Starting ML Pipeline

echo Step 1: Data Creation
python data_creation.py

echo Step 2: Data Preprocessing
python model_preprocessing.py

echo Step 3: Model Training
python model_preparation.py

echo Step 4: Model Testing
python model_testing.py

echo Pipeline finished
pause