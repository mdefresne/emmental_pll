echo "Installing python dependencies"
uv sync

# downloading 10 train grids and full test of HRM extreme grids (https://github.com/sapientinc/HRM/)
# 128 grids at the end of the test set are used for validation. Only 10000 grids are used as test.
echo "Downloading HRM dataset"
uv run Scripts/generate_HRM_sudoku_dataset.py --output-dir Data_raw/sudoku-extreme-10-aug-99  --subsample-size 10 --num-aug 99
tail -n 128 Data_raw/sudoku-extreme-10-aug-99/test.csv > Data_raw/sudoku-extreme-10-aug-99/valid.csv

cd Data_raw

echo "Downloading RRN dataset"
wget https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip
unzip sudoku-hard.zip
rm -f sudoku-hard.zip

echo "Downloading 1of many dataset"
mkdir -p one_of_many
cd one_of_many
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12ddIWaL_Xha-Py1ocOg11928Ci9p_8lz' -O sudoku_9_dev_e.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UoIfHdyTSo2_f5i4HYDS4Htacdq8r_-g' -O sudoku_9_test_e_big_amb.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vtR0e-FKqy7XJScxye72D4SbD2jdUOrz' -O sudoku_9_test_e_big_unique.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wx5A199avh-UqzKn0lGAnAP7fhj9eTjp' -O sudoku_9_val_e_big_unique.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14-IZVopBiMZpXG3YEUkMNJWkxFSCTM1a' -O sudoku_9_train_e.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12VycMN40HrNtVpxdiFcsyLaRzEX-PpS2' -O sudoku_9_val_e_big_unique.pkl
cd ..

echo "Downloading SATNet visual Sudoku dataset"
wget -c powei.tw/sudoku.zip
unzip sudoku.zip
rm -f sudoku.zip
mv sudoku sudoku_satnet

