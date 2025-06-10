cd Data_raw

# downloading RRN sudokus
wget https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip
unzip sudoku-hard.zip
rm -f sudoku-hard.zip

# downloading 1of many sudokus
mkdir -p one_of_many
cd one_of_many
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12ddIWaL_Xha-Py1ocOg11928Ci9p_8lz' -O sudoku_9_dev_e.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UoIfHdyTSo2_f5i4HYDS4Htacdq8r_-g' -O sudoku_9_test_e_big_amb.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vtR0e-FKqy7XJScxye72D4SbD2jdUOrz' -O sudoku_9_test_e_big_unique.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wx5A199avh-UqzKn0lGAnAP7fhj9eTjp' -O sudoku_9_val_e_big_unique.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=14-IZVopBiMZpXG3YEUkMNJWkxFSCTM1a' -O sudoku_9_train_e.pkl
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12VycMN40HrNtVpxdiFcsyLaRzEX-PpS2' -O sudoku_9_val_e_big_unique.pkl
cd ..

# downloading the SATNet visual Sudoku data set

wget -c powei.tw/sudoku.zip
unzip sudoku.zip
rm -f sudoku.zip
mv sudoku sudoku_satnet
