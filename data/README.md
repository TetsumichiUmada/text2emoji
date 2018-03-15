# Datasets

The data is originally downloaded form [DeepMoji GitHub repository](https://github.com/bfelbo/DeepMoji/tree/master/data). Please see the detailed descriptions for more information.

## Data Conversion
Since the original data format is pickle, the file is converted to txt file with the python script, `pickleToTxt.py`. The script is written for the pickle format files provided on the DeepMoji repository. The script is written in python2.

## Usage
To convert a pickle file to export txt file, there are two steps:
#### Change the directory to the file
```
file_name = "PsychExp/raw.pickle"
```
#### Run the script with
```
python pickleToTxt.py > new_file.txt
```
where `new_file.txt` is a file name for txt file.
