""" 
This script filters the given table which contains MRI metadata for given criterias and saves the results in csv files.

METADATE FROM THE TABLE: 
Label = 0, no lesions. 
label = 1, lesions.
Data quality: 3D -> pixel dims < 1.5
same scanner Scanner_Manufacturer 
lesion masks vorhanden? (Depends on dataset) in ['ADNI', 'OASIS-3', 'BTH'] are no lesion masks, rest of data should have them (but double-check!)

WHAT WE NEED: 
Train GAN1 and Train GAN2 can be overlapping datasets: 
MRI with white matter lesion + lesion mask.
for GAN2 Siemens scanner, for GAN1 dosen't matter, 
also für GAN1 evtl. gleiche Daten + die aus anderen Scannern. 

Test Daten: 
MRI without white matter lesions, "healthy". Siemens Scanner.
"""

import numpy as np 
import pandas as pd

# Scanners we want: 
scanners = ['SIEMENS', 'Siemens']
#scanners = ['Philips','Philips Medical Systems','Philips Healthcare']

save_datasets_to_csv = False



with open("experiments/select_data/MRIs_List_Final_new.csv") as file: 
    df = pd.read_csv(file, header=0, sep=(","))
    
    df = df.astype({
        'pixdim1_mm': float, 
        'pixdim2_mm': float,
        'pixdim3_mm': float
    })
    print(df.head(22))
    print(df.columns)

    print("WIe viele UMCL patienten sind gelistet?",df.query("'UMCL' in Dataset").shape)


    # for all 3 datasets we want nice quality / pixel resolution: 
    df = df.query("`pixdim1_mm` < 1.5 and `pixdim2_mm` < 1.5 and `pixdim3_mm` < 1.5")

    print("All datasets: ", df.Dataset.unique())

    print("Alle Firmen : ", df["Scanner_Manufacturer"].unique())

    print("how many oasis imgs?", df.query("`Dataset` == 'OASIS-3'").shape)
    print("how many oasis imgs with lesions?", df.query("`Dataset` == 'OASIS-3' and `Label` == 1").shape)
    print("how many oasis imgs without lesions?", df.query("`Dataset` == 'OASIS-3' and `Label` == 0").shape)
    # erstes Dataset: MS MRI mit Lesion mask, Scanner egal 
    train_gan1_df = df.copy()

    # with lesions
    train_gan1_df = train_gan1_df.query("`Label` == 1")

    # with lesionmasks (TODO CHECK IF TRUE ! )
    train_gan1_df = train_gan1_df.query("`Dataset` not in ['ADNI', 'OASIS-3', 'BTH']")
    
    print("Datasets in train_gan1 : ", train_gan1_df.Dataset.unique())
    # zweites Dataset: same as first, but Scanner = Siemens (subset of first dataset)
    train_gan2_df = train_gan1_df.copy()
    train_gan2_df = train_gan1_df.query("`Scanner_Manufacturer` in @scanners")

    # drittes Dataset: MRI without white matter lesions, Scanner = Siemens

    
    test_pipeline_df = df.copy().query("`Scanner_Manufacturer` in @scanners")
    test_pipeline_df = test_pipeline_df.query("Label == 0")
    
    # only Siemens: 
    #second_df = second_df[second_df['Protocol_Group'].str.startswith('Sie')]

    # second_df braucht keine Lesion-masks ? und soll Gesund sein ? -> ja, also label = 0
    #print(second_df)
    print("Results: ")
    print("all quality data :", df.shape)
    print("train_gan1_df : ", train_gan1_df.shape)
    print("train_gan2_df :", train_gan2_df.shape)
    print("test_pipeline_df: ", test_pipeline_df.shape)

    if save_datasets_to_csv: 
        datasets = [train_gan1_df, train_gan2_df, test_pipeline_df]
        filenames = ["train_gan1.csv", "train_gan2.csv", "test_pipeline.csv"]  # Define unique filenames for each dataset

        for dataset, filename in zip(datasets, filenames): 
            dataset.to_csv(filename, index=False) 
        

    print("end")
    print(df.Dataset.unique())
    print("WIe viele UMCL patienten sind gelistet?",df.query("'UMCL' in Dataset").shape)

