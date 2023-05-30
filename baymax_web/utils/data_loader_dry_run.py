import csv
import os
import pandas as pdf

# Declare file path variables
data_path = f"/home/collins/Desktop/projects/baymax/baymax_web/data/Respiratory_Sound_Database/"
audio_file_path = f"{data_path}audio_and_txt_files"
# demographics_file = f"{data_path}/demographic_info.csv"
# events_path = f"{data_path}events"
diagnosis_file = f"{data_path}patient_diagnosis.csv"


df = pdf.read_csv(diagnosis_file, header=None, names=["patient_no", "diagnosis"])
# df = df[df['diagnosis'].isin(["Healthy", "Pneumonia"])] # Only deal with pneumonia and healthy


# df2 = pdf.read_csv(demographics_file, header=None, names=["patient_no", "age", "sex", "adult_bmi", "child_weight", "child_height"])
# df3 = pdf.merge(df, df2, on="patient_no")

# Add rows for each chest location
df["file"] = None
df.loc[df['diagnosis'] != 'Pneumonia', 'diagnosis'] = 0
df.loc[df['diagnosis'] == 'Pneumonia', 'diagnosis'] = 1
# df3["Tc"] = None
# df3["Al"] = None
# df3["Pl"] = None
# df3["Ll"] = None
# df3["Ar"] = None
# df3["Pr"] = None
# df3["Lr"] = None

for d in os.listdir(audio_file_path):
    p = d.split("_")

    if p[4].endswith("wav"):
        # df3.loc[df3['patient_no'] == int(p[0]), p[2]] = f"{audio_file_path}/{d}"
        df.loc[df['patient_no'] == int(p[0]), 'file'] = f"{audio_file_path}/{d}"
