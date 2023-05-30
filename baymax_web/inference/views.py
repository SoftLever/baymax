from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import uuid
from csv import writer
from utils import inference, pneumonia_cnn, dataset, plots
import torch
import pandas as pd
from torch.utils.data import DataLoader

def wavUpload(request):
    if request.method == "POST":
        # Get the uploaded file
        fs = FileSystemStorage()
        sound_file = request.FILES['sound_file']

        # Add the new file to our dataset with the appropriate format
        # patient ID will be a UUID
        patient_id = uuid.uuid1()
        fs.save(f"{settings.DATA_PATH}uploaded/{patient_id}_1b1_Al_sc_Meditron.wav", sound_file)

        # Add metadata
        with open(f'{settings.DATA_PATH}uploaded_patient_diagnosis.csv', 'a', newline='') as fo:
            writer_object = writer(fo)
            writer_object.writerow([patient_id,''])

        # Load saved model
        path = f'{settings.BASE_DIR}/inference/saved_model'
        pneumonia_cnn.lung_sound_classifier_model.load_state_dict(torch.load(path))
        pneumonia_cnn.lung_sound_classifier_model.eval()

        # Prepare uploaded data
        df = pd.DataFrame(
            {
                "patient_no": [patient_id],
                "file": [f"{settings.DATA_PATH}/uploaded/{patient_id}_1b1_Al_sc_Meditron.wav"],
                "diagnosis": [0]
            }
        )

        ds = lung_sound_ds = DataLoader(dataset.LungSoundDataset(df),batch_size=1)

        # Infer diagnosis
        diagnosis = inference.inference(pneumonia_cnn.lung_sound_classifier_model, ds)

        print(diagnosis)

        if float(diagnosis) <= 0.5:
            diagnosis = "Healthy"
        else:
            diagnosis = "Pneumonia"

        # Get plots
        for i, d in df.iterrows():
            filepath = f"{settings.BASE_DIR}/static/waveforms/{d['patient_no']}.png"
            plots.createPlots(d, filepath)

        return render(request, 'report.html', {"patient_id": patient_id, "diagnosis": diagnosis, "plot": d['patient_no']})

    return render(request, 'sound_upload.html')
