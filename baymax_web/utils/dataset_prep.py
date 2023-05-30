# patient_number = "122" # Those with pneumonia in the dataset are 122, 135, 140, 191, 219 and 226
# sound_location = ""

# Colors


# def remove_spikes(data):
#     # Calculate the median absolute deviation (MAD) of the signal
#     # mad = math.floor(median_abs_deviation(data))

#     # Determine the window size based on the MAD
#     # ws = mad * 10
    
#     # Ensure window size is odd
#     ws = 501 # ws if ws % 2 else ws + 1
    
#     # print(ws)
    
#     filtered_signal = medfilt(data)
    
# #     b, a = butter(3, [50/(0.5*sampling_rate), 2500/(0.5*sampling_rate)], 'band')    
# #     data = lfilter(b, a, data)

#     return filtered_signal


