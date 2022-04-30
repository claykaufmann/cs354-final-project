from MidiProcessing import midi_to_array
from mido import MidiFile, MidiTrack, MetaMessage, Message
import matplotlib.pyplot as plt
import numpy as np
import os


directory = "data/raw/"

for filename in os.listdir(directory):
    if filename.endswith(".mid"):
        mid = MidiFile(directory + filename, clip=True)
        result_array = midi_to_array(mid)
        # plt.plot(
        #     range(result_array.shape[0]),
        #     np.multiply(np.where(result_array > 0, 1, 0), range(1, 89)),
        #     marker=".",
        #     markersize=1,
        #     linestyle="",
        # )
        # plt.title("nocturne_27_2_(c)inoue.mid")
        # plt.show()

        # print(result_array)

        # resize the array to the base size of 5000 features to start
        new_array = result_array[:5000]
        print(new_array.shape)

        np.save(f"data/processed/{filename}.npy", new_array)
