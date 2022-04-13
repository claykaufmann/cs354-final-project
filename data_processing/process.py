from MidiProcessing import midi_to_array
from mido import MidiFile, MidiTrack, MetaMessage, Message
import matplotlib.pyplot as plt
import numpy as np
import os



directory = '../scrape/songs/'

for filename in os.listdir(directory):
    if filename.endswith(".mid"):
        mid = MidiFile("../songs/0.mid", clip=True)
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
        print(result_array.shape)
        np.save(f"data/{filename}.npy", result_array)