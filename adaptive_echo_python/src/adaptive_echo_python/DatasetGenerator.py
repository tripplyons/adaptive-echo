from os import write
from adaptive_echo_python.synth import Synth, synth_parallel
from pathlib import Path
import torch
import numpy as np
from adaptive_echo_python.WavHandler import writeData
import csv

settings_encoder_input_size = Synth().encode_settings().shape[0]
batchSize = 8
settingsScale = 1.5
# Runs on the cpu
synthDevice = torch.device("cpu")
# Number of expected encoder settings
settingsEncoderInputSize = 46
sampleRate = 48000
numSeconds = 5

def generateAllFiles(outputPath, numFiles):
    synth = Synth().to(synthDevice)
    numSamples = int(sampleRate * numSeconds)
    with open(outputPath+"data.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        numFinished = 0
        timeTensor = torch.linspace(0, numSeconds, numSamples, device=synthDevice)
        while numFinished < numFiles:
            # Dont create more files than necessary
            currentBatchSize = min(batchSize, numFiles-numFinished)
            newSettings = settingsScale * torch.randn(
                currentBatchSize, settingsEncoderInputSize, device=synthDevice
            )
            # Normalize generated settings to range with sigmoid
            sig = torch.sigmoid(newSettings)
            # Generate audio
            audioMatrix = synth_parallel(sig, timeTensor)
            for i in range(currentBatchSize):
                curFile = numFinished + i + 1
                # Save row to the csv file
                writer.writerow([f"file{curFile}.wav"] + sig[i].tolist())
                max_int32 = 2**31 - 1
                scaled = np.round(audioMatrix[i].cpu().numpy() * max_int32)
                scaled = np.clip(scaled.astype(np.int64), -max_int32, max_int32)
                curAudio = scaled.astype(np.int32)
                writeData(f"{outputPath}file{curFile}.wav", curAudio, sampleRate)
            numFinished += currentBatchSize
            print(f"Generated {numFinished}/{numFiles} files")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python adaptive_echo.py <num-files> <output-path>")
        sys.exit(-1)
    try:
        numFiles = int(sys.argv[1])
        if numFiles <= 0:
            raise ValueError()
    except ValueError:
        print("Argument 1 must be positive integers")
        sys.exit(-1)

    outputPath = sys.argv[2]
    generateAllFiles(outputPath, numFiles)