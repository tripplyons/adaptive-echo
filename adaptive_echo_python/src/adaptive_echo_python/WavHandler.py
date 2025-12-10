import wave
import numpy as np
# Writes the data to the file
def writeData(filename: str, data: np.ndarray, sampleRate: int):
    data32bit = data.astype(np.int32)
    with wave.open(filename, 'wb') as f:
        nbrChannels = 1 # mono
        sampleWidth = 4 # 32 bits
        f.setnchannels(nbrChannels)
        f.setsampwidth(sampleWidth)
        f.setframerate(sampleRate)
        f.writeframes(data32bit.tobytes())

def readData(filename: str):
    with wave.open(filename, 'rb') as f:
        sampleRate = f.getframerate()
        nbrFrames = f.getnframes()
        byteData = f.readframes(nbrFrames)
        data = np.frombuffer(byteData,dtype=np.int32)
    return data, sampleRate