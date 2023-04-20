import librosa

audio, sr = librosa.load('sample_audio\XC247572.ogg')
print(audio.shape)