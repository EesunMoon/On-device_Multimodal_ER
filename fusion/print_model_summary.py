from keras.models import load_model


# hr_model_shallow = load_model('./model/HR_model_shallow.hdf5')
# hr_model_deep = load_model('./model/HR_model_deep.hdf5')
# hr_model_shallow.summary()
# hr_model_deep.summary()


# eeg_model_shallow = load_model('./model/EEG_model_shallow.hdf5')
# eeg_model_deep = load_model('./model/EEG_model_deep.hdf5')
# eeg_model_shallow.summary()
# eeg_model_deep.summary()


speech_model_shallow = load_model('./model/Speech_model_shallow.hdf5')
speech_model_deep = load_model('./model/Speech_model_deep.hdf5')
speech_model_shallow.summary()
speech_model_deep.summary()


# video_model_shallow = load_model('./model/Video_model_shallow.hdf5')
# video_model_deep = load_model('./model/Video_model_deep.hdf5')
# video_model_shallow.summary()
# video_model_deep.summary()
