from openai import OpenAI
import os
import numpy as np
import time
import ailia_speech

class S2T():
    transcript = ""
    first = True

    def __init__(self):
        self.init_ailia_speech()

    def init_ailia_speech(self):
        if self.first:
            self.speech = ailia_speech.Whisper(callback=self.callback)
            self.speech.initialize_model(model_path = "./models/", model_type = ailia_speech.AILIA_SPEECH_MODEL_TYPE_WHISPER_MULTILINGUAL_LARGE_V3_TURBO)
            self.speech.set_silent_threshold(silent_threshold = 0.25, speech_sec = 1.0, no_speech_sec = 0.5)
            self.first = False

    def process(self, buf, sample_rate, vad_enable):
        if vad_enable:
            self.transcript = "transcripting..."
        transcript = self.whisper_ailia(buf, sample_rate, vad_enable)
        if transcript != "":
            self.transcript = transcript
        return transcript
    
    def callback(self, text):
        print(text)

    def whisper_ailia(self, buf, sample_rate, vad_enable):
        self.init_ailia_speech()

        if vad_enable:
            recognized_text = self.speech.transcribe(buf, sample_rate)
        else:
            complete = False
            recognized_text = self.speech.transcribe_step(buf, sample_rate, complete, lang="ja")
        transcript = ""
        for text in recognized_text:
            transcript = transcript + text["text"]
        return transcript

    def get_transcript(self):
        return self.transcript

