import os
from pathlib import Path

import soundcard
import soundfile
import threading

import ailia_voice
import librosa
import time

import re
import os

import ailia

from langdetect import detect

def check_is_english(text):
    try:
        return detect(text) == 'en'
    except Exception as e:
        print("Error:", e)
        return False
	
voice_queue = []
voice_samplerate = 0

def speaker_process():
	default_speaker = soundcard.default_speaker()
	while True:
		if len(voice_queue) > 0:
			with default_speaker.player(samplerate=voice_samplerate) as sp:
				data, _ = soundfile.read(voice_queue[0])
				sp.play(data)
				voice_queue.pop(0)
		time.sleep(0.01)

speaker_thread = threading.Thread(target=speaker_process, args=())
speaker_thread.start()

class T2S():
	first = True
	voice = None
	avatar_changed = True
	cnt = 0
	
	def __init__(self):
		self.init_ailia_voice()

	def init_ailia_voice(self):
		if self.first:
			env_list = ailia.get_environment_list()
			env_id = -1
			for env in env_list:
				if "cuDNN" in env.name and not "FP16" in env.name:
					print("GPU Selected", env)
					env_id = env.id
			if env_id == -1:
				print("GPU not found. We will use CPU.")
			self.voice = ailia_voice.GPTSoVITS(env_id = env_id)
			self.voice.initialize_model(model_path = "./models/")
			self.first = False

	def split_text_by_punctuation(self, text, is_english):
		# 句読点でテキストを分割
		if is_english:
			sentences = re.split(r'(?<=[,.!?])', text)
		else:
			sentences = re.split(r'(?<=[、。！？])', text)
		return [sentence.strip() for sentence in sentences if sentence]

	def speech(self, text):
		dir_name = "./chat_verbally/audio/"
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)

		is_english = check_is_english(text)

		text_list = self.split_text_by_punctuation(text, is_english)
		for text in text_list:
			speech_file_name = "speech"+str(self.cnt)+".mp3"
			speech_file_path = Path(dir_name + speech_file_name)

			reference = "水をマレーシアから買わなくてはならない。"
			samplerate = self.speech_ailia(text, speech_file_path, "reference_audio_captured_by_ax.wav", reference, is_english)
			
			if samplerate == None:
				continue

			global voice_queue, voice_samplerate
			voice_samplerate = samplerate
			voice_queue.append(speech_file_path)
			self.cnt = self.cnt + 1

		while True:
			if len(voice_queue) <= 0:
				break
			time.sleep(0.01)

	def speech_ailia(self, text, speech_file_path, ref_file_path, ref_text, is_english):
		# Load reference audio
		audio_waveform, sampling_rate = librosa.load(ref_file_path, mono=True)

		# Infer
		self.init_ailia_voice()
		if self.avatar_changed:
			self.voice.set_reference_audio(ref_text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA, audio_waveform, sampling_rate)
			self.avatar_changed = False

		if is_english:
			buf, sampling_rate = self.voice.synthesize_voice(text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_EN)
		else:
			buf, sampling_rate = self.voice.synthesize_voice(text, ailia_voice.AILIA_VOICE_G2P_TYPE_GPT_SOVITS_JA)
		if buf is None:
			return None

		# Save result
		soundfile.write(speech_file_path, buf, sampling_rate)
		return sampling_rate
