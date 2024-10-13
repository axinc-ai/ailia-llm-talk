import pyaudio
import wave
import audioop
import numpy as np

class Microphone():
	volume = 0

	def start(self, sample_rate=44100, chunk_size=1024, channels=1):
		audio_format = pyaudio.paInt16
		self.p = pyaudio.PyAudio()
		print(self.p.get_default_input_device_info())

		self.stream = self.p.open(format=audio_format,
						channels=channels,
						rate=sample_rate,
						input=True,
						frames_per_buffer=chunk_size)
		self.frames = []
		self.chunk_size = chunk_size

	def step(self):
		data = self.stream.read(self.chunk_size, exception_on_overflow = False)
		frame = np.fromstring(data, dtype=np.int16)

		frame = frame.astype(np.float32) / 32767.0
		self.volume = np.max(frame)

		return frame

	def get_volume(self):
		return self.volume

	def close(self):
		self.stream.stop_stream()
		self.stream.close()
		self.p.terminate()

