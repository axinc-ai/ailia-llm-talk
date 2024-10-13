import ailia
import ailia.audio
import sys
import numpy as np

WEIGHT_PATH = './models/silero_vad.onnx'
MODEL_PATH = './models/silero_vad.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/silero-vad/'

from download import check_and_download_models  # noqa: E402

class Vad():
	buf = np.zeros((0, ))
	processed_buf = np.zeros((0, ))
	vad_buf = np.zeros((0, ))

	def start(self):
		check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
		env_id = -1
		self.session = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)
		self.reset_states()

	def reset_states(self, batch_size=1):
		self._h = np.zeros((2, batch_size, 64)).astype('float32')
		self._c = np.zeros((2, batch_size, 64)).astype('float32')

	def process(self, pcm, sample_rate):
		steps = 1536

		# オーバラップのノイズを抑制するために元空間でリサンプル
		self.buf = np.concatenate([self.buf, pcm])
		input = ailia.audio.resample(self.buf, sample_rate, 16000)
		cnt = input.shape[0] // steps
		if cnt < 2:
			return
		tap_margin = 1 # resampleのタップを考慮したマージン
		input = input[0:(cnt - tap_margin) * steps]
		self.buf = self.buf[int((cnt - tap_margin) * steps * sample_rate // 16000):]
		
		# 処理バッファに追加
		x = np.array(input)
		#print(input.shape)
		x = np.expand_dims(x, axis=0)
		sr = 16000
		ort_inputs = {'input': x, 'h': self._h, 'c': self._c, 'sr': np.array(sr, dtype='int64')}
		ort_outs = self.session.run(ort_inputs)
		out, self._h, self._c = ort_outs

		self.processed_buf = np.concatenate([self.processed_buf, input])
		self.vad_buf = np.concatenate([self.vad_buf, out[0]])

	def get_conf(self):
		if self.vad_buf.shape[0] > 0:
			return self.vad_buf[-1]
		return 0

	def split(self):
		ACTIVE_SEC = 1.0
		SILENT_SEC = 1.0

		STATE_EMPTY = 0
		STATE_ACTIVE = 1
		STATE_SILENT = 2
		STATE_FINISH = 3
		
		threshold = 0.25

		active_cnt = 0
		silent_cnt = 0
		start_i = 0
		end_i = 0
		state = STATE_EMPTY
		steps = 1536
		for i in range(self.vad_buf.shape[0]):
			if (state == STATE_EMPTY):
				if (self.vad_buf[i] > threshold):
					start_i = i
					active_cnt = active_cnt + 1
					state = STATE_ACTIVE
				continue
			if (state == STATE_ACTIVE):
				if (self.vad_buf[i] > threshold):
					active_cnt = active_cnt + 1
				else:
					if (active_cnt >= ACTIVE_SEC * 16000 / steps):
						state = STATE_SILENT
						silent_cnt = 0
					else:
						state = STATE_EMPTY
				continue
			if (state == STATE_SILENT):
				if (self.vad_buf[i] > threshold):
					state = STATE_ACTIVE
				else:
					silent_cnt = silent_cnt + 1
					if (silent_cnt >= SILENT_SEC * 16000 / steps):
						state = STATE_FINISH
						end_i = i
						break
		if (state == STATE_FINISH):
			result = self.processed_buf[start_i * steps:(end_i + 1) * steps]
			self.vad_buf = self.vad_buf[end_i + 1:]
			self.processed_buf = self.processed_buf[(end_i + 1) * steps:]
			return result
		return None

	def close(self):
		self.session.close()
	