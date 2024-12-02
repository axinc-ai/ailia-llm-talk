from chain import Chain
from t2s import T2S
	

WEIGHT_PATH = './models/gemma-2-2b-it-Q4_K_M.gguf'
MODEL_PATH = None
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/gemma/'

#INITIAL_PROMPT = "あなたの名前はあいにゃんです。自己紹介をした後、聞きたいことをユーザに質問してください。"
INITIAL_PROMPT = "You are an English teacher. Please answer questions from the user. Please keep your answers to a few lines."

from download import check_and_download_models  # noqa: E402

class Chat():
	chat = None
	chat_cnt = 0
	system_prompt = ""
	chain = None
	chat_history = []
	json = ""
	display_query = ""
	display_answer = ""
	wait_speech = False
	transcript = ""
	first = True

	def open(self):
		check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
		self.chat_cnt = 0
		self.chain = Chain()
		self.t2s = T2S()

	def _fetch(self):
		if self.chat_cnt >= len(self.chat["chatCommand"]):
			return None
		cmd = self.chat["chatCommand"][self.chat_cnt]
		self.chat_cnt = self.chat_cnt + 1
		return cmd

	def _talk(self, query):
		self.display_answer = self.chain.query(query, self.system_prompt, self.chat_history)
		if self.chain.context_full():
			self.chat_history = []
			if self.display_answer == "":
				self.display_answer = self.chain.query(query, self.system_prompt, self.chat_history)
		self.t2s.speech(self.display_answer)
		self.chat_history.append([query, self.display_answer])
	
	def process(self):
		if self.wait_speech:
			if self.transcript != "":
				self.wait_speech = False
				self._talk(self.transcript)
				self.transcript = ""
				self.wait_speech = True
			return True
		if self.first:
			self._talk(INITIAL_PROMPT)
			self.wait_speech = True
			self.first = False
		return True

	def get_display_answer(self):
		return self.display_answer

	def set_transcript(self, transcript):
		self.transcript = transcript
	
	def get_transcript(self):
		return self.transcript
	
	def is_waiting(self):
		return self.wait_speech
		



