import os
import json
import ailia_llm


class Chain():
	stream_queue = None
	model = "gemma2"
	first = True

	def query(self, query, system_prompt, chat_history):
		return self.query_ailia(query, system_prompt, chat_history)

	def query_ailia(self, query, system_prompt, chat_history):
		print(query)

		if self.first:
			self.llm = ailia_llm.AiliaLLM()
			self.llm.open("./models/gemma-2-2b-it-Q4_K_M.gguf", n_ctx = 8192)
			self.first = False

		messages = []
		if system_prompt is not None:
			messages.append({"role": "system", "content": system_prompt})
		for history in chat_history:
			messages.append({"role": "user", "content": history[0]})
			messages.append({"role": "assistant", "content": history[1]})
		
		messages.append({"role": "user", "content": query})

		stream = self.llm.generate(messages)
		
		answer = ""
		for deltaText in stream:
			answer += deltaText
			if self.stream_queue != None:
				self.stream_queue.put(deltaText)
		return answer

	def context_full(self):
		return self.llm.context_full()
