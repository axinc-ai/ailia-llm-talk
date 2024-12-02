import cv2
import numpy as np
import threading
import time

from mic import Microphone
from PIL import ImageFont, ImageDraw, Image
from chat import Chat
from vad import Vad
from s2t import S2T

FULL_SCREEN = False
VAD_ENABLE = True

def split_text(text, font_size, width):
	text_in_one_line = width // font_size * 2
	text_list = [text[i: i + text_in_one_line] for i in range(0, len(text), text_in_one_line)]
	return text_list

def add_text_to_image(draw, font, text, x, y):
	draw.text((x, y), text, (255, 255, 255), font=font)

def text_multiline(draw, font, text, x, y, width):
	text = text.replace("\n", "")
	font_size = 16
	texts = split_text(text, font_size, width)
	for texts in texts:
		add_text_to_image(draw, font, texts, x, y)
		y += (font_size * 1.5)

def ui_process(chat, mic, s2t):
	w = 640
	h = 480

	img = np.zeros((h, w, 3), dtype = np.uint8)
	
	if FULL_SCREEN:
		cv2.namedWindow("ailia LLM Talk", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("ailia LLM Talk", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	frame_shown = False
	frame_cnt = 0

	size = 16
	font = ImageFont.truetype('./font/NotoSansCJKjp-Bold.otf', size)

	while (True):
		ret = True
		if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
			break
		if frame_shown and cv2.getWindowProperty('ailia LLM Talk', cv2.WND_PROP_VISIBLE) == 0:
			break

		x = 0
		y = 240
		img[:] = 0

		transcript = chat.get_transcript()
		answer = chat.get_display_answer()

		if s2t.is_transcripting():
			transcript = "Transcripting..."

		if chat.is_waiting():
			vol = mic.get_volume()
			if VAD_ENABLE :
				conf = vad.get_conf()
			else:
				conf = vol
		else:
			vol = 0
			conf = 0
		
		img_pil = Image.fromarray(img)
		draw = ImageDraw.Draw(img_pil)
		#draw.text((0, 1080 - 32), 'frame ' + str(frame_cnt)+ " vol " + str(vol) + " conf " + str(conf), fill=(255, 255, 255), font=font)

		margin = 32
		if chat.is_waiting():
			img = text_multiline(draw, font, "Waiting your voice.", margin, h // 2 + margin, w - margin * 2)
		img = text_multiline(draw, font, transcript, margin, h // 2 + margin + margin, w - margin * 2)
		img = text_multiline(draw, font, answer, margin, margin, w - margin * 2)

		img = np.array(img_pil)

		r = int(conf * 100)
		oy = 200
		cv2.rectangle(img, (w//2 - r, y + oy + - r), (w//2 + r, y + oy + r), (255, 255, 255), thickness=-1)
		
		frame_cnt = frame_cnt + 1

		cv2.imshow("ailia LLM Talk", img)
		frame_shown = True
		time.sleep(0.01)

	cv2.destroyAllWindows()

chat_thread_terminate = False

def chat_process(chat):
	while (not chat_thread_terminate):
		ret = chat.process()
		if not ret:
			break
		time.sleep(0.01)

mic_thread_terminate = False

def mic_process(mic, vad, s2t, chat):
	while (not mic_thread_terminate):
		pcm = mic.step()
		sample_rate = 44100
		if not chat.is_waiting():
			continue
		if VAD_ENABLE:
			vad.process(pcm, sample_rate)
			data = vad.split()
			sample_rate = 16000
		else:
			data = pcm
		if data is not None:
			transcript = s2t.process(data, sample_rate, VAD_ENABLE)
			if transcript != "":
				chat.set_transcript(transcript)
		time.sleep(0.01)


if __name__ == "__main__":
	import os
	os.makedirs("models", exist_ok=True)

	chat = Chat()
	chat.open()
	
	mic = Microphone()
	mic.start()

	vad = Vad()
	vad.start()

	s2t = S2T()

	thread1 = threading.Thread(target=chat_process, args=(chat, ))
	thread2 = threading.Thread(target=mic_process, args=(mic, vad, s2t, chat))

	thread1.start()
	thread2.start()

	ui_process(chat, mic, s2t)
	
	chat_thread_terminate = True
	mic_thread_terminate = True
	
	thread1.join()
	thread2.join()

	mic.close()
