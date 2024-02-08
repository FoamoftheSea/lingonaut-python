import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pynput import keyboard
import os
import pyaudio
import wave
from tempfile import TemporaryDirectory

import ollama
import torch
import whisper
from TTS.api import TTS

# Get device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


class Recorder:
    def __init__(
        self,
        wavfile,
        chunksize=8192,
        dataformat=pyaudio.paInt16,
        channels=2,
        rate=44100
    ):
        self.filename = wavfile
        self.chunksize = chunksize
        self.dataformat = dataformat
        self.channels = channels
        self.rate = rate
        self.recording = False
        self.pa = pyaudio.PyAudio()

    def start(self):
        # we call start and stop from the keyboard KeyListener, so we use the asynchronous
        # version of pyaudio streaming. The keyboard KeyListener must regain control to
        # begin listening again for the key release.
        if not self.recording:
            self.wf = wave.open(self.filename, 'wb')
            self.wf.setnchannels(self.channels)
            self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
            self.wf.setframerate(self.rate)

            def callback(in_data, frame_count, time_info, status):
                # file write should be able to keep up with audio data stream (about 1378 Kbps)
                self.wf.writeframes(in_data)
                return (in_data, pyaudio.paContinue)

            self.stream = self.pa.open(format=self.dataformat,
                                       channels=self.channels,
                                       rate=self.rate,
                                       input=True,
                                       stream_callback=callback)
            self.stream.start_stream()
            self.recording = True
            print('recording started')

    def stop(self):
        if self.recording:
            self.stream.stop_stream()
            self.stream.close()
            self.wf.close()

            self.recording = False
            print('recording finished')


class KeyListener(keyboard.Listener):
    def __init__(self):
        super().__init__(on_press=self.on_press, on_release=self.on_release)
        self.recorder = None
        self.exit = False
        self.did_record = False
        self.non_english = False
        self.interrupt = False
        self.lock = False  # Allows user to lock controls

    def on_press(self, key):
        if key is None:  # unknown event
            pass
        elif isinstance(key, keyboard.KeyCode):  # alphanumeric key event
            if not self.lock and not self.did_record:
                if key.char == 'q':  # press q to quit
                    if self.recorder.recording:
                        self.did_record = True
                        self.recorder.stop()
                    self.exit = True
                    return False
        elif isinstance(key, keyboard.Key):  # special key event
            if key == key.caps_lock:
                self.lock = not self.lock
                state = "locked" if self.lock else "unlocked"
                print(f"Keys {state}")
            elif not self.lock:
                if not self.did_record:
                    if key in {key.ctrl, key.ctrl_l, key.ctrl_r}:  # and self.player.playing == 0:
                        self.recorder.start()
                        self.non_english = False
                    elif key in {key.shift, key.shift_l, key.shift_r}:
                        self.recorder.start()
                        self.non_english = True
                elif key == key.end:
                    print("Interrupting...")
                    self.interrupt = True  # Interrupts agent response

    def on_release(self, key):
        if key is None:  # unknown event
            pass
        elif not self.lock:
            if isinstance(key, keyboard.Key):  # special key event
                if not self.did_record:
                    if key in {key.ctrl, key.ctrl_l, key.ctrl_r, key.shift, key.shift_l, key.shift_r}:
                        self.exit = True
                        self.did_record = True
                        self.recorder.stop()
            elif isinstance(key, keyboard.KeyCode):  # alphanumeric key event
                pass

    def reset(self):
        self.did_record = False
        self.exit = False


def play_audio(file_path):
    with wave.open(file_path, "rb") as wf:
        p = pyaudio.PyAudio()

        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()


def treat_chunk(chunk):
    treated_chunk = chunk.replace('"', "").replace("(", "").replace(")", "").replace("*", "")

    return treated_chunk


def process_stream(chat_history: list, listener: KeyListener):
    stream = ollama.chat(
        model='mistral:lingonaut',
        messages=chat_history,
        stream=True,
    )
    total_stream = ""

    with ThreadPoolExecutor(max_workers=1) as pool:
        with TemporaryDirectory() as tmp:
            def dump_to_audio(cur_sentence: List[str], language="en"):
                sentence = "".join(cur_sentence)
                tts.tts_to_file(
                    text=sentence,
                    speaker=tts.speakers[10],
                    language=language,
                    file_path=wav_path,
                    split_sentences=False,
                    verbose=False,
                )
                # pool.submit(play_audio, wav_path)
                play_audio(wav_path)

            current_sentence = []
            print("Assistant:")
            for i, chunk in enumerate(stream):
                if listener.interrupt:
                    listener.interrupt = False
                    break
                wav_path = os.path.join(tmp, f"{i}.wav")
                text_chunk = chunk['message']['content']
                print(text_chunk, end="", flush=True)

                text_chunk = treat_chunk(text_chunk)
                if len(text_chunk) == 0:
                    continue
                non_nn_chunk = text_chunk.replace("\n", "")
                # Break text at sentence-ending punctuation marks for smooth offloading to TTS.
                if text_chunk != " " and text_chunk.replace(" ", "")[-1] in [".", "!", "?", ":", "\n"]:
                    if len(current_sentence) > 0 and len(non_nn_chunk) > 0:
                        current_sentence.append(text_chunk)
                    if len(current_sentence) > 30 or (len(current_sentence) > 0 and text_chunk.replace(" ", "").endswith("\n")):
                        total_stream += "".join(current_sentence)
                        pool.submit(dump_to_audio, current_sentence)
                        current_sentence = []
                    continue
                # Otherwise use a hard limit of 50 chunks to avoid overflow
                if len(current_sentence) > 50:
                    total_stream += "".join(current_sentence)
                    pool.submit(dump_to_audio, current_sentence)
                    current_sentence = []
                elif len(non_nn_chunk) > 0:
                    current_sentence.append(non_nn_chunk)

            if len(current_sentence) > 0:
                pool.submit(dump_to_audio, current_sentence)

            pool.shutdown(wait=True)

    return {"role": "assistant", "content": total_stream}


def main():
    chat_history = []
    listener = KeyListener()
    listener.start()  # keyboard KeyListener is a thread so we start it here
    with TemporaryDirectory() as tmp:
        while True:
            input_path = os.path.join(tmp, "user.wav")
            r = Recorder(input_path)
            listener.recorder = r
            print("\nAwaiting user input...")
            while not listener.exit:
                time.sleep(0.1)
            if listener.did_record:
                print("Transcribing user input...")
                model_size = "medium" if listener.non_english else "base"
                whisper_model = whisper.load_model(model_size, device=device)
                result = whisper_model.transcribe(input_path)
                user_input = result["text"]
                del whisper_model
                torch.cuda.empty_cache()
                print("-------------------------------------------------")
                print("User:", user_input)
                print("-------------------------------------------------")
                if len(user_input) == 0:
                    listener.reset()
                    continue
                chat_history.append({'role': 'user', 'content': user_input})
                chat_history.append(process_stream(chat_history, listener))
                print("\n-------------------------------------------------")
                listener.reset()
            else:
                break
    listener.stop()
    listener.join()


if __name__ == "__main__":
    main()
