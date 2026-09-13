import sys
import types
from pathlib import Path

# lingonaut.py loads the XTTS model when it's imported. Stub out the TTS package so tests
# can import the module without downloading or loading the model.
tts_api = types.ModuleType("TTS.api")


class StubTTS:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self


tts_api.TTS = StubTTS
sys.modules["TTS"] = types.ModuleType("TTS")
sys.modules["TTS.api"] = tts_api

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
