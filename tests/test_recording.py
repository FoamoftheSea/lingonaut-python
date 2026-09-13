import pytest
from pynput import keyboard

import lingonaut


class FakePyAudio:
    def __init__(self, device=None):
        self.device = device

    def get_default_input_device_info(self):
        if self.device is None:
            raise IOError("No Default Input Device Available")
        return self.device


class FakeRecorder:
    def __init__(self):
        self.recording = False

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False


class FakeListener:
    def __init__(self, alive, exit=False, error=None):
        self.alive = alive
        self.exit = exit
        self.error = error

    def is_alive(self):
        return self.alive

    def join(self):
        if self.error is not None:
            raise self.error


def test_input_device_check_accepts_mono_mic(capsys):
    lingonaut.check_input_device(FakePyAudio({"name": "Mono Mic", "maxInputChannels": 1}), channels=1)
    assert 'Microphone: "Mono Mic", recording 1 of 1 channel(s).' in capsys.readouterr().out


def test_input_device_check_tells_stereo_users_how_to_record_more(capsys):
    lingonaut.check_input_device(FakePyAudio({"name": "Stereo Mic", "maxInputChannels": 2}), channels=1)
    output = capsys.readouterr().out
    assert "recording 1 of 2 channel(s)" in output
    assert "INPUT_CHANNELS" in output


def test_input_device_check_rejects_more_channels_than_the_mic_has():
    with pytest.raises(SystemExit, match="has 1 input channel"):
        lingonaut.check_input_device(FakePyAudio({"name": "Mono Mic", "maxInputChannels": 1}), channels=2)


def test_input_device_check_explains_missing_microphone():
    with pytest.raises(SystemExit, match="No microphone found"):
        lingonaut.check_input_device(FakePyAudio(device=None))


def test_key_press_before_recorder_is_ready_warns_instead_of_crashing(capsys):
    listener = lingonaut.KeyListener()
    listener.on_press(keyboard.Key.ctrl)
    assert "Not ready to record yet" in capsys.readouterr().out
    assert not listener.did_record


def test_not_ready_warning_prints_once_per_key_hold(capsys):
    listener = lingonaut.KeyListener()
    listener.on_press(keyboard.Key.ctrl)
    listener.on_press(keyboard.Key.ctrl)  # held keys can repeat press events
    assert capsys.readouterr().out.count("Not ready to record yet") == 1

    listener.on_release(keyboard.Key.ctrl)
    listener.on_press(keyboard.Key.ctrl)
    assert capsys.readouterr().out.count("Not ready to record yet") == 1


def test_key_release_without_a_recording_does_not_start_transcription():
    listener = lingonaut.KeyListener()
    listener.on_release(keyboard.Key.ctrl)
    assert not listener.exit and not listener.did_record

    listener.recorder = FakeRecorder()
    listener.on_release(keyboard.Key.shift)
    assert not listener.exit and not listener.did_record


def test_holding_shift_records_non_english_input():
    listener = lingonaut.KeyListener()
    listener.recorder = FakeRecorder()
    listener.on_press(keyboard.Key.shift)
    assert listener.recorder.recording and listener.non_english

    listener.on_release(keyboard.Key.shift)
    assert listener.exit and listener.did_record
    assert not listener.recorder.recording


def test_quit_works_before_recorder_is_ready():
    listener = lingonaut.KeyListener()
    # pynput wraps callbacks and turns a False return into StopException, which stops the listener.
    with pytest.raises(keyboard.Listener.StopException):
        listener.on_press(keyboard.KeyCode.from_char("q"))
    assert listener.exit and not listener.did_record


def test_crashed_listener_raises_with_explanation_and_original_error():
    crash = AttributeError("'NoneType' object has no attribute 'start'")
    with pytest.raises(RuntimeError, match="keyboard listener stopped") as excinfo:
        lingonaut.ensure_listener_alive(FakeListener(alive=False, error=crash))
    assert excinfo.value.__cause__ is crash


def test_listener_stopped_by_quit_is_not_an_error():
    lingonaut.ensure_listener_alive(FakeListener(alive=False, exit=True))


def test_running_listener_is_left_alone():
    lingonaut.ensure_listener_alive(FakeListener(alive=True))
