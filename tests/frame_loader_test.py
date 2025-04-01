import unittest
from preprocessing import video_processing as prep

class TestFrameLoader(unittest.TestCase):

    def test_frame_loader_with_invalid_video_path(self):
        result = prep.load_frames_from_mp4("../data/invalid.mp4")
        self.assertEqual(len(result), 0)

    def test_frame_loader_with_valid_video_path(self):
        result = prep.load_frames_from_mp4("../data/test.mp4")
        self.assertEqual(len(result), 67)