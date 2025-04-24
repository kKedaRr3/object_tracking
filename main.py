from models.unsupervised_tracking import object_tracking
from preprocessing import video_loader

video = video_loader.load_frames_from_mp4('../data/spoon.mp4')[1:]

object_tracking(video, "../results/spoon/tracked_spoon.mp4", 50, 3)
