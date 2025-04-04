from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
scene_list = detect('media/pulpfiction_trailer.mp4', AdaptiveDetector())
split_video_ffmpeg('media/pulpfiction_trailer.mp4', scene_list)
