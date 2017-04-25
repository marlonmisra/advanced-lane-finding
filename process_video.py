from process_image import *
from image_gen import *
from moviepy.editor import *
from IPython.display import HTML

output_name = "video_annotated_2.mp4"
input_file = VideoFileClip("harder_challenge_video.mp4")
output_clip = input_file.fl_image(process_image)
output_clip.write_videofile(output_name, audio=False)