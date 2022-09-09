import os

path_in = "/raid/fj/kinetics400/compress/val_256"
# path_out = "/raid/fj/kinetics400/compress/val"

# cmd = "ffmpeg -i"
for sub_dir in os.listdir(path_in):
    print('processing '+ sub_dir)
    for root, sub_dirs, sub_videos in os.walk(os.path.join(path_in, sub_dir)):
        for video in sub_videos:
            if video.endswith('.mkv'):
                cmd = "ffmpeg -i" + ' ' + os.path.join(path_in, sub_dir, video) + ' ' + os.path.join(path_in, sub_dir, video[:-4]+'.mp4')
                os.system(cmd)
                os.remove(os.path.join(path_in, sub_dir, video))
            elif video.endswith('webm'):
                cmd = "ffmpeg -i" + ' ' + os.path.join(path_in, sub_dir, video) + ' ' + os.path.join(path_in, sub_dir, video[:-5]+'.mp4')
                os.system(cmd)
                os.remove(os.path.join(path_in, sub_dir, video))