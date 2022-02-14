import ffmpeg
import os


def convert_to_mp4(mkv_file):
    name, ext = os.path.splitext(mkv_file)
    out_name = name + ".mp4"
    ffmpeg.input(mkv_file).output(out_name).run()
    print("Finished converting {}".format(mkv_file))


if __name__ == "__main__":
    input = "/home/chamois/work/CV Hub/Code/peekingduck_familiarization/data/raw/passenger_count.mkv"
    # output = "/home/chamois/work/CV Hub/Code/peekingduck_familiarization/data/raw/passenger_count.mp4"
    convert_to_mp4(input)
