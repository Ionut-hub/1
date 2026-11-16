import subprocess

def assemble_video(audio_file, video_file, output_file):
    command = [
        "ffmpeg",
        "-i", audio_file,
        "-i", video_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_file
    ]
    subprocess.run(command)
    print(f"Video assembled and saved to {output_file}")