import modal

def download_whisperX():
  # Load the WhisperX model
  import os
  import whisperx
  print ("Download the WhisperX model")

  # Parameters for whisperX
  device = "cpu"
  compute_type = "float32"

  # Perform download only once and save to Container storage
  _ = whisperx.load_model("medium", device, compute_type=compute_type)

stub = modal.Stub("corise-youtube-project")
corise_image = modal.Image.debian_slim().pip_install("yt-dlp",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "ffmpeg-python").apt_install("git", "ffmpeg")

corise_image = corise_image.pip_install("torch",
                                        "torchvision",
                                        "torchaudio",
                                        index_url="https://download.pytorch.org/whl/cu118")

corise_image = corise_image.pip_install("git+https://github.com/m-bain/whisperx.git"
                                        ).run_function(download_whisperX)

@stub.function(image=corise_image, gpu="any")
def get_transcribe_youtube_video(youtube_url, local_path):
  print ("Starting Youtube Transcription Function")
  print ("Feed URL: ", youtube_url)
  print ("Local Path:", local_path)

  # Download the youtube video 
  from pathlib import Path
  p = Path(local_path)
  p.mkdir(exist_ok=True)

  print ("Downloading the youtube video")
  import os
  filename = "ytv_audio.mp3"
  cmd = f"yt-dlp -x --audio-format mp3 -o {local_path + filename} {youtube_url}"
  os.system(cmd)
  print (f"Youtube video downloaded to {local_path + filename}")

  # Load the Whisper model
  import os
  import whisperx

  # Parameters for whisperX
  device = "cuda"
  batch_size = 32 # reduce if low on GPU mem
  compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

  # Load model from saved location
  print ("Load the Whisper model")
  model = whisperx.load_model("medium", device=device, compute_type=compute_type)

  # Get the path for audio file
  audio = whisperx.load_audio(local_path + filename)

  # Perform the transcription
  print ("Starting Youtube video transcription")
  result = model.transcribe(audio, batch_size=batch_size)

  # Combine result text
  result["text"] = ""
  for segments in result["segments"]:
    result["text"] = result["text"] + segments["text"]

  # Return the transcribed text
  print ("Youtube video transcription completed, returning results...")
  output = {}
  # extract additional metadata from the youtube video 
 # output['video_title'] = youtube_title
  output['audio_transcript'] = result['text']
  return output
  

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_youtube_video_summary(audio_transcript):
    import openai
    instructPrompt = """
    Summarize the audio transcript that follows:
    """
    request = instructPrompt + audio_transcript
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
    youtubeVideoSummary = chatOutput.choices[0].message.content
    return youtubeVideoSummary


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_youtube_video_qa(audio_transcript):
    import openai
    instructPrompt = """
    Generate 5 question and answer pairs from the audio transcript that follows:
    """
    request = instructPrompt + audio_transcript
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
    youtubeVideoQA = chatOutput.choices[0].message.content
    return youtubeVideoQA


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_youtube_video_mqa(audio_transcript):
    import openai
    instructPrompt = """
    Generate 5 multiple choice questions from the audio transcript that follows. For each question, indicate four possible answer options, named A, B, C, D of which only should one be correct. Provide the correct answer option as well
    """
    request = instructPrompt + audio_transcript
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
    youtubeVideoMQA = chatOutput.choices[0].message.content
    return youtubeVideoMQA


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_youtube_video_highlights(audio_transcript):
    import openai
    instructPrompt = """Extract some highlights from the audio transcript that follows. These could be interesting insights from the guest or critical questions from the youtube host. It could also be a discussion on a hot topic or a controversial opinion."""

    request = instructPrompt + audio_transcript
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
    youtubeVideoHighlights = chatOutput.choices[0].message.content
    return youtubeVideoHighlights

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_youtube(url, path):
    output = {}
    youtube_video_details = get_transcribe_youtube_video.remote(url, path)
    youtube_video_summary = get_youtube_video_summary.remote(youtube_video_details['audio_transcript'])
    youtube_video_highlights = get_youtube_video_highlights.remote(youtube_video_details['audio_transcript'])
    output['youtube_video_details'] = youtube_video_details
    output['youtube_video_summary'] = youtube_video_summary
    output['youtube_video_highlights'] = youtube_video_highlights
    return output

@stub.local_entrypoint()
def main(url, path):
    youtube_video_details = get_transcribe_youtube_video.remote(url, path)
    print ("\nYoutube Video Summary:\n", get_youtube_video_summary.remote(youtube_video_details['audio_transcript']))
    print ("\nYoutube Video Highlights:\n", get_youtube_video_highlights.remote(youtube_video_details['audio_transcript']))
#    print ("\nYoutube Video QA:\n", get_youtube_video_qa.remote(youtube_video_details['audio_transcript']))
    print ("\nYoutube Video MQA:\n", get_youtube_video_mqa.remote(youtube_video_details['audio_transcript']))

