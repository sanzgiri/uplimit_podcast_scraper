import modal

def download_whisper():
  # Load the Whisper model
  import os
  import whisper
  print ("Download the Whisper model")

  # Perform download only once and save to Container storage
  whisper._download(whisper._MODELS["medium"], '/content/youtube/', False)


stub = modal.Stub("corise-youtube-project")
corise_image = modal.Image.debian_slim().pip_install("yt-dlp",
                                                     "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "ffmpeg-python").apt_install("ffmpeg").run_function(download_whisper)

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
  import whisper

  # Load model from saved location
  print ("Load the Whisper model")
  model = whisper.load_model('medium', device='cuda', download_root='/content/youtube/')

  # Perform the transcription
  print ("Starting Youtube video transcription")
  result = model.transcribe(local_path + filename)

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

