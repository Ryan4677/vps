import json

def ms_to_srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1000
    millis = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def json_to_srt(data: list, output_file="output.srt"):
    lines = []
    for i, item in enumerate(data, start=1):
        start = ms_to_srt_time(item["start"])
        end = ms_to_srt_time(item["end"])
        text = item["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


with open("output/4c24febd-247f-400a-aba6-4b1fc36fd488/metadata/metadata.json", "r") as f:
  sub = json.loads(f.read())
  data = []
  for i in sub.get("transcript"):
    start = i.get("start")
    stop = i.get("stop")
    text = i.get("text")
    g = {"start": start, "end": stop, "text": text}
    data.append(g)
  json_to_srt(data)