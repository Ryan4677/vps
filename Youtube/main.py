from pathlib import (
    Path,
)
from yt import (
    video,
    transcript,
    channel,
)
from uuid import (
    uuid4,
)
from editing import (
    EditingWrapper,
)
import os
import json

yt_url = "https://youtu.be/a7je2XuoK2A"
output_path = "./output/"

TES_SCRIPT = True
PATH_UUID = "4c24febd-247f-400a-aba6-4b1fc36fd488"
EDITING = True
DOWNLOAD_VIDEO = False
TRANSCRIPT = False

def video_transcript(
    path,
):
    from gemini import (
        content,
    )

    file_path = Path(
        f"{path}/metadata"
    )
    file_path.mkdir(
        parents=False,
        exist_ok=True,
    )
    raw_transcript = json.loads(
        transcript(
            yt_url
        )
    )
    metadata = json.loads(
        channel(
            yt_url
        )
    )
    data = {
        "channel": metadata[
            "name"
        ],
        "channelid": metadata[
            "id"
        ],
        "title": metadata[
            "title"
        ],
        "source": metadata[
            "source"
        ],
        "candidates": content(
            raw_transcript
        ),
        "transcript": raw_transcript,
    }
    with (
        open(
            os.path.join(
                file_path,
                "metadata.json",
            ),
            "w",
        ) as f
    ):
        print(
            "Formating Json"
        )
        json.dump(
            data,
            f,
            indent=2,
        )


def download(
    path,
):
    file_path = Path(
        f"{path}/video"
    )
    file_path.mkdir(
        parents=False,
        exist_ok=True,
    )
    video(
        file_path,
        yt_url,
        780,
        DOWNLOAD_VIDEO,
    )


def edit(
    path,
):
    metadata = Path(
        f"{path}/metadata/metadata.json"
    )
    video = Path(
        f"{path}/video/"
    )
    with (
        open(
            metadata,
            "r",
        ) as f
    ):
        info = json.load(
            f
        )
    wrapper = EditingWrapper()
    wrapper.execute(
        video_src=Path(
            f"{video}/video.mp4"
        ),
        template_path="template/editing_no-sub.py",
        output_dir=Path(
            f"{video}/output"
        ),
        metadata=info,
    )


def main():
    if not TES_SCRIPT:
      uuid = uuid4()
    else:
      uuid = PATH_UUID
    path_uuid = Path(
        f"{output_path}/{uuid}"
    )
    path_uuid.mkdir(
        parents=True,
        exist_ok=True,
    )
    if TRANSCRIPT:
      video_transcript(
          path_uuid
      )
    download(
        path_uuid,
    )
    if EDITING:
        edit(
            path_uuid
        )
    print(
        "Done"
    )


main()
