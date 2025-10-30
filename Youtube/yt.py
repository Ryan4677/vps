from dotenv import (
    load_dotenv,
)
from supadata import (
    Supadata,
    SupadataError,
)
from urllib.parse import (
    urlparse,
)
import yt_dlp
import json
import os

load_dotenv()
supadata = Supadata(
    api_key=os.getenv(
        "SUPADATA_API_KEY"
    )
)


def transcript(
    url,
):
    dump = []
    yt_id = urlparse(
        url
    ).path.strip(
        "/"
    )
    transcript = supadata.youtube.transcript(
        video_id=yt_id
    ).content

    for content in transcript:
        duration = int(
            content.duration
        )
        start = int(
            content.offset
        )
        stop = (
            duration
            + start
        )
        data = {
            "text": content.text,
            "start": start,
            "stop": stop,
            "duration": duration,
        }
        dump.append(
            data
        )

    return json.dumps(
        dump
    )


def channel(
    video_url,
):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with (
            yt_dlp.YoutubeDL(
                ydl_opts
            ) as ydl
        ):
            info = ydl.extract_info(
                video_url,
                download=False,
            )
            channel_name = info.get(
                "channel",
                None,
            )
            channel_id = info.get(
                "channel_id",
                None,
            )
            title = info.get(
                "title",
                None,
            )
            toJson = {
                "source": video_url,
                "name": channel_name,
                "id": channel_id,
                "title": title,
            }
            return json.dumps(
                toJson
            )
    except Exception as e:
        print(
            f"Error: {e}"
        )
        return (
            None,
            None,
        )


def download(
    path,
    url,
    res,
):
    ydl_opts = {
        "format": str(
            res
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(
            path,
            "video.%(ext)s",
        ),
    }
    with (
        yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl
    ):
        ydl.download(
            url
        )


def video(
    path,
    url,
    video_resolution,
    download_video=False,
):
    ydl = yt_dlp.YoutubeDL()
    info = ydl.extract_info(
        url,
        download=False,
    )

    # Filter format yang cocok
    candidates = [
        f
        for f in info[
            "formats"
        ]
        if f[
            "ext"
        ]
        == "mp4"
        and f[
            "width"
        ]
        is not None
        and f[
            "width"
        ]
        <= video_resolution
    ]

    if not candidates:
        print(
            "Tidak ada resolusi di bawah target!"
        )
        return None

    best = max(
        candidates,
        key=lambda x: x[
            "width"
        ],
    )
    resolution = best[
        "format_id"
    ]

    print(
        f"🎬 Format terpilih: {best['width']}x{best['height']} ({resolution})"
    )

    if download_video:
        download(
            path,
            url,
            resolution,
        )
    else:
        return resolution


# video(
#     "https://youtu.be/RJ0Zyp5YPOQ",

#     1330,
#     True,
# )

# for i in info['formats']:
#   if i['ext'] == "mp4":
#     print(i['resolution'])

# with (
#     open(
#         "tes.json",
#         "w",
#     ) as f
# ):
#   json.dump(info,f,indent=2)
