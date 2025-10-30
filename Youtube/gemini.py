from dotenv import (
    load_dotenv,
)
from google import (
    genai,
)
from google.genai import (
    types,
)
from pathlib import (
    Path,
)
import os
import json

load_dotenv()
client = genai.Client()


def getText(
    text,
):
    transkrip_text = "\n".join(
        [
            f"[{d['start']} - {d['stop']}] {d['text']}"
            for d in text
        ]
    )
    prompt = ""

    with (
        open(
            "prompt.txt",
            "r",
        ) as f
    ):
        prompt = f.read()

    return (
        prompt
        + transkrip_text
    )


safety_settings = [
    {
        "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": types.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": types.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": types.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": types.HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
        "threshold": types.HarmBlockThreshold.BLOCK_NONE,
    },
]


def content(
    data,
    model="gemini-2.5-pro",
):
    response = client.models.generate_content(
        contents=getText(
            data
        ),
        model=model,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=32768
            ),
            safety_settings=safety_settings,
        ),
    )
    clean = (
        response.text.replace(
            "```json",
            "",
        )
        .replace(
            "```",
            "",
        )
        .strip()
    )
    return json.loads(
        clean
    )
