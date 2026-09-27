import csv
import os

STORY_SEP = "\n<|endofstory|>\n"


def load_stories(path="train.csv", fraction=1.0, text_column="text", fix_mojibake=True):
    """Read TinyStories from its single-column CSV, stopping after `fraction` of the file's bytes.

    Stories are joined with STORY_SEP. The CSV is double-encoded UTF-8;
    fix_mojibake reverses that.
    """
    target_bytes = os.path.getsize(path) * fraction
    stories = []
    consumed = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            story = row[text_column]
            stories.append(story)
            consumed += len(story.encode("utf-8"))
            if consumed >= target_bytes:
                break
    text = STORY_SEP.join(stories)
    if fix_mojibake:
        text = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    return text
