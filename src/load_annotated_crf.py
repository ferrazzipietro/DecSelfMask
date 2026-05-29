import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, Features, Sequence, Value


_NOTE_ID_RE = re.compile(r"(?m)^NoteID:\s*(\d+)\s*$")


@dataclass(frozen=True)
class NoteSegment:
    note_id: str
    start: int
    end: int


def infer_section_from_path(json_path: str | Path) -> str:
    p = Path(json_path)
    return p.stem


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_grouped_notes(text: str) -> List[NoteSegment]:
    """Split a grouped text into note segments using `NoteID: <id>` markers.

    Returns segments with [start, end) offsets into the original text.
    If no markers are found, returns an empty list.
    """
    matches = list(_NOTE_ID_RE.finditer(text))
    if not matches:
        return []

    segments: List[NoteSegment] = []
    for i, m in enumerate(matches):
        note_id = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append(NoteSegment(note_id=note_id, start=start, end=end))
    return segments


def extract_spans_from_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract Label Studio-style span annotations from a task."""
    spans: List[Dict[str, Any]] = []

    def iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
        if obj is None:
            return
        if isinstance(obj, dict):
            yield obj
        elif isinstance(obj, list):
            for item in obj:
                yield from iter_dicts(item)

    for ann in iter_dicts(task.get("annotations") or []):
        annotation_id_raw = ann.get("id")
        completed_by_raw = ann.get("completed_by")
        annotation_id = int(annotation_id_raw) if annotation_id_raw is not None else -1
        completed_by = int(completed_by_raw) if completed_by_raw is not None else -1

        for res in iter_dicts(ann.get("result") or []):
            if res.get("type") != "labels":
                continue
            value = res.get("value")
            if not isinstance(value, dict):
                continue

            start = value.get("start")
            end = value.get("end")
            if start is None or end is None:
                continue

            raw_labels = value.get("labels") or []
            labels: List[str] = []
            if isinstance(raw_labels, list):
                for item in raw_labels:
                    if isinstance(item, list):
                        labels.extend(str(x) for x in item)
                    else:
                        labels.append(str(item))
            else:
                labels = [str(raw_labels)]

            spans.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "text": str(value.get("text") or ""),
                    "labels": labels,
                    "from_name": str(res.get("from_name") or ""),
                    "to_name": str(res.get("to_name") or ""),
                    "annotation_id": annotation_id,
                    "completed_by": completed_by,
                    "result_id": str(res.get("id") or ""),
                }
            )

    return spans


def load_section_as_dataset(
    json_path: str | Path,
    *,
    section: Optional[str] = None,
    keep_only_in_bounds: bool = True,
) -> Dataset:
    """Load a section JSON (Label Studio export) as a Hugging Face Dataset.

    Each JSON element is expected to be a task with at least:
    - `id`
    - `data.text`
    - `annotations[].result[]` entries of type `labels` with `value.start/end/text/labels`

    If `explode_notes=True`, each grouped task is split by `NoteID:` markers, and
    spans are assigned to the note they fall into (offsets are adjusted).
    """
    tasks = _read_json(json_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a list of tasks in {json_path}, got {type(tasks)}")

    section_name = section or infer_section_from_path(json_path)

    rows: List[Dict[str, Any]] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        task_id = task.get("id")
        data = task.get("data") or {}
        text = data.get("text")
        if text is None:
            continue
        text = str(text)

        note_segments = split_grouped_notes(text)
        note_ids = [seg.note_id for seg in note_segments]

        spans = extract_spans_from_task(task)

        if keep_only_in_bounds:
            spans = [s for s in spans if 0 <= s["start"] <= s["end"] <= len(text)]

        if not note_segments:
            rows.append(
                {
                    "section": section_name,
                    "task_id": int(task_id) if task_id is not None else -1,
                    "text": text,
                    "note_ids": [],
                    "spans": spans,
                    "note_id": "",
                    "note_start": 0,
                    "note_end": len(text),
                    "dropped_spans": 0,
                }
            )
            continue

        for seg in note_segments:
            seg_text = text[seg.start : seg.end]
            seg_spans: List[Dict[str, Any]] = []
            dropped = 0
            for s in spans:
                if s["start"] >= seg.start and s["end"] <= seg.end:
                    seg_spans.append(
                        {
                            **s,
                            "start": int(s["start"] - seg.start),
                            "end": int(s["end"] - seg.start),
                        }
                    )
                else:
                    dropped += 1

            rows.append(
                {
                    "section": section_name,
                    "task_id": int(task_id) if task_id is not None else -1,
                    "text": seg_text,
                    "note_ids": note_ids,
                    "spans": seg_spans,
                    "note_id": seg.note_id,
                    "note_start": int(seg.start),
                    "note_end": int(seg.end),
                    "dropped_spans": int(dropped),
                }
            )

    return Dataset.from_list(rows)
