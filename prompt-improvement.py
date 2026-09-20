import argparse, json, re
from pathlib import Path
from typing import Optional

from pydantic import ValidationError, BaseModel, Field
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
EDITABLE_FIELDS = ("num_people", "poses", "text")
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are adapting a synthetic-video prompt to match a real reference frame.
You will see an image and a JSON template describing a staged scenario.

Return ONLY a raw JSON object (no markdown fences, no commentary) with exactly these keys:
"num_people", "poses", "text".

Rules:
- num_people: a rough count/description of people visible, e.g. "2", "5", "7+", "crowded".
  Precision doesn't matter, don't overthink it.
- poses: describe the position/posture of every visible person, keeping the SAME flagged
  actor and SAME anomalous action described in the template's "text" field. Add brief,
  neutral descriptions for any additional bystanders (e.g. "three more passengers seated
  further down the aisle, uninvolved").
- text: adjust ONLY the parts describing who is present / where they are. Do NOT change,
  remove, or invent a different anomalous action than the one in the template. Do NOT
  add new incident types. Keep length and tone similar to the original.
- If the template's scene already roughly matches the image, make minimal or no changes.

Template:
{template_json}"""

# A class to check the answer format
class PromptPatch(BaseModel):
    num_people: str = Field(...)
    poses: str = Field(...)
    text: str = Field(...)

def extract_json(raw: str)-> str:
    raw = re.sub(r"<think>.*?</think>","", raw, flags=re.DOTALL)
    match = re.search(r"{.*}", raw.strip(), re.DOTALL)
    return match.group(0) if match else raw.strip()

def build_messages(template: dict, frame_path: Path, correction: Optional[str] = None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            template_json=json.dumps(template, ensure_ascii=False))},
        {"role": "user", "content": [
            {"type": "image", "image": str(frame_path)},
            {"type": "text", "text": "Adapt the template to this frame."},
        ]},
    ]
    if correction:
        messages.append({"role":"user", "content":"correction"})

    return messages

def run_vlm(model, processor, messages, max_new_tokens=512):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    trimmed2 = generated_ids[len(inputs.input_ids):]
    if trimmed == trimmed2:
        print("YOU WERE RIGHT!!")
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

def redact_prompt(model, processor,
                        template: dict, frame_path: Path, new_id: str) -> dict:
    payload = build_messages(template, frame_path)
    last_raw = None


    for attempt in range(MAX_RETRIES+1):
        raw = run_vlm(model, processor, payload)
        last_raw = raw
        try:
            patch = PromptPatch.model_validate_json(extract_json(raw))
            break
        except (ValidationError, json.JSONDecodeError) as e:
            correction = (f"Your last output was invalid ({e}). "
                          f"Return ONLY raw JSON with keys num_people, poses, text.")
            payload = build_messages(template, frame_path, correction=correction)
    else:
        raise RuntimeError(f"VLM failed to produce valid patch for {new_id}. Last output: {last_raw}")

    merged = {**template, **patch.model_dump(), "id": new_id, "template_id": template["id"], "provenance": {
        "source_frame": str(frame_path),
        "vlm_model": MODEL_ID,
        "edited_fields": list(EDITABLE_FIELDS),
    }}
    return merged

def find_frame(template_id: str, frames_dir: Optional[Path], single_frame: Optional[Path]) -> Optional[Path]:
    if single_frame:
        return single_frame
    if frames_dir:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = frames_dir / f"{template_id}{ext}"
            if candidate.exists():
                return candidate
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=Path("resources/prompts.json"))
    ap.add_argument("--frames-dir", type=Path, default=None,
                     help="Directory of real frames named <template_id>.jpg")
    ap.add_argument("--frame", type=Path, default=None,
                     help="Apply this single frame to every template instead of a frames dir")
    ap.add_argument("--out-dir", type=Path, default=Path("resources/real_batch_01"))
    ap.add_argument("--id-suffix", default="rf")
    args = ap.parse_args()

    if not args.frames_dir and not args.frame:
        ap.error("Provide either --frames-dir or --frame")

    templates = json.loads(args.prompts.read_text(encoding="utf-8"))

    out_prompts_path = args.out_dir / "prompts_real.json"
    out_frames_dir = args.out_dir / "cabin_conditioned"
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results = []
    for template in templates:
        tid = str(template["id"])
        frame_path = find_frame(tid, args.frames_dir, args.frame)
        if frame_path is None:
            print(f"[skip] no matching frame for template {tid}")
            continue

        new_id = f"{tid}_{args.id_suffix}"
        print(f"[{new_id}] redacting against {frame_path.name} ...")
        try:
            redacted = redact_prompt(model, processor, template, frame_path, new_id)
        except RuntimeError as e:
            print(f"[{new_id}] FAILED: {e}")
            continue

        dest = out_frames_dir / f"{new_id}{frame_path.suffix}"
        dest.write_bytes(frame_path.read_bytes())
        results.append(redacted)

    out_prompts_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} redacted prompts to {out_prompts_path}")
    print(f"Point main.py at: PROMPTS={out_prompts_path}  CABIN_IMAGE_PREFIX={out_frames_dir}/")


if __name__ == "__main__":
    main()