"""DeepSeek-R1 compatible generator."""

import re

from generator import OpenAIChatGPTModel

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_LABEL_LINE_RE = re.compile(
    r"(?im)^\s*(?:judg(?:e)?ment|prediction|label|判定|判断|结论)\s*[:：]\s*(normal|anomaly|正常|异常)\b"
)
_CONCLUSION_RE = re.compile(
    r"(?:the\s+(?:answer|result|conclusion|classification)\s+is|"
    r"(?:overall|finally|therefore|thus|hence)\s*,?\s*(?:it\s+is\s+|this\s+is\s+)?|"
    r"classify\s+(?:it\s+)?as|classified?\s+as|"
    r"(?:this|the\s+(?:query|log(?:\s+sequence)?|sequence|case|sample|instance))\s+"
    r"(?:is|indicates|looks|appears|seems)\s+(?:a\s+|an\s+)?)"
    r"\s*[:：-]?\s*(normal|anomaly|正常|异常)\b",
    re.IGNORECASE,
)
_NORMAL_CUE_RE = re.compile(
    r"(?:"
    r"i\s+do\s+not\s+see\s+(?:any\s+)?(?:strong\s+anomaly\s+cue(?:s)?|logs?\s+that\s+indicate\s+an?\s+error|errors?|warnings?|exceptions?)|"
    r"i\s+don't\s+see\s+(?:any\s+)?(?:strong\s+anomaly\s+cue(?:s)?|logs?\s+that\s+indicate\s+an?\s+error|errors?|warnings?|exceptions?|of\s+these)|"
    r"there\s+are\s+no\s+strong\s+anomaly\s+cue(?:s)?|"
    r"not\s+a\s+strong\s+cue|"
    r"soft\s+cue(?:s)?\s+are\s+not\s+sufficient|"
    r"which\s+is\s+normal|"
    r"it\s+should\s+be\s+normal|"
    r"this\s+should\s+be\s+normal|"
    r"all\s+the\s+logs\s+are\s+just"
    r")",
    re.IGNORECASE,
)
_ANOMALY_CUE_RE = re.compile(
    r"(?:"
    r"there\s+is\s+(?:at\s+least\s+one\s+)?strong\s+anomaly\s+cue|"
    r"there\s+are\s+strong\s+anomaly\s+cue(?:s)?|"
    r"contains?\s+(?:a|at\s+least\s+one\s+)?strong\s+anomaly\s+cue|"
    r"this\s+indicates\s+anomaly|"
    r"this\s+is\s+anomaly|"
    r"it\s+should\s+be\s+anomaly|"
    r"this\s+should\s+be\s+anomaly"
    r")",
    re.IGNORECASE,
)


_THINK_EXTRACT_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STEP1_RE = re.compile(r"step\s*1", re.IGNORECASE)
_GATE_KW_RE = re.compile(r"\[(relevant|irrelevant|supported|unsupported|answer|query|accepted|continue|reject)\]", re.IGNORECASE)


class DeepSeekChatGPTModel(OpenAIChatGPTModel):
    """Drop-in replacement that strips reasoning tags from responses."""

    def get_response(self, *args, **kwargs):
        text = super().get_response(*args, **kwargs)
        if not text:
            return text

        # Extract <think> content before stripping, in case it contains structured steps.
        think_match = _THINK_EXTRACT_RE.search(text)
        think_content = think_match.group(1).strip() if think_match else ""

        # Strip <think>...</think> from the visible output.
        stripped = _THINK_RE.sub("", text).strip()

        # If the stripped text already has structured gate steps, use it as-is.
        if _STEP1_RE.search(stripped):
            return stripped

        # If structured steps are inside <think>, return think content for parsing_thought().
        if think_content and _STEP1_RE.search(think_content):
            # Normalize "step by step" so parsing_thought() index() doesn't misfire.
            think_content = re.sub(r'\bstep[\s\-]+by[\s\-]+step\b', 'step-by-step', think_content, flags=re.IGNORECASE)
            return think_content

        # DeepSeek may return reasoning directly without <think> tags (server strips them).
        # If the raw text contains gate keywords like [RELEVANT]/[SUPPORTED], return as-is.
        if _GATE_KW_RE.search(text):
            return text

        # Otherwise this is a response prompt — extract the label.
        text = stripped
        match = _LABEL_LINE_RE.search(text)
        if not match:
            matches = list(_CONCLUSION_RE.finditer(text))
            match = matches[-1] if matches else None
        if match:
            token = (match.group(1) or "").strip().lower()
            if token in {"anomaly", "异常"}:
                head = "Judgment: Anomaly"
            else:
                head = "Judgment: Normal"
            tail = text[match.end():].lstrip()
            text = (head + ("\n" + tail if tail else "")).strip()
        if (not match) and _NORMAL_CUE_RE.search(text):
            text = ("Judgment: Normal\n" + text).strip()
        elif (not match) and _ANOMALY_CUE_RE.search(text):
            text = ("Judgment: Anomaly\n" + text).strip()
        return text
