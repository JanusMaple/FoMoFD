"""CPU-only final-answer parsing and generation stopping for the VLM protocol.

Nothing here masks logits or constrains intermediate text. Only an exact, final
answer outside a balanced reasoning section can become a detector decision.
"""
import re
import time


def parse_final_answer(text, *, require_think=False):
    """Return FAIL/NORMAL, or None for incomplete/malformed/ambiguous output."""
    text = text.strip()
    # The answer must terminate the original response, not merely the text left
    # after removing reasoning. A think block AFTER an answer is not a final answer.
    if re.search(r'<answer>(FAIL|NORMAL)</answer>\s*\Z', text) is None:
        return None
    outside, cursor, inside, think_count = [], 0, False, 0
    for match in re.finditer(r"</?think>", text):
        if match.group() == "<think>":
            if inside:
                return None
            outside.append(text[cursor:match.start()])
            inside = True
            think_count += 1
        else:
            if not inside:
                return None
            inside = False
            outside.append("\n")
        cursor = match.end()
    if inside or think_count > 1 or (require_think and think_count != 1):
        return None
    outside.append(text[cursor:])
    visible = "".join(outside).strip()
    match = re.search(r"(?:^|\n)[ \t]*(<answer>(FAIL|NORMAL)</answer>)\s*\Z", visible)
    if match is None:
        return None
    prefix = visible[:match.start()]
    # Reject extra/partial tags, prose after the decision, and multiple answers.
    if re.search(r"</?(?:answer|think)\b", prefix) or (require_think and prefix.strip()):
        return None
    return match.group(2)


class FinalAnswerStop:
    """Transformers-compatible callable, batch size one; no generation imports.

    Timeout is cooperative: a running prefill/decode kernel is never interrupted.
    Keep the most recent generated suffix so an OOM/error can retain partial text.
    """
    def __init__(self, tokenizer, input_length, *, require_think, started,
                 timeout_s=None, clock=time.perf_counter):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.require_think = require_think
        self.started, self.timeout_s, self.clock = started, timeout_s, clock
        self.reason = None
        self.text, self.token_ids = "", []

    def __call__(self, input_ids, scores=None, **kwargs):
        if len(input_ids) != 1:
            raise ValueError("VLM stopping supports exactly one sequence")
        self.token_ids = input_ids[0, self.input_length:].tolist()
        self.text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        if self.timeout_s is not None and self.clock() - self.started >= self.timeout_s:
            self.reason = "timeout"
        elif parse_final_answer(self.text, require_think=self.require_think) is not None:
            self.reason = "final_answer"
        return self.reason is not None


def classify_completion(text, token_ids, *, require_think, max_new_tokens,
                        eos_token_ids, stopping_reason):
    """An invalid response is indeterminate, never an implicit NORMAL."""
    label = parse_final_answer(text, require_think=require_think)
    if stopping_reason == "timeout":
        return {"status": "timed_out", "final_label": None, "stopping_reason": "timeout"}
    if label is not None:
        return {"status": "valid", "final_label": label, "stopping_reason": "final_answer"}
    if token_ids and token_ids[-1] in eos_token_ids:
        reason, status = "eos", "malformed"
    elif len(token_ids) >= max_new_tokens:
        reason, status = "token_budget", "truncated"
    else:
        reason, status = "unexpected_stop", "malformed"
    return {"status": status, "final_label": None, "stopping_reason": reason}
