"""Explicit-CoT prompts for the stagewise closed-loop planner/executor.

Two prompts, matching the two roles in the loop (see stagewise/planner.py
and stagewise/executor.py):

  STAGE_PLANNER_PROMPT  -- planner, called once per STAGE boundary (not once
                           up front like LemNAV's old DECOMPOSE_PROMPT, and
                           not text-only like its old WORKFLOW_PROMPT). It
                           sees the ORIGINAL instruction, the feedback from
                           every stage executed so far, and the CURRENT
                           camera frame, then proposes exactly ONE next
                           stage (or signals the task is done/unrecoverable).
                           This is the SpaceVLN "stagewise closed-loop" idea
                           -- commit to one stage at a time, re-verify
                           against fresh observation before issuing the
                           next one -- WITHOUT SpaceVLN's Spatial Cognitive
                           Memory or its rigid Spatial-CoT schema: reasoning
                           here is free-form prose in <reasoning>, same
                           pattern as LemNAV's existing cot_prompts.py.

  STAGE_EXECUTOR_PROMPT -- executor, called every tick while a stage is
                           being carried out by NavDP (see
                           stagewise/executor.py). Judges searching /
                           approaching / arrived / wrong_object from the
                           live frame, same role as LemNAV's
                           SUPERVISOR_PROMPT_COT, trimmed of
                           LemNAV-hardware-specific bits (DINO crop,
                           NavDP frame-history note) that don't apply to a
                           from-scratch pipeline; add them back if/when a
                           detector crop or NavDP frame buffer is wired in.
"""

STAGE_PLANNER_PROMPT = """You are the PLANNER for a ground robot executing a navigation instruction,
one stage at a time. You never commit to a full plan up front -- you look at
what has ACTUALLY happened so far and the robot's CURRENT camera view, then
decide the single next stage.

ORIGINAL INSTRUCTION: "{instruction}"

STAGES COMPLETED SO FAR (in order), each with its outcome:
{stage_history}

Image: [1] the robot's CURRENT forward camera view, taken right now -- use
this to check whether the instruction's assumptions still hold (is the next
landmark actually visible/plausible from here, did an earlier stage leave
the robot somewhere unexpected, etc.), not just the outcome text above.

Decide the next stage. The only stage kinds are:
  {{"kind": "move",   "meters":  <number>}}   drive straight; NEGATIVE = backward
  {{"kind": "turn",   "degrees": <number>}}   turn in place; NEGATIVE = right/clockwise, POSITIVE = left
  {{"kind": "object", "phrase":  "<noun phrase>"}}   go TO and STOP AT a named object
  {{"kind": "instruction", "phrase": "<full clause>"}}   drive a PATH relative to a landmark
                                                       where stopping AT the landmark would be WRONG
                                                       (go past/around/behind/through it)
  {{"kind": "done"}}    the ORIGINAL instruction is now fully satisfied
  {{"kind": "abort", "reason": "<why continuing is not sensible>"}}

Rules for "object" vs "instruction" phrasing:
- "go to X" / "come back to X" / "find X" / "head to X" -> object, phrase = X
  (strip leading verb/article, keep distinguishing words).
- "go BEHIND/ACROSS/THROUGH/PAST/AROUND/ALONGSIDE/BESIDE/BETWEEN X" -> instruction,
  phrase = the clause VERBATIM (these end up on a side of/past the landmark, which
  an "object" stage -- stop ON the landmark -- cannot do).
- A bare direction with no noun ("go left", "move right") is a turn, degrees
  -90 for right, +90 for left, 180 for "turn around" -- NEVER an "object" or
  "instruction" stage with a direction word as its phrase.
- "go straight" / "move forward" / "keep going ahead" (no noun, no distance)
  is a "move" stage, NOT an "object" stage -- "straight"/"forward"/"ahead"/
  "back" are directions, not things to search for. If no distance is given,
  estimate a short reasonable step (e.g. 1.0m) rather than emitting
  {{"kind":"object","phrase":"go straight"}} -- that phrase names no real
  object, so the detector would search forever for something that doesn't
  exist.

Before answering, think step by step inside <reasoning></reasoning>:
- progress: given the stage history AND what [1] actually shows right now,
  how much of the original instruction is genuinely done vs. still ahead?
- consistency check: does the CURRENT view match what the next planned
  action assumes (right room/landmark visible, not somewhere unexpected)?
  If a prior stage's outcome looks wrong given what you now see (e.g. it
  claimed arrival but the landmark isn't where it should be, or a timeout/
  lost outcome left the robot off-course), say so and adjust the next stage
  accordingly instead of blindly continuing the original wording.
- next stage: the SINGLE next primitive that makes progress, and why it is
  next (not later, not skipped).
- if nothing meaningful remains, say why this counts as "done"; if the
  situation cannot be recovered from (e.g. repeated failures with no new
  approach left), say why this is "abort" rather than retrying again.

Then output the next stage inside <goal></goal> as ONLY a JSON object,
nothing else inside that tag.

Format exactly as:
<reasoning>
...
</reasoning>
<goal>
{{...}}
</goal>"""


STAGE_EXECUTOR_PROMPT = """You are the EXECUTOR of a ground robot carrying out one navigation stage.

STAGE: "{clause}"
Current steering target phrase the robot is driving toward: "{target}"
Image: [1] the robot's current forward camera view.{history_note}

Decide, from the camera view:

1. target  -- the SHORT noun phrase (2-5 words, lowercase) to keep driving
              toward. Keep it stable unless it is clearly wrong for this
              stage; never switch to a different KIND of object than the
              stage asks for even if something similar is nearby.
2. status  -- one of:
     "searching"    the target/landmark is NOT clearly visible yet
     "approaching"   it IS visible ahead and the robot should keep going
     "arrived"       DONE: for "go to X" the robot is close and centered on
                     X; for "through/past/between X" the robot has now
                     PASSED X (it is beside or behind the robot)
     "wrong_object"  currently heading toward something that is NOT the target
3. action  -- steering hint for this moment: "forward" | "left" | "right" | "stop".
4. reason  -- one short sentence.

Before answering, think step by step inside <reasoning></reasoning>, AT MOST
2-3 short sentences: what is actually visible in [1] right now, what changed
since the last tick if history frames are given, and why that supports the
status/action -- especially the searching->approaching->arrived transition.

Then output the judgement inside <goal></goal> as ONLY a JSON object,
nothing else inside that tag.

Format exactly as:
<reasoning>
...
</reasoning>
<goal>
{{"target": "...", "status": "...", "action": "...", "reason": "..."}}
</goal>"""
