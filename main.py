#!/usr/bin/env python3
"""Werewolf agent for the make.fun bounty werewolf-agent-01.

Plays one seat in 30 games of 8-player Werewolf against house agents
(claude-haiku-5-5) through the env at MAKEFUN_ENV_URL. Python 3.13, standard
library only. Deterministic given its model replies: no wall clock, no
unseeded randomness (every random draw comes from the trial seed), and every
model request is built from one game's view alone.

How it plays
------------
* One model call per decision (when a provider key is available), with a
  compact rendered view plus role-specific strategy in the prompt. Provider
  order: Anthropic claude-sonnet-5-5, then OpenAI gpt-6.1-sol, then
  gemini-4-argon through the OpenAI-compatible gateway route. A tiny probe
  at startup picks the first provider whose gateway answers 200; a 401
  means that vendor's key was not pushed, so the next one is tried.
* A heuristic fallback answers every decision when no model is available,
  a call fails, or the reply cannot be read: suspicion scoring from the
  public log (votes, accusations, revealed roles, night deaths), wolf kill
  coordination with the partner's pick, seer/doctor value targeting, and
  templated statements. The fallback is deterministic.
* Decisions across live games run in parallel (up to 10 at once, the env's
  live-game cap); what one game does never shapes another game's request.

Run: python3 -P -s main.py  (as the grader runs it)
Env: MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED,
     ANTHROPIC_BASE_URL, OPENAI_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY
     (GEMINI_API_KEY is read too when the Gemini route is probed).
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------- constants

ANTHROPIC_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-6.1-sol"
GEMINI_MODEL = "gemini-4-argon"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 1024
MAX_REQUEST_BYTES = 131_072
CONCURRENCY = 10
RETRY_WAIT_SEC = 5
RETRY_AFTER_MAX_SEC = 20
HTTP_TIMEOUT_S = 60

# Gateway refusals that mean spending is done: final, stop calling models.
SPENT_TYPES = ("budget_exhausted", "request_cap")
# Retryable HTTP statuses (one retry, like the baseline).
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)

ROLE_WORD = {
    "werewolf": "werewolf",
    "seer": "seer",
    "doctor": "doctor",
    "villager": "villager",
}
ROLE_ARTICLE = {
    "werewolf": "a werewolf",
    "seer": "the seer",
    "doctor": "the doctor",
    "villager": "a villager",
}

# ---------------------------------------------------------------- seeded RNG
# xoshiro256** seeded from sha256(domain, seed, label): the same values for
# the same seed and label, exactly like the kit SDK's Rng.

RNG_DOMAIN = "werewolf-agent-01/rng/v1"
_MASK64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Rng:
    def __init__(self, seed, label=""):
        h = hashlib.sha256(
            RNG_DOMAIN.encode() + b"\0" + str(seed).encode() + b"\0" + label.encode()
        ).digest()
        s = [int.from_bytes(h[i:i + 8], "big") for i in range(0, 32, 8)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self):
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        a2 = s2 ^ s0
        a3 = s3 ^ s1
        self._s = [s0 ^ a3, s1 ^ a2, a2 ^ t, _rotl(a3, 45)]
        return result

    def randint(self, lo, hi):
        span = hi - lo + 1
        limit = (1 << 64) - (1 << 64) % span
        while True:
            x = self.next_u64()
            if x < limit:
                return lo + x % span

    def choice(self, items):
        return items[self.randint(0, len(items) - 1)]


# ---------------------------------------------------------------- HTTP

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HttpError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def http_post(url, payload, headers=None, timeout=HTTP_TIMEOUT_S):
    """POST JSON, return (status, parsed JSON or raw text). Raises on no answer."""
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 **(headers or {})},
    )
    try:
        with _OPENER.open(req, timeout=timeout) as res:
            raw = res.read()
            return res.status, _parse_body(raw)
    except urllib.error.HTTPError as err:
        try:
            return err.code, _parse_body(err.read())
        finally:
            err.close()
    except (urllib.error.URLError, OSError) as err:
        raise ConnectionError(str(getattr(err, "reason", err))) from None


def _parse_body(raw):
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def _error_type(body):
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            t = e.get("type")
            if isinstance(t, str):
                return t
            c = e.get("code")
            if isinstance(c, str):
                return c
    return ""


def _retry_wait_s(retry_after):
    v = (retry_after or "").strip()
    if re.fullmatch(r"\d+", v):
        return min(int(v), RETRY_AFTER_MAX_SEC)
    return RETRY_WAIT_SEC


# ---------------------------------------------------------------- model clients


class ModelClient:
    """One provider behind the make.fun gateway (or the real vendor API when
    graded locally with the user's own key in the environment)."""

    def __init__(self, kind, base_url, api_key, model):
        self.kind = kind  # "anthropic" | "openai"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dead = False  # set when the gateway refuses for spending reasons

    def _post(self, path, body, headers):
        status, raw = http_post(f"{self.base_url}{path}", body, headers)
        # http_post returns parsed body; re-raise style: return status/body
        return status, raw

    def call(self, system, user_text):
        """One completion. Returns the reply text, or None when there is no
        readable reply (caller falls back to the heuristic). Raises
        ValueError('unauthorized') on 401, RuntimeError('spent') on a
        spending refusal."""
        if self.kind == "anthropic":
            body = {
                "model": self.model,
                "max_tokens": MAX_TOKENS,
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_text}],
            }
            path = "/v1/messages"
            headers = {"anthropic-version": ANTHROPIC_VERSION,
                       "x-api-key": self.api_key}
        else:
            body = {
                "model": self.model,
                "max_completion_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
            }
            path = "/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
        attempt = 0
        while True:
            try:
                status, resp = self._post(path, body, headers)
            except ConnectionError:
                return None  # no answer: the gateway never saw it; fall back
            if status == 401:
                raise ValueError("unauthorized")
            if status == 200 and isinstance(resp, dict):
                text = _extract_text(self.kind, resp)
                return text if text else None
            etype = _error_type(resp)
            if etype in SPENT_TYPES:
                self.dead = True
                raise RuntimeError("spent:" + etype)
            if status in RETRY_STATUSES and attempt == 0:
                attempt += 1
                time.sleep(_retry_wait_s(None))
                continue
            return None


def _extract_text(kind, resp):
    if kind == "anthropic":
        parts = []
        for block in resp.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    choices = resp.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def probe(client):
    """One tiny call: True when the gateway answers 200 for this provider."""
    try:
        text = client.call(
            "You are a helpful assistant. Reply with one JSON object only.",
            'Reply with one JSON object and nothing else: {"ok": true}',
        )
    except (ValueError, RuntimeError):
        return False
    return text is not None


# ---------------------------------------------------------------- view analysis

def name_of(view, seat):
    for p in view.get("players") or []:
        if p.get("seat") == seat:
            return p.get("name") or f"seat {seat}"
    return f"seat {seat}"


def living(view):
    return [p["seat"] for p in view.get("players") or [] if p.get("alive")]


def alive_non_wolves(view, me):
    """Living seats that are not me and not my known wolf partner."""
    partner = (view.get("you") or {}).get("partner") or {}
    p_seat = partner.get("seat")
    return [s for s in living(view) if s != me and s != p_seat]


def revealed(view):
    """seat -> role for roles revealed by death or exile."""
    out = {}
    for p in view.get("players") or []:
        if p.get("role"):
            out[p["seat"]] = p["role"]
    return out


def suspicion(view):
    """Deterministic suspicion scores over living players, from the public
    log only (plus the seer's private results, which override). Higher =
    more likely a werewolf. Keys: seat -> float."""
    me = (view.get("you") or {}).get("seat")
    rev = revealed(view)
    scores = {s: 0.0 for s in living(view) if s != me}
    # Seer knowledge overrides everything.
    for r in view.get("seerResults") or []:
        seat = r.get("seat")
        if seat in scores:
            scores[seat] = 100.0 if r.get("werewolf") else -100.0
    # Vote record: voting to exile a revealed villager is suspicious;
    # voting to exile a revealed wolf clears you. (Speech mentions are
    # not scored: a name in a sentence is not an accusation.)
    for e in view.get("log") or []:
        kind = e.get("kind")
        if kind == "vote":
            voter, target = e.get("seat"), e.get("target")
            if voter in scores and target in rev:
                if rev[target] == "werewolf":
                    scores[voter] -= 2.0
                elif rev[target] in ("villager", "seer", "doctor"):
                    scores[voter] += 2.0
    # The dead were killed by wolves: whoever they accused most is suspect.
    for e in view.get("log") or []:
        if e.get("kind") == "dawn" and e.get("died"):
            dead = e["died"]
            for f in view.get("log") or []:
                if f.get("kind") == "speech" and f.get("seat") == dead:
                    text = f.get("text") or ""
                    for seat in list(scores):
                        if seat != dead and name_of(view, seat) in text:
                            scores[seat] += 0.5
    return scores


def ranked_suspects(view):
    """Living non-self seats, most suspicious first; ties by lowest seat."""
    sc = suspicion(view)
    return sorted(sc, key=lambda s: (-sc[s], s))


def render_view(view):
    """The view as compact text (players, private info, game log), the way
    the house players see it. Shorter than raw JSON, same information."""
    you = view.get("you") or {}
    lines = []
    phase = view.get("phase")
    lines.append(f"Game, {phase} of day {view.get('day')}.")
    lines.append("")
    lines.append("PLAYERS")
    for p in view.get("players") or []:
        role = p.get("role")
        state = "alive" if p.get("alive") else f"dead, was a {ROLE_WORD.get(role, role)}"
        mark = " (you)" if p.get("seat") == you.get("seat") else ""
        lines.append(f"[moderator] seat {p['seat']} ({p.get('name')}): {state}{mark}.")
    lines.append("")
    lines.append("YOUR PRIVATE INFORMATION")
    lines.append(
        f"[moderator] You are seat {you.get('seat')} ({you.get('name')}), "
        f"{ROLE_ARTICLE.get(you.get('role'), you.get('role'))}, "
        f"{'alive' if you.get('alive') else 'dead'}."
    )
    partner = you.get("partner")
    if partner:
        lines.append(
            f"[moderator] Your partner is seat {partner.get('seat')} "
            f"({partner.get('name')}), {'alive' if partner.get('alive') else 'dead'}."
        )
    for v in view.get("wolfVotes") or []:
        tgt = "named nobody" if v.get("target") is None else f"named seat {v['target']} ({name_of(view, v['target'])})"
        lines.append(f"[moderator] Night {v.get('night')}: seat {v.get('seat')} {tgt}.")
    if view.get("seerResults") is not None:
        if not view["seerResults"]:
            lines.append("[moderator] You have no inspection results yet.")
        for r in view["seerResults"]:
            lines.append(
                f"[moderator] Night {r.get('night')}: you inspected seat {r.get('seat')} "
                f"({name_of(view, r['seat'])}): {'a werewolf' if r.get('werewolf') else 'not a werewolf'}."
            )
    if view.get("lastProtect") is not None:
        lp = view["lastProtect"]
        tgt = "nobody" if lp.get("seat") is None else f"seat {lp['seat']} ({name_of(view, lp['seat'])})"
        lines.append(f"[moderator] Night {lp.get('night')}: you protected {tgt}.")
    lines.append("")
    lines.append("GAME LOG")
    lines.extend(_render_log(view))
    outcome = view.get("outcome")
    if outcome:
        lines.append(
            f"[moderator] The game is over: "
            f"{'a draw, nobody wins' if outcome.get('winner') is None else 'the ' + str(outcome.get('winner')) + ' win'}."
        )
    return "\n".join(lines)


def _render_log(view):
    out = []
    round_no = 1
    for e in view.get("log") or []:
        kind = e.get("kind")
        day = e.get("day")
        if kind == "debate":
            round_no = 1
        elif kind in ("speech", "pass"):
            if e.get("round") != round_no:
                round_no = e.get("round") or round_no
                out.append(f"[moderator] Day {day} debate, round {round_no} of 2.")
            if kind == "speech" and e.get("text"):
                out.append(f"seat {e['seat']} ({name_of(view, e['seat'])}): {e['text']}")
            else:
                out.append(f"seat {e['seat']} ({name_of(view, e['seat'])}): (passes)")
        elif kind == "dawn":
            died = e.get("died")
            if died is None:
                out.append(f"[moderator] Day {day} dawn: no one died last night.")
            else:
                out.append(
                    f"[moderator] Day {day} dawn: seat {died} ({name_of(view, died)}) died; "
                    f"they were a {ROLE_WORD.get(e.get('role'), e.get('role'))}."
                )
        elif kind == "vote":
            tgt = e.get("target")
            if tgt is None:
                out.append(f"seat {e['seat']} ({name_of(view, e['seat'])}): abstains.")
            else:
                out.append(
                    f"seat {e['seat']} ({name_of(view, e['seat'])}): votes seat {tgt} "
                    f"({name_of(view, tgt)})."
                )
        elif kind == "exile":
            tally = e.get("tally") or []
            votes = "no votes" if not tally else "votes: " + ", ".join(
                f"seat {t['seat']} {t['votes']}" for t in tally
            )
            if e.get("seat") is None:
                out.append(f"[moderator] Nobody was exiled ({votes}).")
            else:
                out.append(
                    f"[moderator] seat {e['seat']} ({name_of(view, e['seat'])}) was exiled "
                    f"({votes}); they were a {ROLE_WORD.get(e.get('role'), e.get('role'))}."
                )
        elif kind == "end":
            winner = e.get("winner")
            out.append(
                "[moderator] The game is over: "
                + ("a draw, nobody wins." if winner is None else f"the {winner} win.")
            )
    return out


# ---------------------------------------------------------------- strategy prompts

def rules_from_brief(brief):
    parts = re.split(r"(?=^## )", brief, flags=re.M)
    kept = [p.strip() for p in parts if p.startswith("## The game\n")]
    return "\n\n".join(kept) if kept else brief.strip()


STRATEGY = {
    "werewolf": (
        "You are a WEREWOLF. Your partner is named in your private information.\n"
        "Night: the victim dies ONLY if every wolf who names someone names the SAME player, so agree with your partner. "
        "If your partner already named someone tonight, name the same seat. Otherwise name the most dangerous villager: "
        "a revealed seer first, then the most vocal player leading accusations, then anyone who is not your partner. "
        "Never name your partner.\n"
        "Day: hide what you are. Blend in, sound like an ordinary villager reasoning from the same evidence, and steer "
        "suspicion and votes onto villagers with plausible reasons tied to what they actually said or how they voted. "
        "Never defend your partner openly; if your partner is in danger, deflect by accusing someone else. "
        "Vote with the emerging plurality against a villager. Claim to be a villager if asked; only claim a power role "
        "if it clearly saves you and you can fake it. Keep statements to 1-3 natural sentences."
    ),
    "seer": (
        "You are the SEER. Each night inspect the living player you find most suspicious (never yourself). "
        "Your results are in your private information.\n"
        "Day: once you have found a werewolf, say so plainly and lead the village to vote them out, with your reasoning. "
        "Before you have a wolf result, do not claim to be the seer unless pressed; a revealed seer becomes the wolves' "
        "next victim. Vote for a known werewolf first; otherwise vote the most suspicious player. "
        "Build your case from specific evidence: votes, contradictions, and who defended whom."
    ),
    "doctor": (
        "You are the DOCTOR. Each night protect a living player other than yourself, and not the seat you protected "
        "the night before. Protect the most valuable villager: a revealed seer first, then the most vocal village leader, "
        "then someone the wolves are likely to target. You cannot protect yourself.\n"
        "Day: vote the most suspicious player and push suspicion onto wolves with specific evidence. Do not claim to be "
        "the doctor unless it clearly helps; your value is staying alive to protect."
    ),
    "villager": (
        "You are a VILLAGER. You know only your own role.\n"
        "Day: read the votes and the statements. Werewolves often vote together, avoid voting each other, and push "
        "suspicion onto villagers with thin reasons. Track who voted to exile revealed villagers (suspicious) and who "
        "voted to exile a revealed wolf (cleared). Vote the most suspicious living player, and speak to build a case "
        "from specific evidence: exact votes, contradictions, and who defended whom. Keep statements to 1-3 sentences."
    ),
}


def system_for(rules, role):
    return "\n".join([
        "You are playing one seat in a game of Werewolf. These are the rules.",
        "",
        rules,
        "",
        "YOUR ROLE AND STRATEGY",
        STRATEGY.get(role, STRATEGY["villager"]),
        "",
        "Answer with one short JSON object and nothing else, no code fence. "
        "Put your answer first; \"thought\" is your private reasoning in one short sentence, and nobody else sees it.",
    ])


def decision_text(view, pending):
    kind = pending["decision"]
    options = pending.get("options") or []
    opts = ", ".join(f"seat {s} ({name_of(view, s)})" for s in options)
    if kind == "speak":
        return (
            f"It is your turn to speak in the day's debate. Make one statement of at most "
            f"{pending.get('max_chars', 500)} characters (two or three sentences), or pass.\n"
            'Reply with one JSON object and nothing after it: {"say": "<your statement>", "thought": "<reason>"}, '
            'or {"say": null, "thought": "<reason>"} to pass.'
        )
    labels = {
        "vote": "vote to exile one of",
        "kill": "name the seat you want the werewolves to kill, one of",
        "protect": "choose the seat to protect, one of",
        "inspect": "choose the seat to inspect, one of",
    }
    return (
        f"It is your turn to {kind}. {labels[kind]}: {opts}; or choose nobody.\n"
        'Reply with one JSON object and nothing after it: {"target": <seat number>, "thought": "<reason>"}, '
        'or {"target": null, "thought": "<reason>"} to choose nobody.'
    )


# ---------------------------------------------------------------- reply parsing

def _json_objects(text):
    out = []
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            return out
        depth = 0
        in_str = False
        j = start
        while j < len(text):
            ch = text[j]
            if in_str:
                if ch == "\\":
                    j += 1
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:j + 1])
                        if isinstance(v, dict):
                            out.append(v)
                    except ValueError:
                        pass
                    break
            j += 1
        i = start + 1


def fold_statement(text):
    folded = re.sub(r"[\t\n\v\f\r\u0085\u2028\u2029]+", " ", text)
    folded = re.sub(r"[\x00-\x1f\x7f]", "", folded)
    return folded.strip()


def fallback_act(pending):
    if pending["decision"] == "speak":
        return {"game": pending["game"], "decision": "speak", "pass": True}
    return {"game": pending["game"], "decision": pending["decision"], "target": None}


def parse_reply(text, pending):
    """The /act body a reply's text answers with; (ok, act)."""
    key = "say" if pending["decision"] == "speak" else "target"
    found = [o for o in _json_objects(text or "") if key in o]
    if not found:
        return False, fallback_act(pending)
    value = found[-1][key]
    if pending["decision"] == "speak":
        if value is None:
            return True, fallback_act(pending)
        if not isinstance(value, str):
            return False, fallback_act(pending)
        folded = fold_statement(value)
        if folded == "":
            return True, fallback_act(pending)
        max_chars = pending.get("max_chars") or 500
        chars = list(folded)
        if len(chars) > max_chars:
            folded = "".join(chars[:max_chars]).strip()
        return True, {"game": pending["game"], "decision": "speak", "text": folded}
    seat = value
    if isinstance(seat, str) and re.fullmatch(r"\s*\d+\s*", seat):
        seat = int(seat)
    target = seat if isinstance(seat, int) and seat in (pending.get("options") or []) else None
    return (value is None or target is not None), {
        "game": pending["game"], "decision": pending["decision"], "target": target,
    }


# ---------------------------------------------------------------- heuristic fallback

def _rng_for(seed, game, pending):
    view = pending["view"]
    label = f"game{game}:{pending['decision']}:{len(view.get('log') or [])}"
    return Rng(seed, label)


def heuristic_act(seed, pending):
    """A deterministic answer with no model: suspicion scoring, wolf
    coordination, seer/doctor value targeting, templated statements."""
    view = pending["view"]
    you = view.get("you") or {}
    me = you.get("seat")
    role = you.get("role")
    kind = pending["decision"]
    options = pending.get("options") or []
    rng = _rng_for(seed, pending["game"], pending)
    suspects = ranked_suspects(view)
    rev = revealed(view)

    if kind == "speak":
        text = _heuristic_statement(view, rng, suspects)
        if not text:
            return fallback_act(pending)
        return {"game": pending["game"], "decision": "speak", "text": text}

    if kind == "kill":
        # Coordinate: match the partner's pick when they already named one.
        for v in view.get("wolfVotes") or []:
            if v.get("night") == view.get("day") and v.get("target") is not None:
                if v["target"] in options:
                    return {"game": pending["game"], "decision": "kill", "target": v["target"]}
        target = _wolf_target(view, options, rng)
        return {"game": pending["game"], "decision": "kill", "target": target}

    if kind == "inspect":
        target = suspects[0] if suspects and suspects[0] in options else (options[0] if options else None)
        return {"game": pending["game"], "decision": "inspect", "target": target}

    if kind == "protect":
        target = _doctor_target(view, options, rng)
        return {"game": pending["game"], "decision": "protect", "target": target}

    # vote
    target = None
    if role == "werewolf":
        # Vote with the plurality against a villager: the most-voted living
        # non-wolf today, else the top suspect who is not a known wolf.
        target = _plurality_target(view, options) or (suspects[0] if suspects else None)
        if target is not None and rev.get(target) == "werewolf":
            target = next((s for s in suspects if rev.get(s) != "werewolf"), None)
    else:
        known = [s for s in suspects if (view.get("seerResults") or []) and any(
            r.get("seat") == s and r.get("werewolf") for r in view["seerResults"])]
        target = known[0] if known else (suspects[0] if suspects else None)
    if target is not None and target not in options:
        target = None
    return {"game": pending["game"], "decision": "vote", "target": target}


def _wolf_target(view, options, rng):
    """The most valuable villager to kill: a claimed seer, then the most
    vocal accuser, then a deterministic pick among the options."""
    if not options:
        return None
    me = (view.get("you") or {}).get("seat")
    # A player who claimed to be the seer (or announced an inspection) dies first.
    for e in view.get("log") or []:
        if e.get("kind") == "speech" and (e.get("text") or ""):
            t = e["text"].lower()
            if ("i am the seer" in t or "i'm the seer" in t or "i inspected" in t
                    or "inspected" in t and "werewolf" in t):
                if e["seat"] in options and e["seat"] != me:
                    return e["seat"]
    # The most vocal living non-wolf (most speeches) is the next threat.
    counts = {}
    for e in view.get("log") or []:
        if e.get("kind") == "speech" and e.get("seat") in options:
            counts[e["seat"]] = counts.get(e["seat"], 0) + 1
    if counts:
        top = max(counts.values())
        loud = [s for s, c in counts.items() if c == top]
        return sorted(loud)[0]
    return sorted(options)[0]


def _doctor_target(view, options, rng):
    if not options:
        return None
    # A revealed seer (someone who announced a wolf inspection) is protected first.
    for e in view.get("log") or []:
        if e.get("kind") == "speech" and (e.get("text") or ""):
            t = e["text"].lower()
            if "i am the seer" in t or "i'm the seer" in t or "i inspected" in t:
                if e["seat"] in options:
                    return e["seat"]
    # Else the most vocal living player (the wolves' likeliest target).
    counts = {}
    for e in view.get("log") or []:
        if e.get("kind") == "speech" and e.get("seat") in options:
            counts[e["seat"]] = counts.get(e["seat"], 0) + 1
    if counts:
        top = max(counts.values())
        return sorted(s for s, c in counts.items() if c == top)[0]
    return sorted(options)[0]


def _plurality_target(view, options):
    """The living option with the most votes cast for it today, if any."""
    day = view.get("day")
    tally = {}
    for e in view.get("log") or []:
        if e.get("kind") == "vote" and e.get("day") == day and e.get("target") in options:
            tally[e["target"]] = tally.get(e["target"], 0) + 1
    if not tally:
        return None
    top = max(tally.values())
    return sorted(s for s, c in tally.items() if c == top)[0]


def _heuristic_statement(view, rng, suspects):
    """A short templated statement (<= 500 chars), role-aware."""
    you = view.get("you") or {}
    me = you.get("seat")
    role = you.get("role")
    rev = revealed(view)
    living_seats = [s for s in living(view) if s != me]
    if not living_seats:
        return None

    def nm(s):
        return name_of(view, s)

    # Seer with a found wolf announces it.
    if role == "seer":
        for r in view.get("seerResults") or []:
            if r.get("werewolf") and r.get("seat") in living_seats:
                return (
                    f"I inspected {nm(r['seat'])} last night and they are a werewolf. "
                    f"We should vote {nm(r['seat'])} out today."
                )[:500]
        if suspects:
            s = suspects[0]
            return rng.choice([
                f"I am watching {nm(s)} closely. Their votes do not add up, and I want to hear their reasoning today.",
                f"{nm(s)} has been steering suspicion onto villagers all game. I think we should look at them today.",
            ])[:500]
    if role == "werewolf":
        # Accuse the top non-wolf suspect; never name the partner.
        partner = (you.get("partner") or {}).get("seat")
        target = next((s for s in suspects if s != partner and rev.get(s) != "werewolf"), None)
        if target is None:
            target = next((s for s in living_seats if s != partner), None)
        if target is None:
            return None
        return rng.choice([
            f"I keep coming back to {nm(target)}. Their votes have pushed villagers out while the wolves stayed quiet. I think they are a wolf.",
            f"{nm(target)} has been quick to accuse others but never answers for their own votes. That looks like a wolf hiding in the open.",
            f"Look at how {nm(target)} voted yesterday. They helped exile a villager and moved on like nothing happened. I vote we look at them.",
        ])[:500]
    # Village / doctor: accuse the top suspect with evidence-ish wording.
    if suspects:
        s = suspects[0]
        if rev.get(s) == "werewolf":
            return f"{nm(s)} is a confirmed werewolf. Vote {nm(s)} out today."[:500]
        return rng.choice([
            f"I think {nm(s)} is a wolf. Their votes have lined up against villagers, and their accusations never land on a real wolf.",
            f"{nm(s)} keeps deflecting whenever they are asked about their votes. I want them voted out today.",
            f"Watch {nm(s)}: they pushed suspicion onto villagers and stayed silent when a wolf was exiled. That is wolf behavior.",
        ])[:500]
    return rng.choice([
        "I have no strong read yet. I will listen to this round before I vote.",
        "Nothing stands out to me yet. Let us hear everyone before deciding.",
    ])[:500]


# ---------------------------------------------------------------- the run


class Agent:
    def __init__(self, env):
        self.env_url = env["MAKEFUN_ENV_URL"].rstrip("/")
        self.seed = env.get("MAKEFUN_TRIAL_SEED", "")
        self.brief = ""
        prompt_file = env.get("MAKEFUN_PROMPT_FILE", "")
        if prompt_file:
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.brief = f.read()
        self.rules = rules_from_brief(self.brief)
        self.clients = []
        self._build_clients(env)
        self.active_client = None
        self.models_dead = False
        self.summary = {"decisions": 0, "answered": 0, "fallbacks": 0, "calls": 0}
        self._lock = threading.Lock()

    def _build_clients(self, env):
        anthropic_base = env.get("ANTHROPIC_BASE_URL", "").strip()
        anthropic_key = env.get("ANTHROPIC_API_KEY", "").strip()
        openai_base = env.get("OPENAI_BASE_URL", "").strip()
        openai_key = env.get("OPENAI_API_KEY", "").strip()
        if anthropic_base and anthropic_key:
            self.clients.append(ModelClient("anthropic", anthropic_base, anthropic_key, ANTHROPIC_MODEL))
        if openai_base and openai_key:
            self.clients.append(ModelClient("openai", openai_base, openai_key, OPENAI_MODEL))
            self.clients.append(ModelClient("openai", openai_base, openai_key, GEMINI_MODEL))
        elif openai_base and env.get("GEMINI_API_KEY", "").strip():
            self.clients.append(ModelClient("openai", openai_base, env["GEMINI_API_KEY"].strip(), GEMINI_MODEL))

    def pick_provider(self):
        """Probe each provider once; keep the first that answers 200."""
        for client in self.clients:
            if probe(client):
                self.active_client = client
                return client
        return None

    def _request_text(self, pending):
        """The user message for one decision, clipped to the request cap."""
        view = pending["view"]
        role = (view.get("you") or {}).get("role", "villager")
        system = system_for(self.rules, role)
        user = "\n".join([
            "Your view of the game:",
            render_view(view),
            "",
            f"Your decision ({pending['decision']}):",
            decision_text(view, pending),
        ])
        # Clip the oldest speeches until the whole request fits.
        view2 = view
        while True:
            body = self._body(system, user, view2)
            if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) <= MAX_REQUEST_BYTES:
                return system, user
            clipped = _clip_oldest_speech(view2)
            if clipped is None:
                return system, user
            view2 = clipped
            user = "\n".join([
                "Your view of the game:",
                render_view(view2),
                "",
                f"Your decision ({pending['decision']}):",
                decision_text(view2, pending),
            ])

    def _body(self, system, user, view):
        return {"model": "x", "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": user}]}

    def ask_model(self, pending):
        """The model's /act body for one decision, or None to fall back."""
        if self.active_client is None or self.models_dead:
            return None
        system, user = self._request_text(pending)
        try:
            text = self.active_client.call(system, user)
        except ValueError:
            # 401: this provider's key was not pushed; try the next client.
            self.clients = [c for c in self.clients if c is not self.active_client]
            self.active_client = None
            for client in self.clients:
                if probe(client):
                    self.active_client = client
                    break
            if self.active_client is None:
                self.models_dead = True
                return None
            return self.ask_model(pending)
        except RuntimeError:
            self.models_dead = True
            return None
        except ConnectionError:
            return None
        with self._lock:
            self.summary["calls"] += 1
        if text is None:
            return None
        ok, act = parse_reply(text, pending)
        return act if ok else None

    def decide(self, pending):
        """One decision: model first, heuristic fallback, then POST /act."""
        with self._lock:
            self.summary["decisions"] += 1
        act = None
        try:
            act = self.ask_model(pending)
        except Exception:
            act = None
        if act is None:
            act = heuristic_act(self.seed, pending)
            with self._lock:
                self.summary["fallbacks"] += 1
        else:
            with self._lock:
                self.summary["answered"] += 1
        status, _ = http_post(f"{self.env_url}/act", act)
        if status == 400 and act != fallback_act(pending):
            with self._lock:
                self.summary["fallbacks"] += 1
            http_post(f"{self.env_url}/act", fallback_act(pending))
        # 409 (not pending) means the game moved on: nothing to do.

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        self.pick_provider()
        started = set()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {}
            while True:
                try:
                    status, obs = http_post(f"{self.env_url}/observe", {})
                except ConnectionError as err:
                    raise RuntimeError(f"the env is unreachable: {err}") from None
                if status != 200 or not isinstance(obs, dict):
                    raise RuntimeError(f"POST /observe answered {status}")
                if obs.get("done"):
                    break
                pending = obs.get("pending") or []
                submitted = False
                for p in pending:
                    key = f"{p['game']}:{p['decision']}:{len((p.get('view') or {}).get('log') or [])}"
                    if key in started or len(futures) >= CONCURRENCY:
                        continue
                    started.add(key)
                    fut = pool.submit(self._safe_decide, p)
                    futures[fut] = key
                    submitted = True
                if futures:
                    # When nothing new started, wait for a decision to finish
                    # instead of re-asking /observe (it would answer the same
                    # pending list at once). When new work started, poll
                    # briefly so freed slots are picked up promptly.
                    timeout = 0.2 if submitted else None
                    done, _ = wait(list(futures), timeout=timeout,
                                   return_when=FIRST_COMPLETED)
                    for fut in done:
                        futures.pop(fut, None)
                        exc = fut.exception()
                        if exc is not None:
                            raise RuntimeError(f"a decision failed: {exc}") from exc
                else:
                    time.sleep(0.25)
            # The trial is done; let in-flight decisions finish (their /act
            # will 409 harmlessly if the game already ended).
            for fut in list(futures):
                try:
                    fut.result(timeout=30)
                except Exception:
                    pass
        s = self.summary
        print(
            f"werewolf-agent: {s['decisions']} decisions, {s['answered']} answered "
            f"by the model, {s['fallbacks']} fell back; {s['calls']} model calls",
            flush=True,
        )
        return s

    def _safe_decide(self, pending):
        try:
            self.decide(pending)
        except Exception as err:
            # A decision must never stall a game: fall back and post it.
            try:
                http_post(f"{self.env_url}/act", fallback_act(pending))
            except Exception:
                pass
            raise err


def _clip_oldest_speech(view):
    """A copy of the view with the oldest speech/pass entry shortened; None
    when nothing left to clip."""
    log = view.get("log") or []
    for i, e in enumerate(log):
        if e.get("kind") == "speech" and isinstance(e.get("text"), str) and len(e["text"]) > 80:
            new_log = [dict(x) for x in log]
            new_log[i] = dict(e)
            new_log[i]["text"] = e["text"][:80] + "…"
            new_view = dict(view)
            new_view["log"] = new_log
            return new_view
    return None


def main():
    env = os.environ
    if not env.get("MAKEFUN_ENV_URL"):
        print("MAKEFUN_ENV_URL is not set: run the entry under the agent/v1 harness",
              file=sys.stderr)
        return 1
    if not env.get("MAKEFUN_PROMPT_FILE"):
        print("MAKEFUN_PROMPT_FILE is not set", file=sys.stderr)
        return 1
    try:
        agent = Agent(env)
        agent.run()
    except Exception as err:
        print(f"werewolf-agent: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
