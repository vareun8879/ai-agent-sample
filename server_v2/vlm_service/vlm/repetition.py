"""출력이 토큰 한도에 걸렸을 때 끝부분의 반복 루프를 잘라내는 후처리.

실측(09-21): 사진 한 장에 문구 한 줄뿐인 홍보 슬라이드(ZERO-PASS.pdf 12p 등)에서 VLM 이 흐릿한 배경 글자를
읽으려다 같은 문구를 되풀이해 4,096토큰(5천~1만 자)을 다 채우고 잘렸다. 토큰 상향으로는 해결되지 않고,
잘림 직전까지의 본문은 쓸 만하므로 반복이 시작된 지점에서 잘라 정상 결과로 돌려준다.

잘림(finish_reason == 'length')일 때만 부른다 — 정상 종료된 표 전사에는 같은 행이 여러 번 나올 수 있어 손대지 않는다.
"""

# 반복 단위(줄 수)의 최대 길이 — 여러 줄짜리 문단이 통째로 되풀이되는 경우까지 잡는다
_MAX_PERIOD_LINES = 20
# 이 횟수 이상 잇달아 되풀이되면 루프로 본다
_MIN_REPEATS = 3
# 줄바꿈 없이 한 줄 안에서 도는 루프 판정용 꼬리 길이(자)와 최소 출현 횟수
_TAIL_CHARS = 40
_TAIL_MIN_COUNT = 4


def trim_trailing_repetition(text: str) -> tuple[str, bool]:
    """끝부분이 같은 줄 묶음의 반복이면 첫 묶음만 남기고 잘라 (텍스트, 잘랐는지)를 돌려준다."""
    lines = text.split("\n")
    # 마지막 줄은 토큰 한도에 걸려 중간에 끊긴 조각일 수 있으므로 그 줄을 뺀 경우도 검사한다
    for drop_last in (0, 1):
        body = lines[: len(lines) - drop_last] if drop_last else lines
        n = len(body)
        for k in range(1, _MAX_PERIOD_LINES + 1):
            if k * _MIN_REPEATS > n:
                break
            unit = body[n - k:n]
            if not any(l.strip() for l in unit):
                continue
            reps = 1
            while (reps + 1) * k <= n and body[n - (reps + 1) * k:n - reps * k] == unit:
                reps += 1
            if reps >= _MIN_REPEATS:
                return "\n".join(body[: n - (reps - 1) * k]).rstrip(), True
    # 줄바꿈 없는 루프: 꼬리 40자가 본문에 여러 번 나오면 첫 출현 직후에서 자른다
    tail = text[-_TAIL_CHARS:]
    if len(text) > _TAIL_CHARS * 10 and tail.strip() and text.count(tail) >= _TAIL_MIN_COUNT:
        first = text.find(tail)
        return text[: first + len(tail)].rstrip(), True
    return text, False
