# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated chess evaluation functions.
Drop-in replacements for chess_board.evaluate_position_advanced and evaluate_material_pst.
"""

cimport cython

# ─── C-level lookup tables (filled once at import time) ───────────────────────

cdef double W_PST[7][64]   # [piece_type][square], piece_type 1-5
cdef double B_PST[7][64]
cdef double W_KING_MG[64]
cdef double W_KING_EG[64]
cdef double B_KING_MG[64]
cdef double B_KING_EG[64]
cdef unsigned long long W_PASSED[64]
cdef unsigned long long B_PASSED[64]
cdef unsigned long long ADJ_FILE[8]
cdef unsigned long long CENTER_MASK_C

# ─── Bit manipulation intrinsics ──────────────────────────────────────────────

cdef inline int bsf64(unsigned long long bb) noexcept nogil:
    """Bit scan forward - find index of least significant set bit."""
    cdef int idx = 0
    if not (bb & 0xFFFFFFFFULL):
        idx += 32
        bb >>= 32
    if not (bb & 0xFFFFULL):
        idx += 16
        bb >>= 16
    if not (bb & 0xFFULL):
        idx += 8
        bb >>= 8
    if not (bb & 0xFULL):
        idx += 4
        bb >>= 4
    if not (bb & 0x3ULL):
        idx += 2
        bb >>= 2
    if not (bb & 0x1ULL):
        idx += 1
    return idx

cdef inline int popcount64(unsigned long long x) noexcept nogil:
    """Population count - count number of set bits."""
    x = x - ((x >> 1) & 0x5555555555555555ULL)
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL
    return <int>((x * 0x0101010101010101ULL) >> 56)

# ─── Table initialization (called once at module import) ──────────────────────

def _init_tables():
    """Copy Python lookup tables into C arrays for zero-overhead access."""
    from chess_board import (
        _W_PST_LIST, _B_PST_LIST,
        _W_KING_MG_LIST, _W_KING_EG_LIST,
        _B_KING_MG_LIST, _B_KING_EG_LIST,
        _W_PASSED_INT, _B_PASSED_INT,
        _ADJ_FILE_INT, _CENTER_MASK,
    )

    cdef int pt, sq

    for pt in range(7):
        for sq in range(64):
            W_PST[pt][sq] = _W_PST_LIST[pt][sq]
            B_PST[pt][sq] = _B_PST_LIST[pt][sq]

    for sq in range(64):
        W_KING_MG[sq] = _W_KING_MG_LIST[sq]
        W_KING_EG[sq] = _W_KING_EG_LIST[sq]
        B_KING_MG[sq] = _B_KING_MG_LIST[sq]
        B_KING_EG[sq] = _B_KING_EG_LIST[sq]
        W_PASSED[sq] = <unsigned long long>_W_PASSED_INT[sq]
        B_PASSED[sq] = <unsigned long long>_B_PASSED_INT[sq]

    for sq in range(8):
        ADJ_FILE[sq] = <unsigned long long>_ADJ_FILE_INT[sq]

    global CENTER_MASK_C
    CENTER_MASK_C = <unsigned long long>_CENTER_MASK

# Run init at import time
_init_tables()

# ─── Lightweight eval: material + PST only (for quiescence search) ────────────

cpdef double cy_evaluate_material_pst(object bb):
    """Fast material + PST eval for quiescence. Skips pawn structure/king safety.
    Takes a chess.Board directly. Returns score in pawns from white's perspective.
    """
    cdef unsigned long long w_occ = <unsigned long long>int(bb.occupied_co[True])
    cdef unsigned long long b_occ = <unsigned long long>int(bb.occupied_co[False])
    cdef double score = 0.0
    cdef unsigned long long tmp
    cdef int sq

    # Pawns
    tmp = <unsigned long long>int(bb.pawns) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 100.0 + W_PST[1][sq]
    tmp = <unsigned long long>int(bb.pawns) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 100.0 + B_PST[1][sq]

    # Knights
    tmp = <unsigned long long>int(bb.knights) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 320.0 + W_PST[2][sq]
    tmp = <unsigned long long>int(bb.knights) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 320.0 + B_PST[2][sq]

    # Bishops
    tmp = <unsigned long long>int(bb.bishops) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 330.0 + W_PST[3][sq]
    tmp = <unsigned long long>int(bb.bishops) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 330.0 + B_PST[3][sq]

    # Rooks
    tmp = <unsigned long long>int(bb.rooks) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 500.0 + W_PST[4][sq]
    tmp = <unsigned long long>int(bb.rooks) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 500.0 + B_PST[4][sq]

    # Queens
    tmp = <unsigned long long>int(bb.queens) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 900.0 + W_PST[5][sq]
    tmp = <unsigned long long>int(bb.queens) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 900.0 + B_PST[5][sq]

    # Tempo
    if bb.turn:
        score += 10.0
    else:
        score -= 10.0

    return score * 0.01

# ─── Full evaluation: material + PST + pawn structure + king safety ───────────

cpdef double cy_evaluate_position_advanced(object board):
    """Full classical evaluation. Returns score from white's perspective (pawns).
    Takes a ChessBoard wrapper (board.board is the chess.Board).
    """
    cdef object bb = board.board

    cdef unsigned long long w_occ = <unsigned long long>int(bb.occupied_co[True])
    cdef unsigned long long b_occ = <unsigned long long>int(bb.occupied_co[False])

    cdef int phase = 0
    cdef double score = 0.0
    cdef int w_bishops = 0, b_bishops = 0
    cdef int w_minors = 0, b_minors = 0
    cdef unsigned long long tmp
    cdef int sq

    # ── Kings (always exactly one per side) ──
    cdef unsigned long long wk_bb = <unsigned long long>int(bb.kings) & w_occ
    cdef int wk_sq = bsf64(wk_bb)
    cdef unsigned long long bk_bb = <unsigned long long>int(bb.kings) & b_occ
    cdef int bk_sq = bsf64(bk_bb)

    # ── Pawns ──
    cdef unsigned long long w_pawns_int = <unsigned long long>int(bb.pawns) & w_occ
    cdef int w_pawns_by_file[8]
    cdef int i
    for i in range(8):
        w_pawns_by_file[i] = 0

    tmp = w_pawns_int
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 100.0 + W_PST[1][sq]
        w_pawns_by_file[sq & 7] += 1

    cdef unsigned long long b_pawns_int = <unsigned long long>int(bb.pawns) & b_occ
    cdef int b_pawns_by_file[8]
    for i in range(8):
        b_pawns_by_file[i] = 0

    tmp = b_pawns_int
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 100.0 + B_PST[1][sq]
        b_pawns_by_file[sq & 7] += 1

    # ── Knights ──
    tmp = <unsigned long long>int(bb.knights) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 320.0 + W_PST[2][sq]
        phase += 1
        w_minors += 1

    tmp = <unsigned long long>int(bb.knights) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 320.0 + B_PST[2][sq]
        phase += 1
        b_minors += 1

    # ── Bishops ──
    tmp = <unsigned long long>int(bb.bishops) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 330.0 + W_PST[3][sq]
        phase += 1
        w_bishops += 1
        w_minors += 1

    tmp = <unsigned long long>int(bb.bishops) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 330.0 + B_PST[3][sq]
        phase += 1
        b_bishops += 1
        b_minors += 1

    # ── Rooks ──
    cdef int w_rook_files[10]  # max 10 rooks (promotion edge case)
    cdef int w_rook_count = 0
    tmp = <unsigned long long>int(bb.rooks) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 500.0 + W_PST[4][sq]
        phase += 2
        w_rook_files[w_rook_count] = sq & 7
        w_rook_count += 1

    cdef int b_rook_files[10]
    cdef int b_rook_count = 0
    tmp = <unsigned long long>int(bb.rooks) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 500.0 + B_PST[4][sq]
        phase += 2
        b_rook_files[b_rook_count] = sq & 7
        b_rook_count += 1

    # ── Queens ──
    tmp = <unsigned long long>int(bb.queens) & w_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score += 900.0 + W_PST[5][sq]
        phase += 4

    tmp = <unsigned long long>int(bb.queens) & b_occ
    while tmp:
        sq = bsf64(tmp)
        tmp &= tmp - 1
        score -= 900.0 + B_PST[5][sq]
        phase += 4

    # ── Game phase and king PST ──
    if phase > 24:
        phase = 24
    cdef double eg_w = 1.0 - phase * 0.041666666666666664  # 1/24
    cdef double mg_w = 1.0 - eg_w

    score += W_KING_MG[wk_sq] * mg_w + W_KING_EG[wk_sq] * eg_w
    score -= B_KING_MG[bk_sq] * mg_w + B_KING_EG[bk_sq] * eg_w

    # ── Bishop pair bonus ──
    if w_bishops >= 2:
        score += 50.0
    if b_bishops >= 2:
        score -= 50.0

    # ── Pawn structure ──
    cdef int f, wp, bp
    cdef unsigned long long a
    for f in range(8):
        wp = w_pawns_by_file[f]
        bp = b_pawns_by_file[f]
        a = ADJ_FILE[f]
        if wp > 1:
            score -= 20.0 * (wp - 1)
        if bp > 1:
            score += 20.0 * (bp - 1)
        if wp and not (w_pawns_int & a):
            score -= 15.0 * wp
        if bp and not (b_pawns_int & a):
            score += 15.0 * bp

    # ── Passed pawns ──
    if w_pawns_int:
        tmp = w_pawns_int
        while tmp:
            sq = bsf64(tmp)
            if not (b_pawns_int & W_PASSED[sq]):
                score += 20.0 + 10.0 * (sq >> 3)
            tmp &= tmp - 1

    if b_pawns_int:
        tmp = b_pawns_int
        while tmp:
            sq = bsf64(tmp)
            if not (w_pawns_int & B_PASSED[sq]):
                score -= 20.0 + 10.0 * (7 - (sq >> 3))
            tmp &= tmp - 1

    # ── Rook on open / semi-open files ──
    cdef int rf
    for i in range(w_rook_count):
        rf = w_rook_files[i]
        if w_pawns_by_file[rf] == 0:
            if b_pawns_by_file[rf] == 0:
                score += 25.0
            else:
                score += 15.0
    for i in range(b_rook_count):
        rf = b_rook_files[i]
        if b_pawns_by_file[rf] == 0:
            if w_pawns_by_file[rf] == 0:
                score -= 25.0
            else:
                score -= 15.0

    # ── Center control ──
    cdef int w_center = popcount64(w_occ & CENTER_MASK_C)
    cdef int b_center = popcount64(b_occ & CENTER_MASK_C)
    score += (w_center - b_center) * 15.0

    # ── Mobility approximation ──
    score += (w_minors - b_minors) * 5.0

    # ── King safety (pawn shield + semi-open file penalty) ──
    cdef double sf
    cdef int kf, df, ff
    if eg_w < 0.6:
        sf = 1.0 - eg_w
        # White king
        kf = wk_sq & 7
        for df in range(-1, 2):
            ff = kf + df
            if ff < 0 or ff > 7:
                continue
            if w_pawns_by_file[ff] == 0:
                score -= 15.0 * sf
                if b_pawns_by_file[ff] == 0:
                    score -= 5.0 * sf
                else:
                    score -= 10.0 * sf
        # Black king
        kf = bk_sq & 7
        for df in range(-1, 2):
            ff = kf + df
            if ff < 0 or ff > 7:
                continue
            if b_pawns_by_file[ff] == 0:
                score += 15.0 * sf
                if w_pawns_by_file[ff] == 0:
                    score += 5.0 * sf
                else:
                    score += 10.0 * sf

    # ── Tempo bonus ──
    if bb.turn:
        score += 10.0
    else:
        score -= 10.0

    return score * 0.01
