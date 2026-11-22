"""
unicode_nd_app.py  —  Unicode → n-Dimensional Data Explorer (v3)
================================================================

Takes a dataset and:

  1. Preprocesses each data point (optional lowercase / whitespace strip /
     punctuation removal).
  2. Encodes every character into a numeric value using a selectable scheme
     (Unicode code point, UTF-8 bytes, UTF-16 code units, lossy ASCII).
  3. Pads / truncates every vector to a common ambient dimension.
  4. Optionally transforms each vector (raw, L2-normalized, z-score, min-max).
  5. Wraps each row in a `DataPoint` object and the whole dataset in an
     `NDimensionalObject` — an axis-aligned hyperrectangle in n-space.
  6. Projects the cloud to 2-D (PCA, random projection, first/last dims) and
     draws it with a convex hull and projected bounding box.

New in v3
---------
  • Built-in EXAMPLES library with 10 illustrative datasets:
      - Greek alphabet            (short words, ASCII)
      - Animal names              (varied lengths)
      - Digits & numbers          (numeric strings)
      - Mixed case words          (case sensitivity demo)
      - Emoji                       (multi-byte Unicode demo)
      - CJK characters            (3-byte UTF-8 demo)
      - Greek / Cyrillic letters  (script comparison)
      - Sentences                 (long, variable-length vectors)
      - Punctuation marks         (symbol code points)
      - Fibonacci numbers         (growing string length)
  • "Examples" dropdown in the toolbar + "Examples…" dialog that documents
    what each example demonstrates and which settings pair well with it.
  • Loads the first example by default so the app is never empty.

Requires:  pip install PySide6
Run with:  python unicode_nd_app.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import string
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import (
    QPointF,
    QRectF,
    QSettings,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# ============================================================================
# 1.  Encoding / preprocessing / transform primitives
# ============================================================================

def _encode_unicode(text: str) -> List[int]:
    return [ord(ch) for ch in text]


def _encode_utf8(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="replace"))


def _encode_utf16(text: str) -> List[int]:
    raw = text.encode("utf-16-be", errors="replace")
    return [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw), 2)]


def _encode_ascii(text: str) -> List[int]:
    return [ord(ch) if ord(ch) < 128 else 0x3F for ch in text]


ENCODINGS: Dict[str, Callable[[str], List[int]]] = {
    "Unicode code point": _encode_unicode,
    "UTF-8 bytes":        _encode_utf8,
    "UTF-16 code units":  _encode_utf16,
    "ASCII (lossy)":      _encode_ascii,
}


def _transform_raw(v: Sequence[float]) -> List[float]:
    return [float(x) for x in v]


def _transform_l2(v: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _transform_zscore(v: Sequence[float]) -> List[float]:
    if not v:
        return []
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) or 1.0
    return [(x - m) / s for x in v]


def _transform_minmax(v: Sequence[float]) -> List[float]:
    if not v:
        return []
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    return [(x - lo) / span for x in v]


TRANSFORMS: Dict[str, Callable[[Sequence[float]], List[float]]] = {
    "Raw":           _transform_raw,
    "L2 normalized": _transform_l2,
    "Z-score":       _transform_zscore,
    "Min-max":       _transform_minmax,
}


def _pad_zero(v: List[float], dim: int, pad_value: float = 0.0) -> List[float]:
    return v + [pad_value] * (dim - len(v))


def _pad_repeat(v: List[float], dim: int, pad_value: float = 0.0) -> List[float]:
    if not v:
        return [pad_value] * dim
    return v + [v[i % len(v)] for i in range(dim - len(v))]


def _pad_mirror(v: List[float], dim: int, pad_value: float = 0.0) -> List[float]:
    if not v:
        return [pad_value] * dim
    out = list(v)
    while len(out) < dim:
        out += v[::-1]
    return out[:dim]


def _pad_mean(v: List[float], dim: int, pad_value: float = 0.0) -> List[float]:
    m = (sum(v) / len(v)) if v else pad_value
    return v + [m] * (dim - len(v))


PADDINGS: Dict[str, Callable[[List[float], int, float], List[float]]] = {
    "Zero":   _pad_zero,
    "Repeat": _pad_repeat,
    "Mirror": _pad_mirror,
    "Mean":   _pad_mean,
}


def _dist_euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _dist_manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _dist_chebyshev(a: Sequence[float], b: Sequence[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _dist_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - sum(x * y for x, y in zip(a, b)) / (na * nb)


DISTANCES: Dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "Euclidean": _dist_euclidean,
    "Manhattan": _dist_manhattan,
    "Chebyshev": _dist_chebyshev,
    "Cosine":    _dist_cosine,
}


# ============================================================================
# 2.  Configuration object
# ============================================================================

@dataclass
class DatasetConfig:
    encoding: str = "Unicode code point"
    transform: str = "Raw"
    padding: str = "Zero"
    pad_value: float = 0.0
    lowercase: bool = False
    strip_whitespace: bool = True
    remove_punctuation: bool = False
    distance: str = "Euclidean"
    projection: str = "PCA (2 components)"
    color_mode: str = "By index"
    show_hull: bool = True
    show_box: bool = True
    show_grid: bool = False
    neighbours_k: int = 5


# ============================================================================
# 3.  EXAMPLE DATASETS  (new in v3)
# ============================================================================

@dataclass
class Example:
    name: str
    description: str
    items: List[str]
    suggested: Dict[str, str] = None      # config overrides to try

    def __post_init__(self):
        if self.suggested is None:
            self.suggested = {}


EXAMPLES: List[Example] = [
    Example(
        name="Greek alphabet",
        description=(
            "The 24 Greek letter names. Short, all-ASCII, similar lengths — "
            "a good baseline for PCA projection."
        ),
        items=[
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
            "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
            "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
        ],
        suggested={"projection": "PCA (2 components)", "color_mode": "By index"},
    ),
    Example(
        name="Animal names",
        description=(
            "Animal names of varying length. Shows how padding and length "
            "affect the shape of the n-dimensional cloud."
        ),
        items=[
            "ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "ibex",
            "jay", "kite", "lynx", "mole", "newt", "okapi", "puma", "quail",
            "rat", "seal", "toad", "urchin", "vole", "wolf", "yak", "zebra",
            "albatross", "chimpanzee", "elephant", "flamingo", "giraffe",
        ],
        suggested={"padding": "Repeat", "color_mode": "By length"},
    ),
    Example(
        name="Digits & numbers",
        description=(
            "Numeric strings. Unicode code points for '0'..'9' are contiguous, "
            "so digits form tight, well-separated clusters."
        ),
        items=[
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "20", "25", "42", "50", "99",
            "100", "256", "512", "1024", "2048", "4096",
            "3.14", "2.71", "1.618", "0.577",
        ],
        suggested={"projection": "First two dimensions", "color_mode": "By centroid distance"},
    ),
    Example(
        name="Mixed-case words",
        description=(
            "Same word in lowercase, Title Case and UPPERCASE. Turn on "
            "'lowercase' preprocessing to merge them, or leave it off to see "
            "how case shifts code points by exactly 32."
        ),
        items=[
            "hello", "Hello", "HELLO",
            "world", "World", "WORLD",
            "python", "Python", "PYTHON",
            "unicode", "Unicode", "UNICODE",
            "data", "Data", "DATA",
            "vector", "Vector", "VECTOR",
        ],
        suggested={"lowercase": False, "color_mode": "By first char"},
    ),
    Example(
        name="Emoji",
        description=(
            "Emoji of increasing code-point complexity (some are outside the "
            "BMP). Switch encoding to 'UTF-8 bytes' or 'UTF-16 code units' to "
            "see how the vector dimension jumps."
        ),
        items=[
            "☺", "☹", "★", "♫", "☀", "☁", "☂", "⚡",
            "❤", "✈", "⚽", "⚓", "✂", "✉", "✏", "☎",
            "😀", "😂", "😍", "🤔", "😎", "🥳",
            "🐱", "🐶", "🦊", "🐻", "🐼", "🦁",
            "🍎", "🍌", "🍕", "🍣", "🍰",
            "🚀", "🚗", "✈",
        ],
        suggested={"encoding": "UTF-8 bytes", "padding": "Zero",
                   "color_mode": "By length"},
    ),
    Example(
        name="CJK characters",
        description=(
            "Chinese, Japanese and Korean characters. Each is 1 code point but "
            "3 UTF-8 bytes — a great comparison between encoding schemes."
        ),
        items=[
            "日", "月", "火", "水", "木", "金", "土",
            "山", "川", "田", "人", "口", "目", "耳",
            "東", "西", "南", "北", "中",
            "あ", "い", "う", "え", "お",
            "ア", "イ", "ウ", "エ", "オ",
            "한", "글", "한국", "서울",
        ],
        suggested={"encoding": "UTF-8 bytes", "color_mode": "By length"},
    ),
    Example(
        name="Scripts comparison",
        description=(
            "The same letter 'A' as it appears across scripts: Latin, Greek, "
            "Cyrillic, Hebrew, Arabic, Armenian, Georgian. Shows how code "
            "points from different Unicode blocks scatter."
        ),
        items=[
            "A",      # Latin
            "Α",      # Greek
            "А",      # Cyrillic
            "א",      # Hebrew
            "ا",      # Arabic
            "Ա",      # Armenian
            "ა",      # Georgian
            "अ",      # Devanagari
            "あ",      # Hiragana
            "ᚠ",      # Runic
            "𓀀",      # Egyptian hieroglyph
        ],
        suggested={"encoding": "Unicode code point",
                   "color_mode": "By first char",
                   "projection": "First two dimensions"},
    ),
    Example(
        name="Sentences",
        description=(
            "Short sentences. Long, variable-length vectors — try 'Z-score' or "
            "'L2 normalized' transform to stop length from dominating."
        ),
        items=[
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "Sphinx of black quartz, judge my vow.",
            "The five boxing wizards jump quickly.",
            "Jived fox nymph grabs quick waltz.",
            "Waltz, bad nymph, for quick jigs vex.",
            "Bright vixens jump; dozy fowl quack.",
            "Quick zephyrs blow, vexing daft Jim.",
            "Two driven jocks help fax my big quiz.",
        ],
        suggested={"transform": "Z-score", "padding": "Mean",
                   "color_mode": "By length", "show_hull": False},
    ),
    Example(
        name="Punctuation marks",
        description=(
            "ASCII and Unicode punctuation. Compact code-point ranges make "
            "this useful for testing the convex hull and bounding box."
        ),
        items=list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") +
              ["…", "—", "–", "«", "»", "‹", "›", "“", "”", "‘", "’",
               "¡", "¿", "§", "¶", "†", "‡", "•", "‰", "′", "″"],
        suggested={"projection": "First two dimensions",
                   "color_mode": "By first char",
                   "show_grid": True},
    ),
    Example(
        name="Fibonacci numbers",
        description=(
            "The first Fibonacci numbers as strings. Vector dimension grows "
            "steadily — try 'Mirror' padding to keep structure at every length."
        ),
        items=[
            "1", "1", "2", "3", "5", "8", "13", "21", "34", "55",
            "89", "144", "233", "377", "610", "987", "1597", "2584",
            "4181", "6765", "10946", "17711", "28657", "46368", "75025",
        ],
        suggested={"padding": "Mirror", "color_mode": "By length"},
    ),
]


def example_by_name(name: str) -> Example:
    for ex in EXAMPLES:
        if ex.name == name:
            return ex
    return EXAMPLES[0]


# ============================================================================
# 4.  Core model
# ============================================================================

def _preprocess(text: str, cfg: DatasetConfig) -> str:
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip_whitespace:
        text = "".join(text.split())
    if cfg.remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text


class DataPoint:
    """One dataset row: its text and its numeric vectors."""

    __slots__ = ("index", "raw_text", "text", "base_vector", "vector")

    def __init__(
        self,
        index: int,
        raw_text: str,
        text: str,
        base_vector: List[float],
        vector: List[float],
    ) -> None:
        self.index = index
        self.raw_text = raw_text
        self.text = text
        self.base_vector = base_vector
        self.vector = vector

    @property
    def dim(self) -> int:
        return len(self.vector)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DataPoint(#{self.index}, {self.raw_text!r}, dim={self.dim})"


def build_points(
    texts: Sequence[str], cfg: DatasetConfig
) -> Tuple[List[DataPoint], int]:
    encode = ENCODINGS[cfg.encoding]
    transform = TRANSFORMS[cfg.transform]
    pad = PADDINGS[cfg.padding]

    prepped: List[Tuple[int, str, str, List[float]]] = []
    for i, raw in enumerate(texts):
        p = _preprocess(raw, cfg)
        prepped.append((i, raw, p, list(encode(p))))

    target_dim = max((len(v) for _, _, _, v in prepped), default=0)

    points: List[DataPoint] = []
    for i, raw, p, v in prepped:
        vp = pad(list(v), target_dim, cfg.pad_value)
        vt = transform(vp)
        points.append(DataPoint(i, raw, p, vp, vt))

    return points, target_dim


class NDimensionalObject:
    """Axis-aligned bounding hyperrectangle around the dataset."""

    def __init__(self, points: Sequence[DataPoint]) -> None:
        self.points = list(points)
        self.dim = max((p.dim for p in self.points), default=0)
        self.min_vec, self.max_vec, self.centroid = self._bounds()
        self.std_vec: List[float] = []
        if self.points and self.dim:
            self.std_vec = self._std()

    def _bounds(self) -> Tuple[List[float], List[float], List[float]]:
        if not self.points or self.dim == 0:
            return [], [], []
        d = self.dim
        mins = [math.inf] * d
        maxs = [-math.inf] * d
        sums = [0.0] * d
        for p in self.points:
            v = p.vector
            for i in range(d):
                x = v[i]
                if x < mins[i]:
                    mins[i] = x
                if x > maxs[i]:
                    maxs[i] = x
                sums[i] += x
        n = len(self.points)
        return mins, maxs, [s / n for s in sums]

    def _std(self) -> List[float]:
        d = self.dim
        n = len(self.points) or 1
        c = self.centroid
        sums = [0.0] * d
        for p in self.points:
            v = p.vector
            for i in range(d):
                diff = v[i] - c[i]
                sums[i] += diff * diff
        return [math.sqrt(s / n) for s in sums]

    @property
    def extent(self) -> List[float]:
        return [mx - mn for mn, mx in zip(self.min_vec, self.max_vec)]

    @property
    def volume(self) -> float:
        v = 1.0
        for e in self.extent:
            v *= e
        return v

    @property
    def diagonal(self) -> float:
        return math.sqrt(sum(e * e for e in self.extent))

    @property
    def radius(self) -> float:
        if not self.points or self.dim == 0:
            return 0.0
        c = self.centroid
        return max(
            math.sqrt(sum((p.vector[i] - c[i]) ** 2 for i in range(self.dim)))
            for p in self.points
        )

    def contains(self, vec: Sequence[float]) -> bool:
        return all(
            mn <= x <= mx
            for mn, x, mx in zip(self.min_vec, vec, self.max_vec)
        )

    def corner_count(self) -> int:
        return 2 ** self.dim


# ============================================================================
# 5.  Projection helpers
# ============================================================================

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def pca_axes(
    vectors: Sequence[Sequence[float]],
    k: int = 2,
    iterations: int = 150,
    seed: int = 7,
) -> Tuple[List[List[float]], List[float]]:
    n = len(vectors)
    if n == 0:
        return [], []
    d = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / n for i in range(d)]
    centered = [[v[i] - mean[i] for i in range(d)] for v in vectors]

    rng = random.Random(seed)
    axes: List[Tuple[List[float], float]] = []

    def apply_cov(x: Sequence[float]) -> List[float]:
        y = [0.0] * d
        for row in centered:
            s = 0.0
            for i in range(d):
                s += row[i] * x[i]
            for i in range(d):
                y[i] += row[i] * s
        inv = 1.0 / n
        return [t * inv for t in y]

    for _ in range(min(k, d)):
        x = [rng.uniform(-1.0, 1.0) for _ in range(d)]
        norm = math.sqrt(sum(t * t for t in x)) or 1.0
        x = [t / norm for t in x]
        for _ in range(iterations):
            y = apply_cov(x)
            for u, lu in axes:
                s = sum(y[i] * u[i] for i in range(d))
                for i in range(d):
                    y[i] -= lu * s * u[i]
            norm = math.sqrt(sum(t * t for t in y))
            if norm < 1e-12:
                break
            x = [t / norm for t in y]
            lam = norm
        axes.append((x, lam))

    return [a for a, _ in axes], mean


def random_projection_axes(
    dim: int, seed: int = 13
) -> Tuple[List[List[float]], List[float]]:
    rng = random.Random(seed)
    axes = []
    for _ in range(2):
        v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        axes.append([x / n for x in v])
    return axes, [0.0] * dim


def make_projection(
    vectors: Sequence[Sequence[float]], mode: str, dim: int
) -> Callable[[Sequence[float]], Tuple[float, float]]:
    if mode.startswith("PCA") and len(vectors) >= 2 and dim >= 1:
        axes, mean = pca_axes(vectors, 2)
        while len(axes) < 2:
            axes.append([0.0] * dim)
        a0, a1 = axes[0], axes[1]

        def project(v: Sequence[float]) -> Tuple[float, float]:
            c = [v[i] - mean[i] for i in range(dim)]
            return (_dot(c, a0), _dot(c, a1))

        return project

    if mode.startswith("Random"):
        axes, _ = random_projection_axes(dim)

        def project(v: Sequence[float]) -> Tuple[float, float]:
            return (_dot(v, axes[0]), _dot(v, axes[1]))

        return project

    if mode.startswith("First"):
        i0, i1 = 0, (1 if dim >= 2 else 0)
    else:
        i0, i1 = max(dim - 2, 0), max(dim - 1, 0)

    return lambda v: (float(v[i0]), float(v[i1]))


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# ============================================================================
# 6.  Visualisation widget
# ============================================================================

class NDView(QWidget):
    pointSelected = Signal(int)
    pointHovered  = Signal(int)

    PAD = 46
    MIN_ZOOM = 0.1
    MAX_ZOOM = 200.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(500, 420)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.points: List[DataPoint] = []
        self.obj: Optional[NDimensionalObject] = None

        self.proj_mode: str = "PCA (2 components)"
        self.color_mode: str = "By index"
        self.show_hull = True
        self.show_box = True
        self.show_grid = False

        self.selected: Optional[int] = None
        self.hovered: Optional[int] = None
        self.filter_mask: Optional[set] = None

        self._view_cx = 0.0
        self._view_cy = 0.0
        self._view_scale = 1.0
        self._zoomed = False

        self._screen: List[QPointF] = []
        self._centroid_screen: Optional[QPointF] = None

        self._panning = False
        self._pan_start = QPointF()
        self._pan_origin = QPointF()

    def set_data(
        self,
        points: List[DataPoint],
        obj: Optional[NDimensionalObject],
        proj_mode: str = "PCA (2 components)",
        color_mode: str = "By index",
        show_hull: bool = True,
        show_box: bool = True,
        show_grid: bool = False,
        filter_mask: Optional[set] = None,
    ) -> None:
        self.points = points
        self.obj = obj
        self.proj_mode = proj_mode
        self.color_mode = color_mode
        self.show_hull = show_hull
        self.show_box = show_box
        self.show_grid = show_grid
        self.filter_mask = filter_mask
        self.reset_view()
        self.update()

    def set_selected(self, index: Optional[int]) -> None:
        self.selected = index
        self.update()

    def set_filter(self, mask: Optional[set]) -> None:
        self.filter_mask = mask
        self.update()

    def reset_view(self) -> None:
        self._zoomed = False
        self._recompute(auto_fit=True)
        self.update()

    def _color_for(self, i: int, dp: DataPoint) -> QColor:
        n = max(len(self.points), 1)
        mode = self.color_mode
        if mode == "By length":
            t = dp.dim / max((p.dim for p in self.points), default=1)
            hue = int(210 * t) % 360
            return QColor.fromHsv(hue, 200, 235)
        if mode == "By first char" and dp.text:
            hue = (ord(dp.text[0]) * 47) % 360
            return QColor.fromHsv(hue, 190, 235)
        if mode == "By centroid distance" and self.obj is not None:
            c = self.obj.centroid
            d = math.sqrt(sum((dp.vector[j] - c[j]) ** 2 for j in range(dp.dim)))
            r = self.obj.radius or 1.0
            hue = int(240 * min(d / r, 1.0)) % 360
            return QColor.fromHsv(hue, 200, 235)
        hue = int(200 + 140 * (i / n)) % 360
        return QColor.fromHsv(hue, 190, 235)

    def _to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(
            self.width() / 2.0 + (x - self._view_cx) * self._view_scale,
            self.height() / 2.0 - (y - self._view_cy) * self._view_scale,
        )

    def _to_data(self, sx: float, sy: float) -> Tuple[float, float]:
        return (
            self._view_cx + (sx - self.width() / 2.0) / self._view_scale,
            self._view_cy - (sy - self.height() / 2.0) / self._view_scale,
        )

    def _recompute(self, auto_fit: bool = False) -> None:
        self._screen = []
        self._centroid_screen = None
        if not self.points or self.obj is None or self.obj.dim == 0:
            return

        dim = self.obj.dim
        vecs = [p.vector for p in self.points]
        project = make_projection(vecs, self.proj_mode, dim)
        coords = [project(v) for v in vecs]
        centroid_coord = project(self.obj.centroid)

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)

        if auto_fit or not self._zoomed:
            w = max(self.width(), 80)
            h = max(self.height(), 80)
            s = min((w - 2 * self.PAD) / dx, (h - 2 * self.PAD) / dy)
            self._view_scale = s if (math.isfinite(s) and s > 0) else 1.0
            self._view_cx = (minx + maxx) / 2.0
            self._view_cy = (miny + maxy) / 2.0

        self._screen = [self._to_screen(x, y) for (x, y) in coords]
        self._centroid_screen = self._to_screen(*centroid_coord)

    def _visible_indices(self) -> List[int]:
        if self.filter_mask is None:
            return list(range(len(self.points)))
        return [i for i, dp in enumerate(self.points) if dp.index in self.filter_mask]

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._recompute(auto_fit=not self._zoomed)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self.points:
            return
        pos = event.position()
        before = self._to_data(pos.x(), pos.y())
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_scale = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._view_scale * factor))
        self._view_scale = new_scale
        after = self._to_data(pos.x(), pos.y())
        self._view_cx += before[0] - after[0]
        self._view_cy += before[1] - after[1]
        self._zoomed = True
        self._recompute()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_origin = QPointF(self._view_cx, self._view_cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._nearest_index(event.position())
            if idx is not None:
                self.selected = self.points[idx].index
                self.pointSelected.emit(self.selected)
                self.update()
            else:
                self._panning = True
                self._pan_start = event.position()
                self._pan_origin = QPointF(self._view_cx, self._view_cy)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._panning:
            dx = pos.x() - self._pan_start.x()
            dy = pos.y() - self._pan_start.y()
            self._view_cx = self._pan_origin.x() - dx / self._view_scale
            self._view_cy = self._pan_origin.y() + dy / self._view_scale
            self._recompute()
            self.update()
            return

        idx = self._nearest_index(pos)
        new_hover = self.points[idx].index if idx is not None else None
        if new_hover != self.hovered:
            self.hovered = new_hover
            if new_hover is not None:
                self.pointHovered.emit(new_hover)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._zoomed = True

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self.hovered is not None:
            self.hovered = None
            self.update()

    def _nearest_index(self, pos: QPointF) -> Optional[int]:
        if not self._screen:
            return None
        best, best_d = None, 20 * 20
        for i in self._visible_indices():
            pt = self._screen[i]
            d = (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2
            if d < best_d:
                best_d, best = d, i
        return best

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0d1117"))

        if not self._screen:
            p.setPen(QColor("#7d8590"))
            f = QFont()
            f.setPointSize(11)
            p.setFont(f)
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No data loaded.\nUse an example or “Open CSV…”.\n"
                "Tip: pick an example from the toolbar dropdown.",
            )
            return

        if self.show_grid:
            self._draw_grid(p)

        vis = self._visible_indices()
        xs = [self._screen[i].x() for i in vis]
        ys = [self._screen[i].y() for i in vis]

        if self.show_box and xs and ys:
            rect = QRectF(
                min(xs) - 10, min(ys) - 10,
                (max(xs) - min(xs)) + 20, (max(ys) - min(ys)) + 20,
            )
            p.setPen(QPen(QColor("#4b5563"), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)

        if self.show_hull and len(vis) >= 3:
            hull = convex_hull([(self._screen[i].x(), self._screen[i].y()) for i in vis])
            if len(hull) >= 3:
                poly = QPolygonF([QPointF(x, y) for x, y in hull])
                p.setPen(QPen(QColor("#f0883e"), 2))
                p.setBrush(QBrush(QColor(240, 136, 62, 28)))
                p.drawPolygon(poly)

        if self._centroid_screen is not None:
            c = self._centroid_screen
            p.setPen(QPen(QColor("#f0f6fc"), 2))
            p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))

        for i in vis:
            dp = self.points[i]
            pt = self._screen[i]
            is_sel = self.selected is not None and dp.index == self.selected
            is_hov = self.hovered is not None and dp.index == self.hovered

            color = self._color_for(i, dp)
            radius = 7.0 if is_sel else (5.5 if is_hov else 4.0)

            if is_sel or is_hov:
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(pt, radius + 5, radius + 5)

            p.setPen(QPen(QColor("#0d1117"), 1))
            p.setBrush(QBrush(color))
            p.drawEllipse(pt, radius, radius)

            if len(vis) <= 120:
                p.setPen(QColor("#8b949e"))
                f = QFont()
                f.setPointSize(7)
                p.setFont(f)
                p.drawText(QPointF(pt.x() + 8, pt.y() - 6), str(dp.index))

        if self.hovered is not None:
            self._draw_tooltip(p, self.hovered)

        info = (
            f"points: {len(vis)}/{len(self.points)}   "
            f"dim: {self.obj.dim if self.obj else 0}   "
            f"zoom: {self._view_scale:.2f}×   "
            f"{self.proj_mode}"
        )
        p.setPen(QColor("#c9d1d9"))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(QPointF(12, 20), info)

    def _draw_grid(self, p: QPainter) -> None:
        pen = QPen(QColor(255, 255, 255, 18), 1)
        p.setPen(pen)
        step = 60
        for x in range(0, self.width(), step):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            p.drawLine(0, y, self.width(), y)

    def _draw_tooltip(self, p: QPainter, index: int) -> None:
        try:
            i = next(i for i, dp in enumerate(self.points) if dp.index == index)
        except StopIteration:
            return
        dp = self.points[i]
        pt = self._screen[i]

        lines = [
            f"#{dp.index}   dim={dp.dim}",
            (dp.raw_text if len(dp.raw_text) <= 34 else dp.raw_text[:31] + "…"),
        ]
        text = "\n".join(lines)

        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(ln) for ln in lines) + 16
        h = fm.height() * len(lines) + 12

        tx = pt.x() + 14
        ty = pt.y() + 14
        if tx + w > self.width() - 6:
            tx = self.width() - w - 6
        if ty + h > self.height() - 6:
            ty = self.height() - h - 6

        rect = QRectF(tx, ty, w, h)
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(QBrush(QColor(22, 27, 34, 240)))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QColor("#e6edf3"))
        p.drawText(
            rect.adjusted(8, 6, -8, -6),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            text,
        )


# ============================================================================
# 7.  Examples dialog
# ============================================================================

class ExamplesDialog(QDialog):
    """Browse the built-in examples, read their descriptions, load one."""

    exampleChosen = Signal(str)   # example name

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Built-in examples")
        self.resize(760, 460)

        layout = QHBoxLayout(self)

        self.list = QListWidget()
        self.list.setMinimumWidth(230)
        for ex in EXAMPLES:
            it = QListWidgetItem(f"{ex.name}\n({len(ex.items)} items)")
            it.setData(Qt.ItemDataRole.UserRole, ex.name)
            self.list.addItem(it)
        self.list.currentRowChanged.connect(self._update_preview)
        layout.addWidget(self.list)

        right = QVBoxLayout()
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        right.addWidget(self.browser, 1)

        btns = QDialogButtonBox()
        self.btn_load = btns.addButton("Load example", QDialogButtonBox.ButtonRole.AcceptRole)
        btns.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        btns.accepted.connect(self._accept_current)
        btns.rejected.connect(self.reject)
        right.addWidget(btns)

        layout.addLayout(right, 1)

        self.list.setCurrentRow(0)

    def _update_preview(self, row: int) -> None:
        if row < 0 or row >= len(EXAMPLES):
            self.browser.setHtml("")
            return
        ex = EXAMPLES[row]
        sample_items = ex.items[:10]
        sample_html = "".join(
            f"<li><code>{self._esc(s)}</code></li>" for s in sample_items
        )
        if len(ex.items) > 10:
            sample_html += f"<li>… and {len(ex.items) - 10} more</li>"

        suggested_rows = "".join(
            f"<tr><td><b>{k}</b></td><td><code>{v}</code></td></tr>"
            for k, v in ex.suggested.items()
        )
        suggested_html = (
            f"<h4>Suggested settings</h4><table>{suggested_rows}</table>"
            if suggested_rows else "<p><i>No suggested settings.</i></p>"
        )

        html = f"""
        <h2>{self._esc(ex.name)}</h2>
        <p>{self._esc(ex.description)}</p>
        <p><b>Items:</b> {len(ex.items)}</p>
        <h4>Sample</h4>
        <ul>{sample_html}</ul>
        {suggested_html}
        """
        self.browser.setHtml(html)

    @staticmethod
    def _esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    def _accept_current(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        self.exampleChosen.emit(it.data(Qt.ItemDataRole.UserRole))
        self.accept()


# ============================================================================
# 8.  Main window
# ============================================================================

PROJECTIONS = [
    "PCA (2 components)",
    "Random projection",
    "First two dimensions",
    "Last two dimensions",
]

COLOR_MODES = [
    "By index",
    "By length",
    "By first char",
    "By centroid distance",
]


def fmt_vec(v: Sequence[float], limit: int = 10, nd: int = 2) -> str:
    if not v:
        return "[]"
    shown = ", ".join(f"{x:.{nd}f}" for x in v[:limit])
    if len(v) > limit:
        shown += f", … (+{len(v) - limit} more)"
    return "[" + shown + "]"


def load_texts_from_file(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        for row in csv.reader(fh, dialect):
            if not row:
                continue
            cell = row[0].strip()
            if cell:
                texts.append(cell)
    header_words = {"text", "data", "datapoint", "data point", "word", "label", "value"}
    if texts and texts[0].lower() in header_words:
        texts = texts[1:]
    return texts


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Unicode → n-Dimensional Data Explorer")
        self.resize(1360, 820)

        self.settings = QSettings("unicode-nd", "explorer")
        self.cfg = DatasetConfig()
        self._load_settings()

        self.points: List[DataPoint] = []
        self.obj: Optional[NDimensionalObject] = None
        self.raw_texts: List[str] = []
        self.current_example: str = ""

        self._build_ui()
        self._apply_config_to_widgets()
        self._populate_examples_combo()

        # Load the first example so the app is never empty.
        self.load_example(EXAMPLES[0].name)

    # ------------------------------------------------------------- settings
    def _load_settings(self) -> None:
        s = self.settings
        self.cfg.encoding = s.value("encoding", self.cfg.encoding, type=str)
        self.cfg.transform = s.value("transform", self.cfg.transform, type=str)
        self.cfg.padding = s.value("padding", self.cfg.padding, type=str)
        self.cfg.pad_value = s.value("pad_value", self.cfg.pad_value, type=float)
        self.cfg.lowercase = s.value("lowercase", self.cfg.lowercase, type=bool)
        self.cfg.strip_whitespace = s.value("strip_whitespace", self.cfg.strip_whitespace, type=bool)
        self.cfg.remove_punctuation = s.value("remove_punctuation", self.cfg.remove_punctuation, type=bool)
        self.cfg.distance = s.value("distance", self.cfg.distance, type=str)
        self.cfg.projection = s.value("projection", self.cfg.projection, type=str)
        self.cfg.color_mode = s.value("color_mode", self.cfg.color_mode, type=str)
        self.cfg.show_hull = s.value("show_hull", self.cfg.show_hull, type=bool)
        self.cfg.show_box = s.value("show_box", self.cfg.show_box, type=bool)
        self.cfg.show_grid = s.value("show_grid", self.cfg.show_grid, type=bool)
        self.cfg.neighbours_k = s.value("neighbours_k", self.cfg.neighbours_k, type=int)
        self.current_example = s.value("current_example", "", type=str)

        valid = {
            "encoding": set(ENCODINGS),
            "transform": set(TRANSFORMS),
            "padding": set(PADDINGS),
            "distance": set(DISTANCES),
            "projection": set(PROJECTIONS),
            "color_mode": set(COLOR_MODES),
        }
        for key, allowed in valid.items():
            if getattr(self.cfg, key) not in allowed:
                setattr(self.cfg, key, DatasetConfig().__dict__[key])

    def _save_settings(self) -> None:
        s = self.settings
        for k, v in self.cfg.__dict__.items():
            s.setValue(k, v)
        s.setValue("current_example", self.current_example)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self._build_toolbar()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([640, 720])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Ready — choose an example from the toolbar.")

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(16, 16))
        self.addToolBar(bar)

        def add(name, shortcut, slot):
            a = QAction(name, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            bar.addAction(a)
            return a

        add("Open CSV…", "Ctrl+O", self.open_csv)
        add("Clear", None, self.clear_data)
        bar.addSeparator()

        # --- EXAMPLES dropdown -------------------------------------------
        bar.addWidget(QLabel("  Example:  "))
        self.cmb_examples = QComboBox()
        self.cmb_examples.setMinimumWidth(180)
        self.cmb_examples.setToolTip(
            "Load one of the built-in example datasets.\n"
            "See “Examples…” for descriptions and suggested settings."
        )
        self.cmb_examples.currentIndexChanged.connect(self._on_example_combo)
        bar.addWidget(self.cmb_examples)

        add("Examples…", "Ctrl+E", self.open_examples_dialog)
        bar.addSeparator()

        add("Export vectors…", "Ctrl+S", self.export_vectors)
        add("Export plot…", "Ctrl+Shift+S", self.export_plot)
        bar.addSeparator()
        add("Reset view", "Home", lambda: self.view.reset_view())

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # --- Description banner ------------------------------------------
        self.lbl_example = QLabel()
        self.lbl_example.setWordWrap(True)
        self.lbl_example.setStyleSheet(
            "QLabel { background: #161b22; color: #c9d1d9; padding: 8px;"
            "border: 1px solid #30363d; border-radius: 4px; }"
        )
        v.addWidget(self.lbl_example)

        # --- Pipeline configuration --------------------------------------
        grp = QGroupBox("Pipeline configuration")
        form = QFormLayout(grp)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        self.cmb_encoding = QComboBox()
        self.cmb_encoding.addItems(ENCODINGS.keys())
        self.cmb_encoding.currentTextChanged.connect(self._on_cfg_changed)
        form.addRow("Encoding", self.cmb_encoding)

        self.cmb_padding = QComboBox()
        self.cmb_padding.addItems(PADDINGS.keys())
        self.cmb_padding.currentTextChanged.connect(self._on_cfg_changed)
        form.addRow("Padding", self.cmb_padding)

        self.cmb_transform = QComboBox()
        self.cmb_transform.addItems(TRANSFORMS.keys())
        self.cmb_transform.currentTextChanged.connect(self._on_cfg_changed)
        form.addRow("Transform", self.cmb_transform)

        self.chk_lower = QCheckBox("lowercase")
        self.chk_lower.toggled.connect(self._on_cfg_changed)
        self.chk_strip = QCheckBox("strip whitespace")
        self.chk_strip.toggled.connect(self._on_cfg_changed)
        self.chk_punct = QCheckBox("remove punctuation")
        self.chk_punct.toggled.connect(self._on_cfg_changed)
        form.addRow("Preprocess", self.chk_lower)
        form.addRow("", self.chk_strip)
        form.addRow("", self.chk_punct)

        v.addWidget(grp)

        # --- View options -------------------------------------------------
        grp2 = QGroupBox("View")
        form2 = QFormLayout(grp2)
        form2.setContentsMargins(8, 8, 8, 8)
        form2.setSpacing(4)

        self.cmb_proj = QComboBox()
        self.cmb_proj.addItems(PROJECTIONS)
        self.cmb_proj.currentTextChanged.connect(self._on_view_cfg_changed)
        form2.addRow("Projection", self.cmb_proj)

        self.cmb_color = QComboBox()
        self.cmb_color.addItems(COLOR_MODES)
        self.cmb_color.currentTextChanged.connect(self._on_view_cfg_changed)
        form2.addRow("Colors", self.cmb_color)

        self.chk_hull = QCheckBox("convex hull")
        self.chk_hull.toggled.connect(self._on_view_cfg_changed)
        self.chk_box = QCheckBox("bounding box")
        self.chk_box.toggled.connect(self._on_view_cfg_changed)
        self.chk_grid = QCheckBox("grid")
        self.chk_grid.toggled.connect(self._on_view_cfg_changed)
        form2.addRow("Show", self.chk_hull)
        form2.addRow("", self.chk_box)
        form2.addRow("", self.chk_grid)

        v.addWidget(grp2)

        # --- Search + table ----------------------------------------------
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Filter:"))
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("substring of the data point…")
        self.ed_search.textChanged.connect(self._on_search)
        search_row.addWidget(self.ed_search)
        v.addLayout(search_row)

        v.addWidget(QLabel("<b>Data points</b> — one <code>DataPoint</code> object per row"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Text", "Dim", "Vector", "Preview"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(self._on_row_changed)
        v.addWidget(self.table, 3)

        v.addWidget(QLabel("<b>Selected object</b>"))
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(QFont("Menlo, Consolas, monospace", 9))
        v.addWidget(self.details, 2)

        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        self.view = NDView()
        self.view.pointSelected.connect(self.select_point)
        v.addWidget(self.view, 1)

        v.addWidget(QLabel("<b>n-dimensional object</b>  (bounding hyperrectangle)"))
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(280)
        self.info.setFont(QFont("Menlo, Consolas, monospace", 9))
        v.addWidget(self.info)

        return w

    def _populate_examples_combo(self) -> None:
        self.cmb_examples.blockSignals(True)
        self.cmb_examples.clear()
        self.cmb_examples.addItem("— pick an example —", "")
        for ex in EXAMPLES:
            self.cmb_examples.addItem(ex.name, ex.name)
        self.cmb_examples.blockSignals(False)

    def _apply_config_to_widgets(self) -> None:
        c = self.cfg
        for cmb, val in (
            (self.cmb_encoding, c.encoding),
            (self.cmb_padding, c.padding),
            (self.cmb_transform, c.transform),
            (self.cmb_proj, c.projection),
            (self.cmb_color, c.color_mode),
        ):
            cmb.blockSignals(True)
            cmb.setCurrentText(val)
            cmb.blockSignals(False)
        for chk, val in (
            (self.chk_lower, c.lowercase),
            (self.chk_strip, c.strip_whitespace),
            (self.chk_punct, c.remove_punctuation),
            (self.chk_hull, c.show_hull),
            (self.chk_box, c.show_box),
            (self.chk_grid, c.show_grid),
        ):
            chk.blockSignals(True)
            chk.setChecked(val)
            chk.blockSignals(False)

    # ------------------------------------------------------------- actions
    def _on_cfg_changed(self, *_):
        self.cfg.encoding = self.cmb_encoding.currentText()
        self.cfg.padding = self.cmb_padding.currentText()
        self.cfg.transform = self.cmb_transform.currentText()
        self.cfg.lowercase = self.chk_lower.isChecked()
        self.cfg.strip_whitespace = self.chk_strip.isChecked()
        self.cfg.remove_punctuation = self.chk_punct.isChecked()
        self._rebuild_model()
        self._save_settings()

    def _on_view_cfg_changed(self, *_):
        self.cfg.projection = self.cmb_proj.currentText()
        self.cfg.color_mode = self.cmb_color.currentText()
        self.cfg.show_hull = self.chk_hull.isChecked()
        self.cfg.show_box = self.chk_box.isChecked()
        self.cfg.show_grid = self.chk_grid.isChecked()
        self._refresh_view()
        self._save_settings()

    def _on_search(self, text: str):
        needle = text.strip().lower()
        if not needle:
            self._mask = None
        else:
            self._mask = {
                dp.index for dp in self.points if needle in dp.raw_text.lower()
            }
        self._filter_table(needle)
        self.view.set_filter(self._mask)

    def _filter_table(self, needle: str) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            show = (not needle) or (needle in (item.text() if item else "").lower())
            self.table.setRowHidden(r, not show)

    # -------------------------------------------------------------- dataset
    def open_csv(self) -> None:
        start_dir = self.settings.value("last_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open dataset", start_dir,
            "Tabular files (*.csv *.tsv *.txt);;All files (*)"
        )
        if not path:
            return
        try:
            texts = load_texts_from_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read file", str(exc))
            return
        if not texts:
            QMessageBox.information(self, "Empty file", "No data points were found.")
            return
        import os
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.raw_texts = texts
        self.current_example = ""
        self.cmb_examples.blockSignals(True)
        self.cmb_examples.setCurrentIndex(0)
        self.cmb_examples.blockSignals(False)
        self._rebuild_model()
        self._set_example_banner(
            f"Custom file: {os.path.basename(path)} ({len(texts)} data points)"
        )
        self.statusBar().showMessage(f"Loaded {len(texts)} data points from {path}")

    def load_example(self, name: str) -> None:
        ex = example_by_name(name)
        if ex is None:
            return
        self.raw_texts = list(ex.items)
        self.current_example = ex.name

        # Apply suggested config overrides.
        suggested = ex.suggested or {}
        for k, v in suggested.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self._apply_config_to_widgets()

        self._rebuild_model()
        self._set_example_banner(
            f"<b>Example:</b> {ex.name} &nbsp;•&nbsp; {len(ex.items)} items<br>"
            f"<span style='color:#8b949e'>{ex.description}</span>"
        )
        self.statusBar().showMessage(
            f"Loaded example “{ex.name}” ({len(ex.items)} data points)"
        )

        # Sync combo without re-triggering.
        self.cmb_examples.blockSignals(True)
        idx = self.cmb_examples.findData(ex.name)
        if idx >= 0:
            self.cmb_examples.setCurrentIndex(idx)
        self.cmb_examples.blockSignals(False)

    def _set_example_banner(self, html: str) -> None:
        self.lbl_example.setText(html)

    def _on_example_combo(self, row: int) -> None:
        name = self.cmb_examples.itemData(row)
        if not name:
            return
        self.load_example(name)

    def open_examples_dialog(self) -> None:
        dlg = ExamplesDialog(self)
        dlg.exampleChosen.connect(self.load_example)
        dlg.exec()

    def clear_data(self) -> None:
        self.raw_texts = []
        self.points = []
        self.obj = None
        self.current_example = ""
        self.table.setRowCount(0)
        self.details.clear()
        self.view.set_data([], None)
        self._set_example_banner(
            "No dataset loaded. Pick an example from the toolbar or open a CSV."
        )
        self._update_info()
        self.statusBar().showMessage("Cleared")

    def _rebuild_model(self) -> None:
        if not self.raw_texts:
            self.points, self.obj = [], None
        else:
            self.points, _ = build_points(self.raw_texts, self.cfg)
            self.obj = NDimensionalObject(self.points)

        self._populate_table()
        self._refresh_view()
        self._update_info()
        self._on_search(self.ed_search.text())

    def _populate_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.points))
        for r, dp in enumerate(self.points):
            self.table.setItem(r, 0, QTableWidgetItem(str(dp.index)))
            self.table.setItem(r, 1, QTableWidgetItem(dp.raw_text))
            self.table.setItem(r, 2, QTableWidgetItem(str(dp.dim)))

            if dp.base_vector:
                shown = ", ".join(f"{x:g}" for x in dp.base_vector[:8])
                if dp.dim > 8:
                    shown += ", …"
            else:
                shown = "(empty)"
            self.table.setItem(r, 3, QTableWidgetItem(shown))

            preview = dp.text[:24] + ("…" if len(dp.text) > 24 else "")
            self.table.setItem(r, 4, QTableWidgetItem(preview))
        self.table.blockSignals(False)

        if self.points:
            self.table.selectRow(0)
            self._on_row_changed(0, 0)

    def _refresh_view(self) -> None:
        self.view.set_data(
            self.points,
            self.obj,
            proj_mode=self.cfg.projection,
            color_mode=self.cfg.color_mode,
            show_hull=self.cfg.show_hull,
            show_box=self.cfg.show_box,
            show_grid=self.cfg.show_grid,
            filter_mask=getattr(self, "_mask", None),
        )
        if self.points:
            self.view.set_selected(self.points[0].index)

    # -------------------------------------------------------------- info
    def _update_info(self) -> None:
        if not self.points or self.obj is None:
            self.info.setPlainText("No dataset loaded.")
            return
        o = self.obj
        n = len(self.points)
        inside = sum(1 for p in self.points if o.contains(p.vector))

        lines = [
            f"data points          : {n}",
            f"ambient dimension    : {o.dim}",
            f"hyperrectangle corns : {o.corner_count():,}",
            "",
            f"centroid             : {fmt_vec(o.centroid)}",
            f"min                  : {fmt_vec(o.min_vec)}",
            f"max                  : {fmt_vec(o.max_vec)}",
            f"extent               : {fmt_vec(o.extent)}",
            f"std dev              : {fmt_vec(o.std_vec)}",
            f"volume               : {o.volume:,.4g}",
            f"diagonal length      : {o.diagonal:,.4f}",
            f"radius (from centre) : {o.radius:,.4f}",
            f"contains             : {inside}/{n} data points",
        ]
        self.info.setPlainText("\n".join(lines))

    # ------------------------------------------------------------- selection
    def select_point(self, index: int) -> None:
        for r, dp in enumerate(self.points):
            if dp.index == index:
                self.table.selectRow(r)
                break

    def _on_row_changed(self, row: int, _col: int) -> None:
        if not (0 <= row < len(self.points)):
            return
        dp = self.points[row]
        self.view.set_selected(dp.index)

        lines = [
            f"DataPoint object #{dp.index}",
            f"  raw text   : {dp.raw_text!r}",
            f"  processed  : {dp.text!r}",
            f"  dimension  : {dp.dim}",
            f"  base vec   : {fmt_vec(dp.base_vector, limit=16)}",
            f"  final vec  : {fmt_vec(dp.vector, limit=16, nd=3)}",
        ]
        if dp.base_vector:
            lines.append(
                "  hex        : "
                + " ".join(f"0x{int(c):04X}" for c in dp.base_vector[:12])
                + (" …" if dp.dim > 12 else "")
            )
        if dp.raw_text:
            lines.append(
                "  chars      : "
                + "  ".join(f"{c}=U+{ord(c):04X}" for c in dp.raw_text[:8])
                + (" …" if len(dp.raw_text) > 8 else "")
            )

        if len(self.points) > 1:
            metric = DISTANCES[self.cfg.distance]
            sims = []
            for other in self.points:
                if other.index == dp.index:
                    continue
                sims.append((metric(dp.vector, other.vector), other))
            sims.sort(key=lambda t: t[0])
            k = max(1, min(self.cfg.neighbours_k, len(sims)))
            lines.append("")
            lines.append(f"  {k} nearest neighbours ({self.cfg.distance}):")
            for d, other in sims[:k]:
                preview = other.raw_text[:34] + ("…" if len(other.raw_text) > 34 else "")
                lines.append(f"    d={d:8.4f}   #{other.index:<4}  {preview}")

        self.details.setPlainText("\n".join(lines))

    # -------------------------------------------------------------- export
    def export_vectors(self) -> None:
        if not self.points:
            QMessageBox.information(self, "Nothing to export", "Load a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export vectors", "vectors.csv", "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                payload = [
                    {
                        "index": dp.index,
                        "raw_text": dp.raw_text,
                        "processed_text": dp.text,
                        "dim": dp.dim,
                        "base_vector": dp.base_vector,
                        "vector": dp.vector,
                    }
                    for dp in self.points
                ]
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "config": self.cfg.__dict__,
                            "example": self.current_example,
                            "ambient_dim": self.obj.dim if self.obj else 0,
                            "points": payload,
                        },
                        fh,
                        indent=2,
                    )
            else:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    dim = self.obj.dim if self.obj else 0
                    writer.writerow(
                        ["index", "raw_text", "dim"] + [f"v{i}" for i in range(dim)]
                    )
                    for dp in self.points:
                        row = [dp.index, dp.raw_text, dp.dim] + list(dp.vector)
                        row += [""] * (3 + dim - len(row))
                        writer.writerow(row)
            self.statusBar().showMessage(f"Exported vectors to {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_plot(self) -> None:
        if not self.points:
            QMessageBox.information(self, "Nothing to export", "Load a dataset first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "plot.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        pix = self.view.grab()
        if not pix.save(path):
            QMessageBox.critical(self, "Export failed", "Could not write image file.")
        else:
            self.statusBar().showMessage(f"Exported plot to {path}")


# ============================================================================
# 9.  Entry point
# ============================================================================

def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Unicode ND Explorer")
    app.setOrganizationName("unicode-nd")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())