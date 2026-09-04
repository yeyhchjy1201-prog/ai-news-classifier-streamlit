"""繁體中文新聞的輕量、離線關鍵字提取器。

用途：從單篇新聞選出代表性詞組，供網頁介面與命令列顯示。使用方式：
``extract_keywords("新聞內容", top_k=5)``；可用 ``stopwords`` 排除詞彙，
或用 ``domain_terms`` 加入特定領域詞彙。

名詞：CJK 代表中日韓統一表意文字；n-gram 是連續字元組合；停用詞是常見但
辨識度低的詞；NFKC 是 Unicode 文字正規化方式。分數僅表示同一篇文章內的
相對排序，不是機率，也不可跨文章比較；此為規則式方法，並非語意理解模型。

Lightweight, offline keyword extraction for Traditional Chinese news.

The module deliberately uses only Python's standard library.  It does not
download a tokenizer or model at runtime, which makes it suitable for a small
news-classification service, an air-gapped environment, or a first-run
fallback when an optional NLP dependency is unavailable.

Chinese has no mandatory word separators, so the fallback combines several
signals instead of pretending that character n-grams are a full tokenizer:

* short Chinese phrases from grammatical chunks;
* English, acronym, and numeric tokens (for example ``AI`` and ``GPT-4``);
* a modest built-in news vocabulary plus caller-supplied domain terms; and
* document frequency, headline presence, phrase length, and phrase matches.

The public :func:`extract_keywords` function returns JSON-serialisable
``(term, score)`` pairs, a format also accepted by the accompanying web UI.
Its scores are relative values normalized to ``0.0``--``1.0`` for that one
document.  A score is useful for ranking keywords within a document; it is not
a probability and should not be compared across unrelated documents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Sequence
import unicodedata

__all__ = [
    "Keyword",
    "ChineseKeywordExtractor",
    "extract_keywords",
    "extract_keyword_terms",
]


# CJK Unified Ideographs plus Extension A covers the overwhelming majority of
# Traditional-Chinese news text.  Keeping this separate from Latin tokens lets
# us preserve terms such as "AI", "GPT-4", and "5G" as complete keywords.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_OR_NUMBER_RE = re.compile(
    r"(?<![\w.-])(?:[A-Za-z][A-Za-z0-9+#._/-]*|\d+(?:[.,]\d+)?(?:%|[A-Za-z]+|年|月|日|億元?|萬|兆)?)"
)
_SENTENCE_BREAK_RE = re.compile(r"[。！？!?；;\r\n]+")
_MAX_NGRAM_LENGTH = 12
_MAX_PHRASE_LENGTH = 32


# These are phrase-level stopwords.  They are intentionally conservative:
# topic words such as "政府" and "市場" are allowed because they can be useful
# keywords in short news articles.
_DEFAULT_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "與",
        "和",
        "及",
        "或",
        "而",
        "但",
        "並",
        "且",
        "也",
        "都",
        "很",
        "更",
        "最",
        "再",
        "仍",
        "又",
        "已",
        "將",
        "被",
        "把",
        "對",
        "向",
        "從",
        "由",
        "為",
        "於",
        "以",
        "用",
        "這",
        "那",
        "該",
        "此",
        "其",
        "本",
        "每",
        "各",
        "任何",
        "所有",
        "目前",
        "近日",
        "今日",
        "昨天",
        "明天",
        "今年",
        "去年",
        "明年",
        "表示",
        "指出",
        "認為",
        "強調",
        "提到",
        "透露",
        "說明",
        "報導",
        "消息",
        "新聞",
        "相關",
        "部分",
        "方面",
        "情況",
        "問題",
        "內容",
        "需求",
        "分析",
        "公司",
        "民眾",
        "地方",
        "部分",
        "地區",
        "團隊",
        "進行",
        "提供",
        "使用",
        "成為",
        "可能",
        "可以",
        "需要",
        "希望",
        "持續",
        "開始",
        "完成",
        "宣布",
        "其中",
        "以及",
        "一個",
        "一些",
        "我們",
        "他們",
        "這項",
        "該項",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "from",
        "by",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
    }
)


# A candidate beginning or ending with one of these function characters is
# usually a fragment created by n-gram extraction (for example "的新聞"), not a
# standalone keyword.  "中" is deliberately not included: it is significant
# in names such as "中國" and "中華電信".
_EDGE_STOP_CHARS = frozenset(
    {
        "的",
        "了",
        "在",
        "於",
        "與",
        "和",
        "及",
        "或",
        "而",
        "但",
        "並",
        "且",
        "也",
        "都",
        "很",
        "更",
        "最",
        "再",
        "仍",
        "又",
        "已",
        "將",
        "被",
        "把",
        "對",
        "向",
        "從",
        "由",
        "為",
        "以",
        "用",
        "這",
        "那",
        "該",
        "此",
        "其",
        "本",
        "每",
        "各",
        "則",
        "讓",
        "給",
    }
)


# Split long CJK runs at grammatical connectors and reporting verbs.  This is
# not intended to be a linguistic parser; it simply prevents phrases such as
# "政府宣布新措施" from producing the less useful cross-boundary fragment
# "府宣布新措".  Meaningful action words (e.g. "投資") are not boundaries.
_BOUNDARY_WORDS = (
    "由於",
    "因為",
    "因此",
    "不過",
    "但是",
    "然而",
    "雖然",
    "如果",
    "是否",
    "以及",
    "其中",
    "表示",
    "指出",
    "認為",
    "強調",
    "透露",
    "宣布",
    "呼籲",
    "提到",
    "說明",
    "報導",
    "發布",
    "據悉",
    "獲悉",
    "預計",
    "持續",
    "提升",
    "降低",
    "帶動",
    "推動",
    "擴大",
    "新建",
    "啟動",
    "提醒",
    "要求",
    "接近",
    "注意",
    "改善",
    "完成",
    "透過",
    "訓練",
    "使用",
    "採用",
    "維持",
    "公布",
    "推出",
    "實施",
    "執行",
    "引發",
    "造成",
    "導致",
    "影響",
    "獲得",
    "發現",
    "需求",
    "成長",
    "上升",
    "下降",
    "增加",
    "減少",
    "大幅",
    "的",
    "了",
    "在",
    "於",
    "與",
    "和",
    "及",
    "或",
    "而",
    "但",
    "並",
    "且",
    "被",
    "把",
    "對",
    "向",
    "從",
    "由",
    "為",
    "將",
    "已",
)
_BOUNDARY_RE = re.compile("|".join(re.escape(word) for word in _BOUNDARY_WORDS))


# This small vocabulary is an extra signal, not a required dictionary.  It
# improves single-headline extraction where document frequency is unavailable.
# Applications can add company, product, or local-government terms with
# ``domain_terms`` without changing this module.
_DEFAULT_NEWS_TERMS = frozenset(
    {
        "人工智慧",
        "生成式人工智慧",
        "機器學習",
        "深度學習",
        "大型語言模型",
        "自然語言處理",
        "自然語言",
        "資料科學",
        "資料集",
        "文字分類",
        "新聞分類",
        "關鍵字提取",
        "模型訓練",
        "資料安全",
        "網路安全",
        "資安",
        "半導體",
        "半導體產業",
        "晶片",
        "晶片廠",
        "先進製程",
        "先進晶片",
        "供應鏈",
        "高效能運算",
        "伺服器",
        "資料中心",
        "雲端運算",
        "量子運算",
        "電動車",
        "再生能源",
        "氣候變遷",
        "碳排放",
        "碳中和",
        "疫情",
        "疫苗",
        "地震",
        "颱風",
        "選舉",
        "國會",
        "總統",
        "政府",
        "經濟",
        "金融",
        "股市",
        "市場",
        "通貨膨脹",
        "利率",
        "就業",
        "房市",
        "體育",
        "聯賽",
        "球隊",
        "冠軍",
        "比賽",
        "進球",
        "投手",
        "籃球",
        "棒球",
        "足球",
        "奧運",
        "娛樂",
        "演唱會",
        "歌手",
        "電影",
        "影集",
        "門票",
        "粉絲",
        "獎項",
        "社會",
        "生活",
        "警方",
        "消防",
        "交通",
        "豪雨",
        "防災",
        "民眾",
        "戰爭",
        "停火",
        "外交",
        "教育",
        "醫療",
        "健康",
        "台灣",
        "臺灣",
        "中國",
        "美國",
        "日本",
        "歐盟",
        "聯合國",
        "台積電",
        "輝達",
        "鴻海",
        "聯發科",
    }
)


@dataclass(frozen=True)
class Keyword:
    """One extracted keyword.

    Attributes:
        term: The display form found in the input text.
        score: A relative score normalized against the best keyword in the
            same extraction result.  Higher is more salient.
        occurrences: Number of sentences containing the term.  A headline
            counts as one sentence when ``title`` is provided.
    """

    term: str
    score: float
    occurrences: int

    def as_dict(self) -> dict[str, str | float | int]:
        """Return a JSON-serialisable representation."""

        return {
            "term": self.term,
            "score": self.score,
            "occurrences": self.occurrences,
        }


@dataclass
class _CandidateStats:
    """Mutable ranking data kept private to one extraction call."""

    weighted_count: float = 0.0
    occurrences: int = 0
    sentence_ids: set[int] = field(default_factory=set)
    title_hits: int = 0
    phrase_bonus: float = 0.0
    lexicon_hits: int = 0
    lexicon_sentence_ids: set[int] = field(default_factory=set)
    surfaces: Counter[str] = field(default_factory=Counter)


class ChineseKeywordExtractor:
    """Extract ranked Chinese-news keywords without external dependencies.

    Args:
        stopwords: Additional complete terms to suppress.  They are merged
            with a conservative built-in Traditional-Chinese stopword list.
        domain_terms: Organization-, product-, or domain-specific terms to
            favor whenever they occur in the text.  This is useful for a
            newsroom's local named entities and is the main extension point
            for improving extraction quality without a tokenizer.
        min_length: Minimum length for Chinese n-gram candidates.  The default
            of ``2`` avoids treating individual characters as keywords.
        max_ngram_length: Maximum generated Chinese n-gram length.  Longer
            matching domain terms and grammatical chunks may still be kept.
        max_phrase_length: Longest grammatical chunk retained as a whole.

    The extractor has no mutable per-document state, so a single instance can
    be safely reused by a web service after construction.
    """

    def __init__(
        self,
        *,
        stopwords: Iterable[str] | str | None = None,
        domain_terms: Iterable[str] | str | None = None,
        min_length: int = 2,
        max_ngram_length: int = 6,
        max_phrase_length: int = 8,
    ) -> None:
        for name, value in (
            ("min_length", min_length),
            ("max_ngram_length", max_ngram_length),
            ("max_phrase_length", max_phrase_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if min_length < 1:
            raise ValueError("min_length must be at least 1")
        if min_length > _MAX_NGRAM_LENGTH:
            raise ValueError(f"min_length must not exceed {_MAX_NGRAM_LENGTH}")
        if max_ngram_length < min_length:
            raise ValueError("max_ngram_length must be >= min_length")
        if max_ngram_length > _MAX_NGRAM_LENGTH:
            raise ValueError(f"max_ngram_length must not exceed {_MAX_NGRAM_LENGTH}")
        if max_phrase_length < min_length:
            raise ValueError("max_phrase_length must be >= min_length")
        if max_phrase_length > _MAX_PHRASE_LENGTH:
            raise ValueError(f"max_phrase_length must not exceed {_MAX_PHRASE_LENGTH}")

        supplied_stopwords = self._normalise_term_collection(stopwords, "stopwords")
        supplied_terms = self._normalise_term_collection(domain_terms, "domain_terms")

        self.stopwords = frozenset(_DEFAULT_STOPWORDS | supplied_stopwords)
        self.domain_terms = frozenset(_DEFAULT_NEWS_TERMS | supplied_terms)
        self.min_length = min_length
        self.max_ngram_length = max_ngram_length
        self.max_phrase_length = max_phrase_length

    def extract(
        self,
        text: str,
        *,
        top_k: int = 10,
        title: str | None = None,
    ) -> list[Keyword]:
        """Return up to ``top_k`` ranked keywords from one news article.

        Args:
            text: Article body, summary, or any Chinese news text.
            top_k: Maximum number of keywords to return.  ``0`` returns an
                empty list.
            title: Optional headline.  Headline terms are weighted more
                strongly than body-only terms, but are not returned unless
                they are valid keywords.

        Returns:
            A list ordered from most to least salient.  Scores are rounded to
            four decimals and normalized so the first result is ``1.0``.

        Raises:
            TypeError: If text, title, or ``top_k`` has an invalid type.
            ValueError: If ``top_k`` is negative.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string or None")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        sources: list[tuple[str, bool]] = []
        normalized_title = self._normalise(title or "")
        if normalized_title:
            sources.append((normalized_title, True))
        sources.extend((sentence, False) for sentence in self._sentences(text))
        if not sources:
            return []

        candidates: dict[str, _CandidateStats] = {}
        for sentence_id, (sentence, is_title) in enumerate(sources):
            # Sentence-level de-duplication makes a copied phrase in one long
            # sentence less influential than the same topic recurring across
            # the article.
            seen_in_sentence: set[str] = set()
            source_weight = 2.5 if is_title else 1.0

            for cjk_run in _CJK_RE.findall(sentence):
                for chunk in self._chunks(cjk_run):
                    terms = set(self._cjk_ngrams(chunk))
                    if self.min_length <= len(chunk) <= self.max_phrase_length:
                        terms.add(chunk)

                    for term in terms:
                        if not self._is_valid_term(term):
                            continue
                        phrase_bonus = 0.22 if term == chunk else 0.0
                        self._add_candidate(
                            candidates,
                            seen_in_sentence,
                            term,
                            sentence_id=sentence_id,
                            is_title=is_title,
                            weight=source_weight,
                            phrase_bonus=phrase_bonus,
                        )

            for token in _LATIN_OR_NUMBER_RE.findall(sentence):
                token = token.strip("._/-")
                if self._is_valid_term(token):
                    self._add_candidate(
                        candidates,
                        seen_in_sentence,
                        token,
                        sentence_id=sentence_id,
                        is_title=is_title,
                        weight=source_weight,
                        phrase_bonus=0.30 if token.isupper() or any(char.isdigit() for char in token) else 0.0,
                    )

            # Exact matches from the lightweight vocabulary are a useful
            # fallback for a short headline or a long unsegmented CJK run.
            # A lexicon match can add one extra signal per sentence without
            # counting a repeated phrase in the same sentence as a new hit.
            folded_sentence = sentence.casefold()
            for term in self.domain_terms:
                if term.casefold() not in folded_sentence or not self._is_valid_term(term):
                    continue
                self._add_candidate(
                    candidates,
                    seen_in_sentence,
                    term,
                    sentence_id=sentence_id,
                    is_title=is_title,
                    weight=source_weight,
                    phrase_bonus=0.55,
                    lexicon_hit=True,
                )

        if not candidates:
            return []

        ranked = [
            (self._display_term(stats), term, self._raw_score(term, stats), stats)
            for term, stats in candidates.items()
        ]
        ranked.sort(key=lambda item: (-item[2], -len(item[0]), item[0]))
        selected = self._select_diverse(ranked, top_k)
        if not selected:
            return []

        best_score = selected[0][2]
        return [
            Keyword(
                term=display,
                score=round(raw_score / best_score, 4),
                occurrences=stats.occurrences,
            )
            for display, _key, raw_score, stats in selected
        ]

    @staticmethod
    def _normalise(value: str) -> str:
        """Normalize full-width punctuation and remove invisible separators."""

        if not isinstance(value, str):
            return ""
        normalized = unicodedata.normalize("NFKC", value)
        return normalized.replace("\u200b", "").replace("\ufeff", "").strip()

    @staticmethod
    def _normalise_term_collection(
        values: Iterable[str] | str | None, field_name: str
    ) -> set[str]:
        """Normalize caller-supplied vocabulary without splitting a string."""
        if values is None:
            return set()
        supplied_values: Iterable[str] = (values,) if isinstance(values, str) else values
        try:
            iterator = iter(supplied_values)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be an iterable of strings") from exc

        normalized_terms: set[str] = set()
        for value in iterator:
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must contain only strings")
            normalized = ChineseKeywordExtractor._normalise(value)
            if normalized:
                normalized_terms.add(normalized)
        return normalized_terms

    def _sentences(self, text: str) -> list[str]:
        normalized = self._normalise(text)
        return [part.strip() for part in _SENTENCE_BREAK_RE.split(normalized) if part.strip()]

    def _chunks(self, cjk_run: str) -> list[str]:
        """Break a contiguous Chinese run at low-information connectors."""

        return [
            chunk
            for chunk in _BOUNDARY_RE.split(cjk_run)
            if len(chunk) >= self.min_length
        ]

    def _cjk_ngrams(self, chunk: str) -> Iterable[str]:
        upper_length = min(self.max_ngram_length, len(chunk))
        for size in range(self.min_length, upper_length + 1):
            for start in range(0, len(chunk) - size + 1):
                yield chunk[start : start + size]

    def _is_valid_term(self, term: str) -> bool:
        """Reject obvious grammatical fragments and URL-like noise."""

        term = term.strip()
        if not term or term.casefold() in self.stopwords:
            return False
        if term.casefold().startswith(("http", "www")):
            return False
        if term.isdigit():
            return False

        cjk_characters = _CJK_RE.findall(term)
        if cjk_characters:
            # CJK candidates must be all-CJK here.  Mixed forms such as GPT-4
            # are handled by the Latin-token path instead.
            if "".join(cjk_characters) != term:
                return False
            if len(term) < self.min_length:
                return False
            if term[0] in _EDGE_STOP_CHARS or term[-1] in _EDGE_STOP_CHARS:
                return False
        elif len(term) < 2:
            return False

        return True

    @staticmethod
    def _key(term: str) -> str:
        """Use case-insensitive keys while preserving a source display form."""

        return term.casefold()

    def _add_candidate(
        self,
        candidates: dict[str, _CandidateStats],
        seen_in_sentence: set[str],
        term: str,
        *,
        sentence_id: int,
        is_title: bool,
        weight: float,
        phrase_bonus: float = 0.0,
        lexicon_hit: bool = False,
    ) -> None:
        key = self._key(term)
        stats = candidates.setdefault(key, _CandidateStats())
        is_new_in_sentence = key not in seen_in_sentence
        if is_new_in_sentence:
            stats.weighted_count += weight
            stats.occurrences += 1
            stats.sentence_ids.add(sentence_id)
            stats.surfaces[term] += weight
            if is_title:
                stats.title_hits += 1
            seen_in_sentence.add(key)
            stats.phrase_bonus += phrase_bonus
        elif term not in stats.surfaces:
            # Preserve a case/style variant encountered later without changing
            # its frequency contribution.
            stats.surfaces[term] += 0.01

        if lexicon_hit and sentence_id not in stats.lexicon_sentence_ids:
            # Preserve the intended vocabulary boost even when a generated
            # n-gram found the same term first, but only once per sentence.
            if not is_new_in_sentence:
                stats.phrase_bonus += phrase_bonus
            stats.lexicon_hits += 1
            stats.lexicon_sentence_ids.add(sentence_id)


    @staticmethod
    def _display_term(stats: _CandidateStats) -> str:
        """Choose the most strongly supported original spelling."""

        return min(
            stats.surfaces,
            key=lambda surface: (-stats.surfaces[surface], -len(surface), surface),
        )

    @staticmethod
    def _raw_score(term: str, stats: _CandidateStats) -> float:
        """Calculate a relative salience value before final normalization."""

        frequency = 1.0 + math.log1p(stats.weighted_count)
        # Cap the length reward so a long, poorly segmented phrase cannot win
        # only because it contains many characters.
        length = 1.0 + 0.12 * min(max(len(term) - 2, 0), 8)
        coverage = 1.0 + 0.18 * min(max(len(stats.sentence_ids) - 1, 0), 4)
        headline = 1.0 + 0.30 * min(stats.title_hits, 1)
        lexicon = 1.0 + 0.15 * min(stats.lexicon_hits, 3)
        return frequency * length * coverage * headline * lexicon + stats.phrase_bonus

    @staticmethod
    def _shared_cjk_span(left: str, right: str) -> int:
        """Return the length of the longest contiguous CJK span they share."""

        if not _CJK_RE.fullmatch(left) or not _CJK_RE.fullmatch(right):
            return 0
        for size in range(min(len(left), len(right)), 0, -1):
            if any(left[start : start + size] in right for start in range(len(left) - size + 1)):
                return size
        return 0

    @staticmethod
    def _select_diverse(
        ranked: Sequence[tuple[str, str, float, _CandidateStats]],
        top_k: int,
    ) -> list[tuple[str, str, float, _CandidateStats]]:
        """Suppress near-duplicate nested n-grams for a cleaner result list."""

        selected: list[tuple[str, str, float, _CandidateStats]] = []
        for candidate in ranked:
            display, _key, raw_score, stats = candidate
            redundant = False
            for selected_display, _selected_key, selected_score, _selected_stats in selected:
                # A selected full phrase makes a shorter nested n-gram (for
                # example "氣象署" after "中央氣象署") redundant.
                if (
                    display in selected_display
                    and len(selected_display) >= len(display) + 1
                    and selected_score >= raw_score * 0.80
                ):
                    redundant = True
                    break

                # The inverse relation is often a mechanically extended
                # n-gram, such as "人工智慧伺服" after "人工智慧".  Keep a
                # longer candidate only when it is nearly as salient or is an
                # explicit built-in/custom domain term.
                if (
                    selected_display in display
                    and len(display) >= len(selected_display) + 1
                    and stats.lexicon_hits == 0
                    and raw_score <= selected_score
                ):
                    redundant = True
                    break

                # Offset n-grams can overlap a selected phrase without one
                # string literally containing the other ("工智慧伺服" versus
                # "人工智慧").  Suppress substantial CJK overlap unless the
                # candidate was intentionally supplied through the vocabulary.
                shorter_length = min(len(display), len(selected_display))
                if (
                    shorter_length >= 4
                    and stats.lexicon_hits == 0
                    and raw_score <= selected_score
                    and ChineseKeywordExtractor._shared_cjk_span(display, selected_display)
                    >= min(3, shorter_length - 1)
                ):
                    redundant = True
                    break
            if redundant:
                continue
            selected.append(candidate)
            if len(selected) == top_k:
                return selected
        return selected


def extract_keywords(
    text: str,
    top_k: int = 10,
    *,
    top_n: int | None = None,
    title: str | None = None,
    stopwords: Iterable[str] | str | None = None,
    domain_terms: Iterable[str] | str | None = None,
) -> list[tuple[str, float]]:
    """Convenience wrapper around :class:`ChineseKeywordExtractor`.

    Example:
        >>> extract_keywords(
        ...     "台積電擴大美國投資，人工智慧晶片需求持續成長。",
        ...     title="AI需求帶動半導體產業",
        ...     top_k=3,
        ... )
        [('人工智慧', 1.0), ...]

    ``top_n`` is a backwards-compatible alias for ``top_k``.  Construct
    :class:`ChineseKeywordExtractor` directly when the same custom vocabulary
    will be reused for many articles or when ``occurrences`` is also needed.
    """

    if top_n is not None:
        top_k = top_n
    extractor = ChineseKeywordExtractor(stopwords=stopwords, domain_terms=domain_terms)
    return [
        (keyword.term, keyword.score)
        for keyword in extractor.extract(text, top_k=top_k, title=title)
    ]


def extract_keyword_terms(
    text: str,
    top_k: int = 10,
    *,
    top_n: int | None = None,
    title: str | None = None,
    stopwords: Iterable[str] | str | None = None,
    domain_terms: Iterable[str] | str | None = None,
) -> list[str]:
    """Return just keyword strings for callers that do not need scores."""

    return [
        keyword[0]
        for keyword in extract_keywords(
            text,
            top_k,
            top_n=top_n,
            title=title,
            stopwords=stopwords,
            domain_terms=domain_terms,
        )
    ]
