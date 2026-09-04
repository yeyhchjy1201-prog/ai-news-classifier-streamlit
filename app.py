"""Streamlit 網頁介面：新聞分類、關鍵字提取與選用的文件分析。

用途：這是專案的瀏覽器入口。使用者可貼上新聞，取得本機類別預測、
模型信心分數與關鍵字；另可選擇上傳文件並以 OpenAI 進行分析。

使用方式：優先雙擊 ``start_app.bat``；手動啟動時使用
``.\\.venv\\Scripts\\python.exe -m streamlit run app.py``，不要直接執行
``python app.py``。

名詞：Streamlit 是 Python 網頁介面框架；``joblib`` 是本機模型保存格式；
``Pipeline`` 將文字特徵與分類器串成一個可重複使用的流程；信心分數是模型
對最高類別的機率，並非新聞真實性保證。
"""

from __future__ import annotations

import importlib
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.openai_document import (
    DEFAULT_MODEL,
    MAX_FILE_SIZE_BYTES,
    MAX_INSTRUCTIONS_CHARS,
    OpenAIDocumentError,
    analyze_document,
)

try:
    import joblib
except ImportError:  # 讓尚未安裝依賴時也能顯示可理解的訊息
    joblib = None


# 路徑與輸入限制集中在此處，讓介面與訓練輸出使用同一份專案結構。
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "news_classifier.joblib"
MAX_ARTICLE_CHARS = 20_000
LOW_CONFIDENCE_THRESHOLD = 0.45


def _model_cache_token(path: Path) -> int:
    """Return a token that changes whenever the on-disk model is replaced."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, model_mtime_ns: int) -> tuple[Any | None, str | None]:
    """載入訓練完成的 joblib 模型，並將錯誤轉為介面訊息。"""
    # Include the file timestamp in Streamlit's cache key so a completed
    # retraining run at the same path takes effect without restarting the app.
    _ = model_mtime_ns
    if joblib is None:
        return None, "找不到 joblib。請先安裝 requirements.txt 中的套件。"

    path = Path(model_path)
    if not path.exists():
        return (
            None,
            "尚未找到訓練好的模型。請先完成訓練，讓模型輸出到 "
            "models/news_classifier.joblib。",
        )

    try:
        return joblib.load(path), None
    except Exception:  # noqa: BLE001 - 模型檔可能來自不同版本的 sklearn
        return None, "模型無法載入。請確認它來自可信來源，且與目前套件版本相容。"


def _get_estimator(artifact: Any) -> Any:
    """支援直接儲存 Pipeline，也支援以 dict 包裝的模型 artifact。"""
    if isinstance(artifact, Mapping):
        for key in ("pipeline", "model", "classifier", "estimator"):
            candidate = artifact.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
    return artifact


def _display_label(label: Any, artifact: Any) -> str:
    """將編碼後的類別盡可能轉回容易閱讀的名稱。"""
    if not isinstance(artifact, Mapping):
        return str(label)

    label_names = artifact.get("label_names") or artifact.get("labels")
    if isinstance(label_names, Mapping):
        return str(label_names.get(label, label_names.get(str(label), label)))
    if isinstance(label_names, Sequence) and not isinstance(label_names, str):
        try:
            return str(label_names[int(label)])
        except (IndexError, TypeError, ValueError):
            pass

    encoder = artifact.get("label_encoder")
    if encoder is not None and hasattr(encoder, "inverse_transform"):
        try:
            return str(encoder.inverse_transform([label])[0])
        except Exception:  # noqa: BLE001
            pass
    return str(label)


def predict_article(artifact: Any, article: str) -> tuple[str, float | None]:
    """以 sklearn Pipeline / 分類器預測新聞分類與信心分數。"""
    estimator = _get_estimator(artifact)
    if not hasattr(estimator, "predict"):
        raise TypeError("模型不含 predict()，請確認 news_classifier.joblib 的內容。")

    try:
        raw_label = estimator.predict([article])[0]
    except Exception as first_error:  # 部分自訂模型只接受單一字串
        try:
            raw_label = estimator.predict(article)[0]
        except Exception:  # noqa: BLE001
            raise first_error

    confidence: float | None = None
    if hasattr(estimator, "predict_proba"):
        try:
            probabilities = estimator.predict_proba([article])[0]
            confidence = float(max(probabilities))
        except Exception:  # noqa: BLE001 - 並非每個分類器都提供可用機率
            pass

    return _display_label(raw_label, artifact), confidence


@st.cache_resource(show_spinner=False)
def _find_keyword_function() -> Callable[..., Any] | None:
    """優先使用專案中的 keyword_extractor.extract_keywords。"""
    for module_name in ("src.keyword_extractor", "keyword_extractor"):
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, "extract_keywords", None)
            if callable(function):
                return function
        except Exception:  # noqa: BLE001 - 模組尚未建立或選用套件未安裝時改用備援
            continue
    return None


def _normalise_keywords(result: Any, top_n: int) -> list[str]:
    """接受常見的關鍵字回傳格式：字串、list 或 (keyword, score) tuple。"""
    if isinstance(result, Mapping):
        items = list(result.keys())
    elif isinstance(result, str):
        items = re.split(r"[,，、\n]", result)
    else:
        try:
            items = list(result)
        except TypeError:
            items = [result]

    keywords: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, Mapping):
            # src.keyword_extractor 會回傳 {term, score, occurrences}。
            value = item.get("term") or item.get("keyword") or item.get("text") or ""
        else:
            value = item[0] if isinstance(item, (tuple, list)) and item else item
        word = str(value).strip()
        key = word.casefold()
        if word and key not in seen:
            keywords.append(word)
            seen.add(key)
        if len(keywords) >= top_n:
            break
    return keywords


_STOP_WORDS = {
    "這個", "一個", "我們", "你們", "他們", "以及", "但是", "因為", "所以", "相關",
    "表示", "指出", "今日", "目前", "進行", "將會", "可以", "新聞", "報導", "記者",
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "will", "has",
}


def _fallback_keywords(article: str, top_n: int) -> list[str]:
    """沒有 keyword_extractor.py 時的輕量備援，無須額外下載語言模型。"""
    candidates: list[str] = []
    lowercase = article.lower()
    candidates.extend(re.findall(r"[a-z][a-z0-9_-]{1,}", lowercase))

    # 對中文連續字串取 2 至 4 字片段；正式流程仍建議使用 keyword_extractor.py。
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", article):
        if len(phrase) <= 8:
            candidates.append(phrase)
        for size in (2, 3, 4):
            candidates.extend(phrase[index : index + size] for index in range(len(phrase) - size + 1))

    counts = Counter(
        item for item in candidates if item.casefold() not in _STOP_WORDS and len(item.strip()) >= 2
    )
    return [word for word, _ in counts.most_common(top_n)]


def extract_article_keywords(article: str, top_n: int) -> tuple[list[str], str]:
    """從專案模組取關鍵字；若模組不可用則採內建備援。"""
    extractor = _find_keyword_function()
    if extractor is not None:
        call_patterns = (
            lambda: extractor(article, top_n=top_n),
            lambda: extractor(article, top_k=top_n),
            lambda: extractor(article, top_n),
        )
        for call in call_patterns:
            try:
                keywords = _normalise_keywords(call(), top_n)
                if keywords:
                    return keywords, "專案關鍵字提取器"
            except TypeError:
                continue
            except Exception:  # noqa: BLE001
                break
    return _fallback_keywords(article, top_n), "內建備援提取器"


def _read_openai_setting(name: str, default: str = "") -> str:
    """讀取設定值但不記錄內容；環境變數優先於 Streamlit secrets。"""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        secret_value = st.secrets.get(name, default)
    except Exception:  # Streamlit secrets are optional for local use.
        secret_value = default
    return str(secret_value).strip() if secret_value else default


def render_openai_document_analysis() -> None:
    """Render the optional OpenAI-powered document analysis workflow."""
    st.divider()
    st.subheader("使用 OpenAI 分析文件")
    st.caption(
        "上傳的文件會傳送到你的 OpenAI API 專案進行分析。"
        "系統不會把 API 金鑰寫入程式碼。"
    )

    api_key = _read_openai_setting("OPENAI_API_KEY")
    model = _read_openai_setting("OPENAI_MODEL", DEFAULT_MODEL)
    if api_key:
        st.success(f"OpenAI 已設定，使用模型：{model}")
    else:
        st.warning("尚未設定 OPENAI_API_KEY，因此無法分析文件。")
        with st.expander("如何設定 API 金鑰"):
            st.code(
                '$env:OPENAI_API_KEY = "你的 OpenAI API Key"\n'
                f'$env:OPENAI_MODEL = "{DEFAULT_MODEL}"  # 可選\n'
                "streamlit run app.py",
                language="powershell",
            )
            st.caption("也可依 .streamlit/secrets.toml.example 建立 secrets.toml。")

    uploaded_file = st.file_uploader(
        "上傳文件",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx"],
        help="支援 PDF、Word、文字、CSV、Excel 與 PowerPoint；單檔上限 20 MB。",
        key="openai_document_upload",
    )
    instructions = st.text_area(
        "希望如何處理這份文件？",
        value=(
            "請以繁體中文分析此文件，依序提供：\n"
            "1. 3 至 5 點重點摘要\n"
            "2. 主要主題與（若適用）新聞分類\n"
            "3. 5 至 8 個關鍵字\n"
            "4. 需要進一步查核的資訊"
        ),
        height=170,
        max_chars=MAX_INSTRUCTIONS_CHARS,
        key="openai_document_instructions",
    )

    file_too_large = uploaded_file is not None and uploaded_file.size > MAX_FILE_SIZE_BYTES
    can_analyze = uploaded_file is not None and not file_too_large and bool(api_key)
    if st.button(
        "使用 OpenAI 分析文件",
        type="primary",
        width="stretch",
        disabled=not can_analyze,
        key="openai_document_analyze",
    ):
        assert uploaded_file is not None  # Enabled only when a file is present.
        with st.spinner("OpenAI 正在閱讀並分析文件……"):
            try:
                result = analyze_document(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type,
                    instructions=instructions,
                    api_key=api_key,
                    model=model,
                )
            except OpenAIDocumentError as exc:
                st.error(str(exc))
                return

        st.subheader("文件分析結果")
        st.markdown(result)
        st.caption(
            f"已分析：{uploaded_file.name}（{uploaded_file.size / 1024 / 1024:.2f} MB）。"
        )
    elif file_too_large:
        st.error("文件大小不可超過 20 MB。")


def main() -> None:
    st.set_page_config(page_title="AI 新聞分類與文件分析", page_icon="📰", layout="centered")
    st.title("📰 AI 新聞分類與文件分析")
    st.caption("貼上新聞內容，或上傳文件交由 OpenAI 進行摘要、分類與關鍵字提取。")

    with st.sidebar:
        st.header("系統狀態")
        artifact, load_error = load_model(str(MODEL_PATH), _model_cache_token(MODEL_PATH))
        if load_error:
            st.warning("模型尚未就緒")
            st.caption(load_error)
        else:
            st.success("模型已載入")
            st.caption(f"模型位置：{MODEL_PATH.relative_to(PROJECT_DIR)}")

        st.divider()
        st.caption("模型檔應位於 models/news_classifier.joblib")
        st.caption("關鍵字模組：src/keyword_extractor.py（可選）")
        st.caption("OpenAI 文件分析：需設定 OPENAI_API_KEY")

    article = st.text_area(
        "新聞內容",
        height=240,
        max_chars=MAX_ARTICLE_CHARS,
        placeholder="例如：輸入一則科技、財經、體育、娛樂或社會生活新聞的標題與內文……",
        help="建議至少輸入一句完整新聞內容；最多 20,000 字。分類結果僅供輔助判讀。",
    )
    top_n = st.slider("顯示幾個關鍵字", min_value=3, max_value=10, value=5)

    if st.button("開始分析", type="primary", width="stretch"):
        clean_article = article.strip()
        if len(clean_article) < 10:
            st.info("請先輸入至少 10 個字的新聞內容，再開始分析。")
            return

        if load_error or artifact is None:
            st.error("目前無法進行分類，因為模型尚未完成訓練或無法載入。")
            st.info("請先依 README 的訓練步驟產生 models/news_classifier.joblib，然後重新整理此頁面。")
            return

        with st.spinner("正在分析新聞……"):
            try:
                label, confidence = predict_article(artifact, clean_article)
                keywords, keyword_source = extract_article_keywords(clean_article, top_n)
            except Exception:  # noqa: BLE001
                st.error("本機分析暫時無法完成。請確認模型與目前資料前處理流程相容。")
                st.caption("如問題持續，請使用目前環境重新訓練可信來源的模型。")
                return

        result_column, confidence_column = st.columns(2)
        result_column.metric("新聞分類", label)
        if confidence is not None:
            confidence_column.metric("模型信心", f"{confidence:.1%}")
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning("此結果的模型信心偏低，建議改用完整內文或由人工確認。")
        else:
            confidence_column.metric("模型信心", "未提供")

        st.subheader("關鍵字")
        if keywords:
            st.markdown(" ".join(f"`{word}`" for word in keywords))
        else:
            st.info("找不到足夠的關鍵字，請嘗試輸入較完整的新聞內容。")
        st.caption(f"來源：{keyword_source}")

    render_openai_document_analysis()

main()
