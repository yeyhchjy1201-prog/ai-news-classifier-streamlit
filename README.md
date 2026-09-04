# AI 新聞分類系統 公開部署套件

這個資料夾是從原始專案獨立複製出的公開部署版本。它只包含 Streamlit Community Cloud 執行網站所需的程式、模型與設定範本；不包含虛擬環境、文件、簡報、壓縮檔或任何真正的 API 金鑰。

## 最快的發布方式

1. 在 Windows 直接雙擊 `DEPLOY_TO_STREAMLIT.bat`。
2. 腳本會檢查套件、建立本資料夾自己的 Git 儲存庫，並開啟 GitHub 建立新儲存庫的頁面。
3. 在 GitHub 建立**空白**儲存庫後，將 HTTPS Clone URL 貼回腳本視窗。
4. 腳本推送程式後會開啟 Streamlit Community Cloud。
5. 在 Cloud 頁面選擇剛建立的儲存庫、顯示的分支及 `app.py`，再按 Deploy。
6. 第一版請將 Cloud 的 Secrets 保持空白。網站仍可使用本機新聞分類與關鍵字提取。

首次公開部署仍需要使用者自行登入並授權 GitHub 與 Streamlit Community Cloud；腳本不會也不能代替帳號授權。

執行腳本前，電腦需已安裝 Git for Windows。首次執行時，腳本會詢問 Git 顯示名稱與電子郵件，並只儲存在本資料夾的 Git 設定中，不會改寫電腦的全域 Git 身分。

## 重要安全規則

- 不要建立或提交 `.streamlit/secrets.toml` 到 GitHub。
- 不要將 `OPENAI_API_KEY` 放入公開網站的 Cloud Secrets；否則匿名訪客可能消耗你的付費 API 額度。
- 若 API 金鑰曾出現在聊天室、截圖、電子郵件或公開儲存庫，先在供應商後台撤銷並重建。
- `requirements.txt` 已固定模型測試時的套件版本。修改版本前應重新測試或重新訓練模型。

## 套件內容

- `app.py`：Streamlit 網頁入口。
- `src/`：關鍵字提取與選用文件分析程式。
- `models/news_classifier.joblib`：已訓練的新聞分類模型。
- `.streamlit/config.toml`：非機密的 Streamlit 設定。
- `.streamlit/secrets.toml.example`：只供本機參考，不能填入後上傳。
- `DEPLOY_TO_STREAMLIT.bat`：雙擊入口。
- `DEPLOY_TO_STREAMLIT.ps1`：檢查、Git 建立、推送與開啟雲端部署頁面的核心腳本。

官方說明：

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
