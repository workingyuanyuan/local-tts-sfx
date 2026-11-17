# local-tts：Agent 專案規則

本機 TTS 服務，讓 Claude Code、Codex 等 coding agent 在任務需要旁白或音效時，自行用本機 GPU 生成。語音模型是 MOSS-TTS-Local-Transformer-v1.5，音效模型是 MOSS-SoundEffect v2.0。安裝與使用方式見 [README.md](README.md)；這份文件只寫修改本 repo 時必須遵守的規則。

## 架構

```
Agent ──MCP stdio──▶ tts-mcp ─┐
      ──shell──────▶ tts CLI ─┤  HTTP（回傳音訊 bytes 與中繼資料）
                              ▼
                    tts-server（常駐 FastAPI，systemd user service）
                      └─ 單一 GPU worker ← 有界佇列
                           └─ config/models.yaml 模型註冊表 → adapter（同一時間只載入一個模型）
                                ├─ kind: tts → 語音 adapter（同一程序內）
                                └─ kind: sfx → 音效 adapter ──stdin/stdout 框架協定──▶ worker 子程序
                                                                  （workers/moss-sfx/.venv，獨立環境）
```

| 位置 | 內容 |
|---|---|
| `src/local_tts/server/` | FastAPI 端點（`app.py`）、排程與工作狀態（`scheduler.py`） |
| `src/local_tts/engine.py`、`text.py`、`audio.py` | 切段、時長估計、重試與截斷重切、拼接與編碼 |
| `src/local_tts/adapters/` | 模型 adapter：語音 `moss_local.py`、音效 `moss_sfx.py`（介面在 `base.py`：共用生命週期的 `ModelAdapter`，語音 `Adapter`、音效 `SoundEffectAdapter`）；worker 子程序的傳輸層 `sfx_worker.py`；新模型就加新的 adapter |
| `src/local_tts/sfx.py` | 音效工作：每段生成一次、片段失敗隔離、片段之間檢查取消 |
| `workers/moss-sfx/` | 音效模型的獨立環境（`pyproject.toml`、`uv.lock`）與 worker 腳本 `worker.py`（不 import `local_tts`） |
| `src/local_tts/client.py` | CLI 與 MCP 共用的 HTTP client：寫檔、路徑、期限、重試 |
| `src/local_tts/cli.py`、`mcp_server.py` | `tts` CLI 與 `tts-mcp` |
| `config/` | 伺服器設定、模型與聲音註冊表 |
| `agent-rules/` | 給使用者的 Agent 的使用規則原稿（skill：旁白 `narration`、音效 `sound-effects`），由 `scripts/install_agent_rules.sh` 安裝到 Claude Code 與 Codex 的 skills 目錄；不寫入 `~/.codex/AGENTS.md`（那會載入每個專案的上下文） |
| `tests/` | 單元測試（假 adapter，不需 GPU） |
| `scripts/` | 實測與驗收腳本（需要執行中的服務與 GPU） |
| `docs/` | 開發各步的實測報告與審查紀錄 |

## 不可破壞的契約

**排程**
- 只有一個服務程序、一個 GPU worker。請求進有界佇列；佇列滿時回 `queue_full`，未取走的音訊超過 `max_result_bytes` 時回 `results_full`，不無限堆積。
- 模型的載入與卸載受互斥鎖保護，同一模型不會重複載入。切換模型前先卸載目前的模型。
- 閒置卸載與手動卸載（`/v1/models/unload`）都只在沒有執行中工作、且佇列為空時進行，並持有排程器的鎖（worker 在模型管理器的鎖之外使用 adapter）。
- 程序拓撲：一個服務程序、同一時間一個 GPU 執行者，是同程序內的語音 adapter，或音效 worker 子程序，兩者不同時載入。音效模型的 `load()` 啟動子程序並等到模型載入完成；`unload()` 結束並回收（reap）子程序，這才算釋放顯存。卸載失敗（子程序無法回收）時該模型仍是目前模型，在它釋放前不載入其他模型。
- worker 死掉、逾時或違反協定時：終止並回收它，該工作以明確錯誤失敗，模型管理器之後仍可再載入。所有對 worker 的讀寫都有期限與大小上限；協定通道不是函式庫看到的 stdout（worker 把原本的 fd 1 另存為協定通道，fd 1 與 `sys.stdout` 指向 stderr），stderr 由執行緒持續讀出，不會塞滿管線。
- 快取維護（`release_cache`）的任何錯誤都不能讓 GPU worker 執行緒結束。

**模型與聲音**
- 未指定 `model` 時用 `models.yaml` 的 `default`。
- 未指定 `voice` 時不使用任何已註冊的聲音。為了讓同一次呼叫的音色一致，以第一個至少 3 秒的合格片段作為後續片段的參考；在它之前較短的片段，會用這個參考重新生成。
- 指定 `voice` 時，每一段都用同一個參考音檔。
- 多片段工作（`clips`）視為一次呼叫：未指定 `voice` 時所有片段共用同一個參考；某段最終失敗只讓該片段失敗，其他錯誤讓整個工作失敗。片段 id 會成為檔名，伺服器與用戶端都要驗證。
- 使用者註冊的聲音放在 repo 外（`config.user_voices_dir()`），不寫入 `config/voices.yaml`；兩個聲音檔修改後自動重新讀取，讀取失敗時保留上一份有效的聲音並回報。服務不開啟呼叫端的檔案，註冊時由 tts-mcp／CLI 讀檔上傳。
- 未知的模型或聲音要回錯，並列出可用項目。
- 模型有 `kind`（`tts` 預設、`sfx`）。語音端點收到音效模型、音效端點收到語音模型，回 `wrong_model_kind` 並列出正確種類的可用模型；未指定時分別用 `default`、`default_sfx`。MCP 用某一類的取回工具取另一類的工作時回 `wrong_job_kind` 並指出正確工具，不寫檔；`pending` 的 `next.tool` 一定是該類工作的工具。
- 不支援的參數一律明確報錯，絕不靜默忽略（HTTP 與 MCP 都一樣）。
- HTTP 的 OpenAI 相容範圍只有 `/v1/audio/speech` 的 `model`、`input`、`voice`、`response_format`（wav、flac、mp3）；其他擴充欄位要寫進文件。

**長文**
- 依句界切段，`[pause …]` 等標記不可拆開。
- 截斷（達到 frame 上限）時切細重試；重試次數有上限，最終失敗要回報是哪一段。
- 時長門檻是和專案擁有者約定的（`engine.py`）：
  - 漏唸：輸出為空，或估計超過 2 秒、且生成時長低於「估計 × 0.3 − 0.5 秒」。重試，最終報錯。
  - 疑似漏句：估計超過 2 秒、且低於「估計 × 0.7」（沒有寬限）。重試；仍偏短時保留最長的一次，並在 `warnings` 註明。
  - 持續發聲：高於「估計 × 2.2 + 3 秒」，代表模型唸完後還在發聲。重試，最終報錯。
- 句中少幾個字的漏唸是模型問題，**不要**在服務端加 ASR 或其他防護。

**音效**
- 只有非同步（`POST /v1/sfx/jobs`），沒有同步端點；查詢、取消、取回、`Idempotency-Key`（依種類分開比對）、佇列、`max_result_bytes`、保留時間與部分取回都和語音工作共用。
- `seconds` 必填（沒有靜默的預設值），有限數值，介於模型的最小與最大秒數（30）之間；提示長度、片段數、每個工作的總秒數上限在 `config/server.yaml` 的 `sfx:`。
- 音效不做任何語音專用的處理：不切句、不依時長重試、不去除靜音、沒有聲音參考與 `[pause]`。每段生成一次；worker 回報的單段錯誤只讓該段失敗，worker 本身失敗讓整個工作失敗。
- 取消：排隊中立即取消；執行中在片段之間停止，正在生成的那段完成後捨棄（即使那段失敗，取消也優先）。
- 等待預估：音效以「每段秒數」計（與長度無關），與語音的 RTF 分開記錄；未載入時加上載入（含編譯暖機）時間。不可對音效工作回報語音式的估計。
- 沒有 worker 環境、沒有權重或沒有 GPU 時給明確錯誤（附安裝或下載指令），不能讓服務崩潰。沒有 C 編譯器時改為不編譯執行並在 `/health`、`/v1/models` 與結果的 `warnings` 註明，不是錯誤。

**檔案與工作**
- 服務只回傳音訊，不寫入呼叫端的檔案；由 tts-mcp 或 CLI 寫到呼叫端指定的**絕對路徑**。
- 目標已存在時預設報錯，`overwrite=true` 才覆寫。未指定路徑時寫到 `~/.local/share/local-tts/output/`。Windows 路徑只在 WSL 下用 `wslpath` 轉換。
- 工作建立後，每個錯誤都要帶 `job_id` 與接續方式，永遠不要自動重送相同的文字。
- 送出一律帶 `Idempotency-Key`。
- 一次呼叫的等待請求（送出、輪詢、健康查詢）共用同一個期限，期限後最多 0.5 秒的寬限。完成音訊的下載（最多 30 秒）與本機寫檔不在此限。

## 模型與相依套件

- **不要改 `n_vq_for_inference`**。
- 注意力用 SDPA，照官方設定關閉 cuDNN SDPA（`enable_cudnn_sdp(False)`），flash、mem_efficient、math 保持開啟。不裝 flash-attn。
- `codec_weight_dtype` 必須明確設定：HF 上的程式碼預設是 bf16，GitHub 上的是 fp32。
- 模型與 tokenizer 的 HF revision 固定在 `config/models.yaml`。權重放在 HF 預設快取（`~/.cache/huggingface/`），不放進 repo。
- MOSS-TTS 是固定 commit 的 submodule（`third_party/MOSS-TTS`）。套件由 `uv.lock` 鎖定：改相依要用 `uv lock`，安裝用 `uv sync --frozen`。
- 官方取樣參數寫在 `models.yaml`，沒有實測依據不要改。
- 音效模型的環境獨立鎖定：`workers/moss-sfx/pyproject.toml` 與 `uv.lock`（以路徑 editable 安裝 submodule 中的 `moss_soundeffect_v2` 與其 `torch-cu128` extra，torch 來自 cu128 index）。改相依用 `uv lock --project workers/moss-sfx`，安裝用 `uv sync --frozen --project workers/moss-sfx`。`worker.py` 不 import `local_tts`，服務程序也不 import `moss_soundeffect_v2`。不要用 `torchaudio.save`／`pipe.save_audio`（本機 torchcodec 載入失敗）：worker 回傳 float32 樣本，由服務的 `audio.py` 編碼。音效權重的 revision 同樣固定在 `config/models.yaml`。
- torch.compile 依 `models.yaml` 的 `compile`（預設開啟），各機器可用 `LOCAL_TTS_SFX_COMPILE` 覆寫；關閉時 worker 環境設 `TORCHDYNAMO_DISABLE=1`。編譯快取放在每個使用者的持久目錄 `<快取目錄>/torch-compile`（快取目錄為 `LOCAL_TTS_CACHE_DIR`，未設時為 `$XDG_CACHE_HOME/local-tts`，預設 `~/.cache/local-tts`）。
- 批次生成（`batch_size`）在 repo 預設為 1（關閉），由各機器以 `LOCAL_TTS_BATCH_SIZE` 開啟；依據與限制見 `docs/batch-bench-report.md`。批次只用於參考音色確定後各段的第一次生成，重試與切分一律逐段；批次顯存不足時退回逐段，不報錯。

## 可攜性

這個 repo 要能在別台機器上使用，不只是開發者的筆電。
- 程式、設定、腳本、規則與文件中不寫死任何本機路徑、使用者名稱或硬體。路徑一律由檔案自身位置推導、用 `$PWD`，或由安裝腳本代入 `{{LOCAL_TTS_ROOT}}`（見 `agent-rules/` 與 `deploy/local-tts.service`）。
- 支援的環境：Linux x86_64（原生或 WSL2），加上 NVIDIA GPU 與支援 CUDA 12.8 的驅動。只在特定環境才有的功能，在其他環境要給明確的錯誤，不能直接崩潰：例如 WSL 才有的 `wslpath`、systemd，或沒有 GPU 時。
- 不要假設 GPU 的型號或顯存大小；需要的資源寫在 README 的需求段落。

## 開發與測試

- 用 uv 管理的 Python 3.12 venv（`.venv/`），不要動系統 Python，執行時用 `.venv/bin/python`。
- 單元測試：`.venv/bin/python -m pytest -q tests`（約 45 秒，不需 GPU 也不需音效 worker 環境：音效用假 adapter `tests/fakes.py`，傳輸層用服務自己的 Python 執行的假 worker `tests/fake_sfx_worker.py`；Agent 規則安裝腳本以暫時的 HOME 測試）。修 bug 時要補回歸測試。計時相關的斷言要計算請求次數，或留足寬鬆的上限，否則會因為機器忙碌而不穩定。
- `scripts/step*_run.sh` 是使用 GPU 的實測，產物寫在 `outputs/`（不進 repo）：
  - 第 1 步：自己載入模型，必須先停止服務或確認它已卸載，以免兩份模型搶顯存。
  - 第 2 步：會自行停止並重新啟動 systemd 服務。
  - 第 3、4 步：需要服務正在執行。
  - 音效：`scripts/sfx_run.sh` 從這個 checkout 在非預設埠啟動服務並執行 `scripts/sfx_smoke.py`（埠已被佔用時拒絕）；`scripts/sfx_mcp_stdio.py` 經 stdio 驅動 tts-mcp 的音效工具。先卸載常駐服務的模型。
- 改了 `config/*.yaml`、伺服器程式或 `workers/moss-sfx/worker.py` 後，要重啟服務（`systemctl --user restart local-tts`）才會生效。改了 `agent-rules/` 後要重新執行 `scripts/install_agent_rules.sh`。
- 結束程序要用確切的 PID，**不要用 `pkill -f`**：它可能比對到自己的 shell。

## 文件

- README 用繁體中文撰寫，內容要和程式一致：CLI 參數、MCP 工具、HTTP 端點、錯誤碼與設定。
- `docs/` 是各步當時的實測與審查紀錄，不改寫其中的結果；新的實測或審查另寫一份新的報告。
