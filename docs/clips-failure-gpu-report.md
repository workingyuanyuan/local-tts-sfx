# 多片段工作：片段失敗與接續的 GPU 實測

日期：2026-11-17。補 `docs/clips-accept-report.md`「未涵蓋」中的片段失敗、`partial` 與接續（之前只有單元測試）。

## 方法

- 在另一個埠（8777）另開測試服務，使用 `LOCAL_TTS_BATCH_SIZE=4`。正式服務先卸載模型。
- 測試用的設定是 `config/` 的副本，改了兩個值，讓長片段在真實生成中必定截斷且不切分：
  - `max_new_tokens: 75`（6 秒）
  - `max_split_depth: 0`
- 片段（中文）：
  - `a` 短（約 1.4 秒）
  - `b` 一長句不含標點（估計遠超 6 秒，預期失敗）
  - `c`、`d` 約 3 秒
  - `e` 短
- `b` 排在第二個，也就是沒有 `voice` 時第一個參考候選，用來確認它失敗後參考會改用下一個片段。
- 產物放在 session 暫存目錄，不進 repo。

## 結果

### 伺服器（沒有 `voice`，`tts clips`）

- 工作狀態是 `succeeded`，`b` 為 `segment_truncated`（`stage: synthesis`），其餘 4 個片段完成。
- 參考音色：
  - 日誌順序：`a` 生成，`b` 截斷失敗，`c` 成為參考，`d`、`e` 以一個批次生成，`a` 用參考重新生成。
  - `b` 失敗沒有影響其他片段，也沒有讓參考落空。
- 片段時長：`a` 1.44 秒、`c` 3.44 秒、`d` 3.28 秒、`e` 1.12 秒。沒有重試，也沒有 `warnings`。
- 所有片段都失敗（`b` 單獨一個，或 `b` 加 `b2`）時，整個工作失敗：
  - 錯誤附 `job_status: "failed"`、每個片段的錯誤和 `recovery`，沒有 `fetch_command`。
  - MCP 與 CLI 都一樣（CLI exit 1）。

### CLI 接續（`tts job <id> --wait -d`）

| 情況 | 結果 |
|---|---|
| 檔案都在，再取一次 | `partial`，exit 4；sha256 相同，沒有重寫任何檔案（mtime 不變） |
| 刪掉 `c.wav` 再取 | 只補回 `c`，sha256 與伺服器相同 |
| `d.wav` 被改過 | `d` 為 `output_exists`（`stage: delivery`），exit 4 |
| 加 `--overwrite` | `d` 重寫，sha256 與伺服器相同 |

### MCP（`tts-mcp` 經 stdio，`voice: zh_male_student`）

- `generate_speech_clips` 的 `wait_seconds: 1` 回 `pending`（附 `progress`、`clips_total`、`next.tool: get_speech_clips`）。依 `next.arguments` 呼叫一次就回 `partial`，`recovery` 指示只把 `b` 重新送進 `generate_speech_clips`。
- 再呼叫一次，結果相同，也沒有重寫檔案。
- 把 `c.wav` 改掉後，`recovery` 另外列出 `c`，並給出帶 `overwrite: true` 的 `get_speech_clips` 參數。

## 發現並修正

CLI 的 `partial` 結果原本一律附 `fetch_command`，沒有 `recovery`：

- 只有合成失敗時，照 `fetch_command` 重取永遠得到同樣的 `partial`。Agent 可能因此一直重取，而不是重新送出失敗的片段。
- 目標已存在（`output_exists`）時，`fetch_command` 沒有帶 `--overwrite`。

修正（`cli._clips_recovery`，與 MCP 的同名函式對應）：

- `partial` 一律附 `recovery`。
- 只有未寫入的片段才附 `fetch_command`，有 `output_exists` 時帶 `--overwrite`。
- 文字模式也印出 `recovery` 與指令。

在同一個 GPU 工作上重跑，兩種情況的輸出都正確。回歸測試在 `tests/test_clips.py::test_cli_clips`，README 與 agent-rules 已同步。

## 未涵蓋

- `segment_failed`（重試後仍過短或持續發聲）沒有在 GPU 上觸發。它和 `segment_truncated` 同屬 `CLIP_ERRORS`，走相同的片段失敗路徑。
- 下載中斷造成的 `delivery` 失敗只有單元測試。
- 英文與混合長短片段的效率仍未實測。
