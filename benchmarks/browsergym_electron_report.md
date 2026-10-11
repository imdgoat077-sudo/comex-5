# ServiceNow BrowserGym Benchmark Report — Comex Electron Browser

- **Model**: `ag/gemini-3.8-flash`
- **Runtime Environment**: Live Comex Electron Shell (`WebContentsView` via CDP Port 9222)
- **Overall Accuracy**: **13/15 (86.7%)**
- **Average Duration**: **12.00s**
- **Evaluation Mode**: Strict Zero-Cheating Multi-Turn Agent Loop with BrowserGym BIDs

## Category Performance

| Category | Tasks | Passed | Accuracy |
|---|---|---|---|
| Button Click | 2 | 2 | 100.0% |
| Checkbox Selection | 1 | 1 | 100.0% |
| Complex List Selection | 1 | 0 | 0.0% |
| Complex Multi-Field Form | 1 | 0 | 0.0% |
| Dropdown Select | 1 | 1 | 100.0% |
| Form Submission | 1 | 1 | 100.0% |
| Hierarchy & Toggle | 1 | 1 | 100.0% |
| Input Focus | 1 | 1 | 100.0% |
| Modal Dialog | 1 | 1 | 100.0% |
| Multi-Step Click | 1 | 1 | 100.0% |
| Radio / Option | 1 | 1 | 100.0% |
| Search & Navigate | 1 | 1 | 100.0% |
| Text Input | 2 | 2 | 100.0% |

## Task Execution Details

| ID | Task Name | Category | Result | Reward | Latency | Turns | Actions Taken |
|---|---|---|---|---|---|---|---|
| T01 | `click-test` | Button Click | **PASS** | 1.00 | 5.33s | 1 | `click(bid=1)` |
| T02 | `click-button` | Button Click | **PASS** | 1.00 | 7.94s | 1 | `click(bid=5)` |
| T03 | `click-button-sequence` | Multi-Step Click | **PASS** | 1.00 | 6.47s | 1 | `click(bid=1), click(bid=2)` |
| T04 | `enter-text` | Text Input | **PASS** | 1.00 | 14.73s | 1 | `fill(bid=1), click(bid=2)` |
| T05 | `enter-text-dynamic` | Text Input | **PASS** | 1.00 | 7.57s | 1 | `fill(bid=1), click(bid=2)` |
| T06 | `focus-text` | Input Focus | **PASS** | 1.00 | 6.55s | 1 | `click(bid=1)` |
| T07 | `click-checkboxes` | Checkbox Selection | **PASS** | 1.00 | 8.46s | 1 | `click(bid=2), click(bid=4), click(bid=6), click(bid=11)` |
| T08 | `choose-list` | Dropdown Select | **PASS** | 1.00 | 7.11s | 1 | `select_option(bid=1), click(bid=2)` |
| T09 | `click-dialog` | Modal Dialog | **PASS** | 1.00 | 6.35s | 1 | `click(bid=1)` |
| T10 | `login-user` | Form Submission | **PASS** | 1.00 | 7.39s | 1 | `fill(bid=2), fill(bid=4), click(bid=5)` |
| T11 | `search-engine` | Search & Navigate | **PASS** | 1.00 | 14.2s | 2 | `fill(bid=1), click(bid=2), click(bid=4)` |
| T12 | `click-collapsible` | Hierarchy & Toggle | **PASS** | 1.00 | 7.17s | 1 | `click(bid=1), click(bid=2)` |
| T13 | `click-option` | Radio / Option | **PASS** | 1.00 | 7.21s | 1 | `click(bid=2), click(bid=9)` |
| T14 | `email-inbox` | Complex List Selection | **FAIL** | 0.00 | 40.33s | 4 | `click(bid=13), click(bid=2), click(bid=2), click(bid=1)` |
| T15 | `book-flight` | Complex Multi-Field Form | **FAIL** | 0.00 | 33.25s | 4 | `fill(bid=1), fill(bid=2), fill(bid=3), click(bid=4), click(bid=4), click(bid=12), click(bid=4), fill(bid=3), click(bid=4)` |
