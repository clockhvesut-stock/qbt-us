# qbt-us — 米国株の自動売買システム

パソコンの電源が入っていなくても、毎朝LINEにレポートが届きます。
すべてクラウド（GitHub Actions）上で動くためです。

---

## 全体の流れ

```
  日本時間 07:00   GitHub Actions がデータを集める
                   ├ 価格（S&P500 + 中型株 + ETF）
                   ├ マクロ指標（VIX・金利・ドル・原油・金）
                   ├ ニュース見出し
                   └ SEC提出書類（8-K, 10-Q, 大量保有報告 …）
                          ↓  reports/YYYY-MM-DD_raw.json としてコミット

  日本時間 07:30   Claude がそれを読んで解釈する
                   ├ ニュースが効くのか効かないのかを判断
                   ├ ルールベースの候補と突き合わせる
                   └ 保有銘柄の手仕舞い判断
                          ↓  outbox/YYYY-MM-DD_message.json としてコミット

  日本時間 08:15   GitHub Actions が LINE に送信
```

米国市場の大引けは日本時間の早朝5〜6時です。
朝にレポートを読み、その日の夜（22:30）の寄付で執行する、というサイクルになります。
考える時間が丸一日あるので、慌てて判断する必要がありません。

---

## セットアップ（一度だけ、30分ほど）

### 1. GitHubリポジトリを作る

[github.com/new](https://github.com/new) で**Private**のリポジトリを作り、
このフォルダの中身をすべてアップロードします。

### 2. LINEの送信設定

LINE Notify は2025年3月末で終了したので、**LINE Messaging API** を使います。
個人利用なら月200通まで無料で、毎営業日送っても22通程度です。

1. [LINE Developers Console](https://developers.line.biz/console/) にLINEアカウントでログイン
2. 「新規プロバイダー作成」→ 名前は何でもよい（例: `myquant`）
3. 「Messaging API」チャネルを作成
4. **Messaging API設定**タブ → 下部の「チャネルアクセストークン（長期）」を発行してコピー
   → これが `LINE_CHANNEL_TOKEN`
5. 同じタブのQRコードを自分のLINEで読み取り、**Botを友だち追加**する
6. **チャネル基本設定**タブ → 「あなたのユーザーID」をコピー
   → これが `LINE_USER_ID`（`U` で始まる33文字）

> 応答メッセージ（自動返信）はオフにしておくと、送信のたびに定型文が返ってこなくなります。

### 3. GitHubにSecretsを登録

リポジトリの **Settings → Secrets and variables → Actions → New repository secret**

| 名前 | 値 | 必須 |
|---|---|---|
| `LINE_CHANNEL_TOKEN` | 手順2で発行したトークン | 必須 |
| `LINE_USER_ID` | `U` で始まるユーザーID | 必須 |
| `SEC_USER_AGENT` | `あなたの名前 メールアドレス` | 推奨 |
| `ALPACA_API_KEY` | Alpacaペーパー口座のキー | 任意 |
| `ALPACA_SECRET_KEY` | 同上 | 任意 |

`SEC_USER_AGENT` は SEC EDGAR が要求する連絡先です。
未設定でも動きますが、アクセスを制限されることがあります。

`ALPACA_*` は未設定なら Yahoo のニュースにフォールバックします。
Alpacaのペーパー口座は無料で作れて、Benzinga配信の質の高いニュースが取れます。

### 4. 動作確認

リポジトリの **Actions** タブから手動で実行できます。

1. 「LINE通知」→ Run workflow → `test` に ✓ → 実行
   → LINEに「接続テスト成功」が届けば設定完了
2. 「データ収集」→ Run workflow → `max_symbols` に `50` と入力 → 実行
   → 3〜5分で終わり、`reports/` にJSONができる
3. もう一度「LINE通知」を実行（testなし）
   → 収集した内容の要約が届く

以降は平日の朝に自動で動きます。

---

## 手元で検証する（任意）

バックテストは手元のパソコンでも走ります。

```bash
pip install -r requirements.txt

python tests/test_engine.py                        # 自己診断（31項目）
python backtest.py strategies/_offline_check.yaml  # ネット不要の動作確認
python build_universe.py --quick                   # 銘柄リストを作る
python backtest.py                                 # 実データで検証
```

`reports/` にHTMLレポートができます。

---

## ファイルの役割

| ファイル | 役割 |
|---|---|
| `config.yaml` | **普段いじるのはここ。** 売買ルールとパラメータ |
| `daily.py` | 毎朝のデータ収集（GitHub Actionsが実行） |
| `notify.py` | LINE送信 |
| `backtest.py` | 過去データでの検証 |
| `build_universe.py` | 銘柄リストの作成 |
| `qbt/engine.py` | バックテストの中核 |
| `qbt/validation.py` | 過学習の検出（ウォークフォワード等） |
| `qbt/feeds.py` | ニュース・SEC書類・マクロ指標の取得 |
| `qbt/universe.py` | 銘柄の母集団 |
| `state/positions.json` | 現在の保有銘柄 |

---

## 少額運用で気をつけること

**端株は必須です。** 2,000ドルを8銘柄に分けると1銘柄250ドル。
端株を使わないと株価250ドル超の銘柄（MSFT・AVGOなど）が最初から買えず、
ユニバースが勝手に低位株に偏ります。`config.yaml` の
`allow_fractional: true` がこれを解決しています。
moomoo・IBKR・Alpaca はいずれも米国株の端株売買に対応しています。

**PDT規制は2026年4月に撤廃されました。** 以前は5営業日に4回以上のデイトレードをするには
口座に25,000ドルが必要でしたが、この最低資本要件は廃止されます（12ヶ月の移行期間あり）。
少額でも短期売買ができるようになりました。

**生存者バイアスに注意。** 無料データには倒産・買収された銘柄が入っていません。
検証成績は実態より必ず良く出ます。有望な戦略が見えたら、
Point-in-Time対応のデータ（Sharadar 月$50前後）で再検証してください。

---

## このツールが守っていること

バックテストのバグは、ほぼ例外なく「成績が良くなる方向」に出ます。
以下は仕様として固定してあり、設定で緩めることはできません。

1. **未来を見ない** — シグナルは終値までの情報、約定は翌営業日の寄付。
   `shift()` に負の値を渡すと文法エラーになります
2. **コストを必ず引く** — スリッページを往復で控除
3. **資金制約を守る** — 現金がなければ買えない。建玉比率は100%を超えない
4. **損切りと利確が同日なら損切りを優先** — 保守的な方を採る

`python tests/test_engine.py` が毎回これを検証します。
特に **[9] ランダムな価格からは利益が出ないこと** が落ちたら、どこかで未来を見ています。
