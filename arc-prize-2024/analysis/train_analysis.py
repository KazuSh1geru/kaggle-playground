from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()

PROMPT = f"""
Kaggle のコンペティションに取り組んでいます。
ARC-AGIの問題に取り組んでいます。

インプットを与えるので、参考出力のようにtrainingデータセットの回答を一つ一つどのようなタスクなのか解説して、CSVにまとめて欲しいです。

## インプット
{input}

## 参考出力
id, データ構造, 観察ポイント, タスク傾向, 具体的なデータ, 考察
007bbfb7, 
データ構造訓練データ（train）：5つの入力/出力ペアが含まれています。
テストデータ（test）：1つの入力グリッドが含まれています。,

入力と出力のサイズ
入力グリッド：すべて3x3のサイズです。
出力グリッド：すべて9x9のサイズです。,

入力グリッドが何らかの形で拡大または繰り返しされている可能性があります。
出力グリッドは、入力グリッドの要素を組み合わせて新しいパターンを形成しています。,

最初の訓練ペア
入力
[ [0,7,7],
  [7,7,7],
  [0,7,7] ]
出力
[ [0,0,0,0,7,7,0,7,7],
  [0,0,0,7,7,7,7,7,7],
  [0,0,0,0,7,7,0,7,7],
  [0,7,7,0,7,7,0,7,7],
  [7,7,7,7,7,7,7,7,7],
  [0,7,7,0,7,7,0,7,7],
  [0,0,0,0,7,7,0,7,7],
  [0,0,0,7,7,7,7,7,7],
  [0,0,0,0,7,7,0,7,7] ]
,

入力グリッドを特定のパターンで拡大または配置しているように見えます。
元の入力グリッドが回転、反転、または複数回配置されている可能性があります。

"""

load_dotenv()
# .envからAPIキーを取得
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if __name__ == "__main__":
    # arc-agi_training_solutions.jsonを読み込む
    with open("arc-prize-2024/inputs/arc-agi_training_solutions.json", "r") as f:
        data = json.load(f)
    # 順々にデータを取得する
    for id, solution in data.items():
        input = f"id: {id}\n{solution}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT.format(input=input)}],
        )
        print(response)
        break
