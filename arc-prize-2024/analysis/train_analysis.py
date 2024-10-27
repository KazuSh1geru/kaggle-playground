from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import csv
from pprint import pprint

load_dotenv()

def get_prompt(input_data):
    return f"""
Kaggle のコンペティションに取り組んでいます。
ARC-AGIの問題に取り組んでいます。

インプットを与えるので、参考出力のようにtrainingデータセットの回答を一つ一つどのようなタスクなのか解説して、CSVにまとめて欲しいです。

## アウトプットポリシー
- CSVのテキストのみにしてください。
- inputのidは、データから読み取って上書きしてくださいそのままにしてください。（example: 007bbfb7）

## CSVのヘッダーとデータの項目
1. `id` - 各入力のユニークな識別子。
2. `データ構造` - 各データセット内の訓練データとテストデータの構造。
3. `観察ポイント` - 入力と出力のサイズや構造における重要な特徴。
4. `タスク傾向` - 入力がどのように処理されて出力に移行するかの傾向。
5. `具体的なデータ` - 特定の訓練ペアに関する詳細なデータ。
6. `考察` - パターンやルールの仮説。


## インプット
{input_data}


## 参考出力

## inputのデータ構造
    id: 007bbfb7
    data: [
        [
                7,
                0,
                7,
                0,
                0,
                0,
                7,
                0,
                7
            ],
            [
                7,
                0,
                7,
                0,
                0,
                0,
                7,
                0,
                7
            ],
            [
                7,
                7,
                0,
                0,
                0,
                0,
                7,
                7,
                0
            ],
            [
                7,
                0,
                7,
                0,
                0,
                0,
                7,
                0,
                7
            ],
            [
                7,
                0,
                7,
                0,
                0,
                0,
                7,
                0,
                7
            ],
            [
                7,
                7,
                0,
                0,
                0,
                0,
                7,
                7,
                0
            ],
            [
                7,
                0,
                7,
                7,
                0,
                7,
                0,
                0,
                0
            ],
            [
                7,
                0,
                7,
                7,
                0,
                7,
                0,
                0,
                0
            ],
            [
                7,
                7,
                0,
                7,
                7,
                0,
                0,
                0,
                0
            ]
        ]
    ],

## 期待するアウトプット:
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
    # CSVファイルを開く（追記モード）
    with open("arc-prize-2024/outputs/arc_analysis.csv", "a", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 順々にデータを取得する
        for id, solution in data.items():
            input_data = f"id: {id}\ndata: {solution}"
            # print(f"input: {input_data}")
            # print(get_prompt(input_data=input_data))
            # break
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": get_prompt(input_data=input_data)}],
            )
            pprint(response)
            # responseからCSVに書き込む
            
            csvwriter.writerow([response.choices[0].message.content.strip()])
