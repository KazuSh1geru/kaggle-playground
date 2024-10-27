# Overview

コンペティションの概要を以下にまとめます。

## 概要

このコンペティションでは、大規模なデータセットに依存せず、新しいスキルを効率的に学習し、オープンエンドな問題を解決できるAIシステムの開発を目指します。上位の提出物は、人間の推論ベンチマークへの改善を示す必要があります。

## 説明

現在のAIシステムは、大規模なデータセットで広範な訓練を受けても、訓練データ外の新しい問題に一般化することができません。LLM（大規模言語モデル）は既知のタスクにおいて主流化していますが、AGI（人工汎用知能）への進歩は停滞しています。AGIの改善は、人間と共に考え、発明するAIシステムを可能にします。

**ARC-AGI（Abstraction and Reasoning Corpus for Artificial General Intelligence）ベンチマーク**は、AIシステムが新しいスキルを効率的に学習する能力を測定します。人間はARCで容易に85%のスコアを達成できますが、最先端のAIシステムでも34%にとどまります。ARC Prizeコンペティションは、大規模なデータセットに依存し、新しい問題に対処しにくいLLM以外のアイデアを探求することを奨励しています。

このコンペティションには複数の要素があります。ここで説明されているコンペティションには$100,000の賞金があり、リーダーボードで85%のスコアを超えたチームには追加で$500,000が与えられます。Kaggle以外でも関連する賞金がある機会が提供されています。詳細は[**ARCprize.org**](http://arcprize.org/)をご覧ください。

あなたの取り組みは、産業全体で適用可能な新しいAIの問題解決に貢献する可能性があります。大幅に改善されたAGIは、人間と機械の相互作用を再定義するでしょう。受賞したソリューションは、AGI分野での透明性と協力を促進するためにオープンソース化されます。

## 評価

提出物は、正答率に基づいて評価されます。各タスクについて、そのタスクに含まれるすべてのテスト入力グリッドに対して正確に2つの出力を予測する必要があります（タスクには複数のテスト入力が含まれる場合があります）。各タスクのテスト出力には1つの正解があります。特定のタスク出力に対して、2つの予測のうちいずれかが正解と完全に一致すれば、そのタスクテスト出力について1点を獲得し、そうでなければ0点です。最終スコアは、各タスク出力ごとの最高スコアの合計をタスクテスト出力の総数で割ったものになります。

## 提出ファイル

提出ファイルは、`submission.json`という名前のJSONファイルである必要があります。

評価セットの各タスク出力について、正確に2つの予測（`attempt_1`、`attempt_2`）を行う必要があります。予測の構造は以下の例で示されています。ほとんどのタスクは単一の出力（リストに含まれる1つの辞書）ですが、複数の出力を予測する必要があるタスクもあります。これらは、以下の例のように、リストに含まれる2つの予測辞書を含める必要があります。複数のテスト出力が必要なタスク（例：タスク`12997ef3`）では、対応するテスト入力と同じ順序でなければなりません。

**重要**：入力のチャレンジJSONファイル内のすべての`task_id`は、`submission.json`ファイル内にも存在しなければなりません。`"attempt_1"`と`"attempt_2"`は、予測が2つない場合でも必ず含める必要があります。

## データセット

コンペティションのデータセットに関する情報を以下にまとめます。

## データセットの説明

このコンペティションの目的は、抽象的な推論タスクを解決できるアルゴリズムを作成することです。これらのタスクはアルゴリズムがこれまでに見たことのない新規のものであり、単純に一連の推論パターンを記憶するだけでは対応できません。

以前のコンペティションとは形式が異なるため、この情報を注意深く読み、必要に応じて補足資料を参照してください。

タスクを見る際、「受験者」はデモンストレーションペア（トレインペア）の入力と出力、およびテストペアの入力にアクセスできます。目標は、各テスト入力グリッドに対応する出力グリッドを、各テスト入力につき2回の試行で構築することです。出力グリッドの構築とは、高さと幅を選択し、グリッド内の各セルにシンボル（0から9までの整数で、色として視覚化される）を配置することを指します。すべてのセルが正解と一致した場合にのみ、解答は正しいと見なされます。

追加の情報やこのコンペティションの目的を詳しく理解するためのインタラクティブなアプリは、[**ARCPrize.org**](http://arcprize.org/)で見つけることができます。目的を深く理解するために、このアプリを活用することを強くお勧めします。

## タスクファイル

情報は以下のファイルに保存されています：

- `arc-agi_training-challenges.json`: 各タスクの「テスト」入力に適用される推論パターンを示す入力/出力ペアを含みます。このファイルと対応する解答ファイルは、モデルのトレーニングに使用できます。
- `arc-agi_training-solutions.json`: 対応するタスクの「テスト」出力（正解）を含みます。
- `arc-agi_evaluation-challenges.json`: モデルの検証データとして使用できる入力/出力ペアを含みます。
- `arc-agi_evaluation-solutions.json`: 対応するタスクの「テスト」出力（正解）を含みます。
- `arc-agi_test-challenges.json`: リーダーボード評価に使用されるタスクが含まれています。各タスクには「トレイン」の入力/出力ペアと「テスト」入力が含まれます。あなたのタスクは「テスト」出力を予測することです。**注意**：このページに表示されているファイルは、`arc-agi_evaluation-challenges.json`からのタスクを使用したプレースホルダーです。ノートブックを提出して再実行すると、このファイルは実際のテストチャレンジに置き換えられます。
- `sample_submission.json`: 正しい形式の提出ファイルのサンプルです。

各タスクは、次の2つのフィールドを持つ辞書で構成されています：

- `"train"`: デモンストレーションの入力/出力ペアのリスト（通常は3つのペア）。
- `"test"`: モデルが出力を予測すべきテスト入力。

「ペア」は、以下の2つのフィールドを持つ辞書です：

- `"input"`: ペアの入力グリッド。
- `"output"`: ペアの出力グリッド。

「グリッド」は、0から9までの整数（含む）からなる長方形の行列（リストのリスト）です。最小サイズは1x1、最大サイズは30x30です。

このページのデータは、モデルの開発と評価に使用するべきものです。ノートブックを提出して再実行すると、`arc-agi_test_challenges.json`という名前の再実行ファイルにある未公開の100のタスクがスコアリングに使用されます。これらのタスクには、入力と出力のトレインペア、およびタスクのテスト入力が含まれます。あなたのアルゴリズムはテスト出力を予測する必要があります。リーダーボードのスコアに使用される100のタスクの大部分は、予測が必要なテスト入力が1つだけですが、一部のタスクでは2つのテスト入力に対する予測が求められます。