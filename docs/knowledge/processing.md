# Processing Knowledge

`notebooks/processing/01_base_processing.ipynb` とここまでのやり取りを、知識メモとして整理したものです。

この文書の目的は次の 2 つです。

- 前処理で何をしているのかを notebook なしでも追えるようにする
- `bbox`, `segmentation`, `mask`, `keypoint` などのラベル追従で混乱しやすい点をまとめる

## 前提

このリポジトリでは `Milestone 1` の前処理確認に COCO を使っています。

特に確認したいのは次です。

- 画像だけを変換する前処理
- 画像とラベルを一緒に変換する前処理
- 前処理でラベルが壊れるパターン

## COCO の基本構造

COCO の JSON は大きく次の 3 つに分かれます。

- `images`
  - 画像そのものの情報
  - 例: `id`, `file_name`, `width`, `height`
- `annotations`
  - 画像内の物体 1 個ごとのラベル
  - 例: `image_id`, `category_id`, `bbox`, `segmentation`
- `categories`
  - `category_id` とクラス名の対応表

### annotation 1 件の中身

`instances` 系 annotation は典型的に次のような dict です。

```python
{
    "id": 1768,
    "image_id": 289343,
    "category_id": 18,
    "segmentation": [[...], [...]],
    "area": 702.10,
    "bbox": [473.07, 395.93, 38.65, 28.67],
    "iscrowd": 0,
}
```

各キーの意味:

- `id`
  - annotation 自体の一意 ID
- `image_id`
  - どの画像に属する annotation か
- `category_id`
  - 物体クラスの ID
- `bbox`
  - COCO 形式の bbox
  - `[x, y, width, height]`
- `area`
  - annotation 領域の面積
- `segmentation`
  - 物体の形状情報
- `iscrowd`
  - crowd annotation かどうか

### `instance_annotations_by_image` とは何か

notebook では `image_id` ごとに annotation をまとめた辞書を使っています。

形はこうです。

```python
{
    289343: [
        {... annotation 1 ...},
        {... annotation 2 ...},
    ],
    123456: [
        {... annotation 1 ...},
    ],
}
```

つまり:

- `key` は `image_id`
- `value` はその画像に属する annotation の list

これにより、画像 1 枚に対する bbox や mask をすぐ引けます。

## `bbox` と `segmentation` の違い

### `bbox`

`bbox` は物体を囲む長方形です。

- 粗い位置情報
- 計算が軽い
- 検出タスクや crop, ROI 抽出に向く

### `segmentation`

`segmentation` は、物体が画像のどのピクセル領域にあるかを、物体の形に近い形で表した情報です。

たとえば犬が写っているとき:

- `bbox`
  - 犬全体を囲む四角
- `segmentation`
  - 耳や胴体の輪郭に近い領域

つまり `segmentation` は `bbox` より細かい位置情報です。

### COCO の `segmentation` の形式

COCO では主に次の 2 形式があります。

- polygon
  - 輪郭の頂点を `[x1, y1, x2, y2, ...]` で持つ
- RLE
  - mask を圧縮した形式
  - `iscrowd=1` で出やすい

### `segmentation` があるなら `bbox` は不要ではないのか

直感的にはそう見えますが、実務では両方使います。

理由:

- `bbox` の方が軽く扱いやすい
- 物体検出は bbox を直接予測することが多い
- crop, ROI 抽出, 小物体確認, annotation cleaning に便利
- `segmentation` は形状情報としてより細かいが、毎回それを使うのは重い

整理すると:

- `bbox`
  - 粗い位置情報
  - 軽い
  - 検出向き
- `segmentation`
  - 細かい形状情報
  - 重い
  - instance segmentation, mask 可視化向き

`segmentation` は `bbox` の完全上位互換というより、役割の違う別種のラベルです。

## 前処理で重要な考え方

前処理は大きく 2 種類に分かれます。

### 1. 画像だけを変える前処理

例:

- color space conversion
- threshold
- denoise
- morphology
- normalize

これは画素値を変える処理で、基本的に bbox 座標は変わりません。

### 2. 画像の幾何を変える前処理

例:

- resize
- crop
- pad
- flip
- affine transform
- perspective transform

これは画像内の位置関係が変わるので、bbox, mask, keypoint も追従させる必要があります。

ここを間違えると、画像とラベルがずれて学習が壊れます。

## notebook で扱った前処理

## resize

目的:

- モデル入力サイズをそろえる
- batch 化しやすくする
- メモリ使用量をある程度そろえる

どうやるか:

- 画像は `cv2.resize`
- bbox は横方向に `scale_x`
- bbox は縦方向に `scale_y`

理解ポイント:

- 画像だけ resize して bbox を変えないとラベルが壊れる
- 横と縦で倍率が違うなら bbox も別倍率で伸縮する

## center crop

これはほぼトリミングです。

やっていること:

- 画像の中央に四角を置く
- その四角の外側を切り落とす
- 残した四角の左上を新しい `(0, 0)` として bbox 座標を引き直す

例:

- 元画像が `640 x 480`
- `500 x 500` を切り出したい
- 横方向は `640 - 500 = 140` 余るので、左 `70`, 右 `70` を捨てる
- bbox が元画像で `[100, 120, 80, 60]` なら、crop 後は `x=30` から始まる bbox として考え直す

crop の外に大きくはみ出す bbox は:

- 見えている部分だけ残す
- ほとんど消えたら捨てる

### なぜ before / after が見た目で分かりにくいのか

今回 notebook の出力では差が小さく見えることがあります。

理由:

- crop サイズがまだ大きい
- 中央付近の内容がほとんど残っている
- matplotlib が同程度の大きさで表示している

つまり:

- `resize`
  - 全体を縮めたり伸ばしたりする
- `center crop`
  - 中央だけ残して周辺を切る

見た目の差を強くしたいなら、`300 x 300` などもっと小さい crop にすると分かりやすいです。

## pad to square
余白を作ることで、画像のサイズをそろえる。余白ができた分位置の修正が必要。

目的:

- 縦横比を大きく壊さずに正方形入力を作る
- letterbox 的な発想を理解する

どうやるか:

- 足りない方向に余白を足す
- bbox は余白ぶんだけ平行移動する

理解ポイント:

- resize と違って bbox の大きさは変わらない
- 位置だけずれる

## horizontal flip

目的:

- 左右反転 augmentation の基本を理解する

どうやるか:

- 画像を左右反転
- bbox の左端 `x` を `image_width - x - w` に変える

理解ポイント:

- これは倍率変換ではなく鏡映変換
- keypoint がある場合は左右ラベル交換も必要になる

## affine transform

目的:

- 回転, 平行移動, 拡大縮小を一つの枠組みで扱う

どうやるか:

- `cv2.getRotationMatrix2D` で affine 行列を作る
- `cv2.warpAffine` で画像を変換する

理解ポイント:

- resize, shift, rotate をまとめて扱える
- bbox を厳密に追うには少し複雑

## perspective transform

目的:

- 遠近感のある変形を扱う

どうやるか:

- 4 点対応から perspective 行列を作る
- `cv2.warpPerspective` を使う

理解ポイント:

- affine より自由度が高い
- bbox より polygon や mask の方が自然に扱える場面がある

### 具体的に画像をどうしたいのか

`perspective transform` は、簡単に言うと「カメラの見え方の違いをまねる変形」です。

イメージ:

- 紙を真正面から撮る
  - 長方形に見える
- 紙を斜めから撮る
  - 台形っぽく見える

この「斜めから見た感じ」を作るのが perspective transform です。

つまり、画像をどうしたいのかというと:

- 正面の見え方を
- 斜めから見たような見え方に変えたい

### affine との違い

- `affine`
  - 回転, 平行移動, 拡大縮小, せん断
  - 平行な線は平行のまま
- `perspective`
  - 遠近感のあるゆがみを作れる
  - 平行線が平行でなくなってもよい

一言でいうと:

- `affine` は平面上のきれいな変形
- `perspective` は視点が変わったような変形

### 使う場面

- 書類や看板を斜め撮影した画像
- 監視カメラで角度がついた映像
- 実運用で視点がぶれるケースの augmentation
- OCR や document 処理で台形補正を考えるとき

## color space conversion

目的:

- threshold や document 系処理の前段を理解する
- RGB 以外の表現を知る

扱ったもの:

- grayscale
- HSV

理解ポイント:

- RGB のままだと見にくい情報がある
- 明るさや色相を分けて扱える

### 目的に応じてどう使い分けるか

- `RGB / BGR`
  - 普通のカラー画像として扱いたいとき
  - 表示用
  - 学習入力の基本形

- `Grayscale`
  - 明るさだけ見たいとき
  - threshold
  - edge 検出
  - document 処理
  - ノイズ除去の前段

- `HSV`
  - 色そのものと明るさを分けたいとき
  - 特定の色だけ抽出したいとき
  - 照明変化に少し強く色判定したいとき
  - 例: 赤い物体だけマスクしたい

- `LAB`
  - 人間の知覚に近い形で明るさと色差を分けたいとき
  - 色の違いを見たいとき
  - illumination の影響を少し切り分けたいとき
  - 画像補正や色差評価

- `YCrCb`
  - 輝度と色差を分けたいとき
  - 肌色抽出
  - 映像処理でよく使う
  - 明るさだけ処理したい場合

- `HLS`
  - HSV に近いが lightness ベースで見たいとき
  - 光の強さと色を分けて扱いたい場合

### 具体例

- 書類を白黒にしたい
  - `Grayscale -> threshold`
- 赤いマーカーだけ取りたい
  - `HSV`
- 肌色領域をざっくり取りたい
  - `YCrCb` か `HSV`
- 明るさだけ均したい
  - `LAB` の `L` や `YCrCb` の `Y`
- 普通の分類モデルへ入れる
  - 多くは `RGB`

## threshold

目的:

- 画素を白黒に分けて情報を切り出す
- document, binary mask 的な発想を理解する

どうやるか:

- `cv2.threshold`

理解ポイント:

- threshold は見た目改善よりも情報の切り分けが目的

## denoise

目的:

- ノイズを弱める
- threshold や edge 系処理の前段を理解する

扱ったもの:

- Gaussian blur
- Median blur

理解ポイント:

- blur の種類で消え方が違う
- エッジを残すか, ノイズを消すかで向き不向きがある

### なぜ denoise するとぼやけるのか

`denoise` の目的はノイズ除去です。ですが、ノイズ除去をすると多少ぼやけるのは普通です。

理由:

- ノイズ除去は細かい揺らぎや急な画素値変化を弱める処理
- 画像の細かい模様や輪郭も、見方によっては急な画素値変化
- そのためノイズと一緒に細部も少し消える

結果として、ぼやけて見えることがあります。

### なぜオリジナル画像の方がはっきり見えるのか

オリジナル画像には次の両方が入っています。

- 本物の細部
- ノイズ

人間の目には、その両方込みでシャープに見えることがあります。

つまり:

- 見た目のシャープさ
- ノイズが少ないこと

は同じではありません。

### GaussianBlur と MedianBlur の違い

- `GaussianBlur`
  - 周囲の画素を重み付き平均する
  - 全体的に自然にぼかす
  - 細かいノイズに強い

- `MedianBlur`
  - 周囲の画素の中央値で置き換える
  - 塩胡椒ノイズに強い
  - 条件によってはエッジを保ちやすい
  - 画像によってはテクスチャがつぶれて、よりぼやけて見えることがある

### どういうときに denoise を考えるか

`denoise` は、ノイズが後段処理の邪魔をするときに検討します。

典型例:

- threshold で小さなゴミが大量に出る
- edge 検出がざらつきに反応しすぎる
- OCR で文字の周りにノイズが多い
- 小さな斑点を誤検出してしまう
- 暗所画像や高 ISO 画像でざらつきが強い
- 圧縮ノイズやセンサノイズが目立つ

### 逆に毎回入れるものではない

画像が十分きれいで、細かい模様やエッジが重要なときは、denoise を入れない方がよいこともあります。

判断基準:

- オリジナル画像で後段処理が安定しているなら不要
- ノイズのせいで誤検出や不安定さが出るなら検討
- 見た目ではなく、後段の結果が改善するかで決める

## morphology

目的:

- 二値画像の形を整える

扱ったもの:

- opening
- closing

理解ポイント:

- morphology は白黒画像の形を変える処理
- 小さいノイズ除去や穴埋めに使う

### 具体的にどういう画像にしたくてやるのか

`morphology` は、白黒画像の形を整えたいときに使います。

イメージ:

- 小さなゴミを消す
- 切れた部分をつなぐ
- 小さな穴を埋める
- 輪郭を少し太くする
- 輪郭を少し細くする

つまり「見た目をきれいにする」というより、

- threshold 後の白黒領域を
- 後段処理しやすい形に直す

のが目的です。

### よくある目的

- OCR で文字の周りの小ノイズを消したい
- 二値化後に点々が多すぎる
- マスクの穴を埋めたい
- 途切れた線をつなぎたい
- 小さい白点を除去したい

### 代表的な操作

- `opening`
  - 小さい白ノイズを消したいとき
- `closing`
  - 白領域の小さい穴を埋めたいとき
- `dilation`
  - 白い部分を太らせたいとき
- `erosion`
  - 白い部分を細らせたいとき

### どういう after を目指すか

一言でいうと、

- ノイズっぽい細かい白黒を減らして
- 意味のある領域だけがまとまって見える画像

を目指します。

## normalize

目的:

- モデル入力用に数値分布を整える

どうやるか:

- 画素値を `[0, 1]` にする
- mean, std で標準化する

理解ポイント:

- normalize は見た目を良くする処理ではない
- normalize 後の画像をそのまま表示すると色が不自然でも正常

## Albumentations

### image のみ変換

目的:

- augmentation を簡単に比較する

理解ポイント:

- Albumentations は augmentation 比較に向いている
- 画像だけなら書きやすい

### image + bbox 同期変換

目的:

- 画像と bbox を同時に安全に変換する

やること:

- bbox を COCO 形式のまま渡す
- transform 後の bbox を返してもらう

理解ポイント:

- bbox を壊さず同期変換できることが重要

### 画像をどうしたくてやっているのか

Albumentations の目的は、学習時に見せる入力のバリエーションを増やすことです。

たとえば実運用では:

- 少し暗い
- 少しぼける
- 少し斜め
- 少しずれる
- 左右反転しても本質は同じ

のようなことが起きます。Albumentations は、そういう変化を人工的に画像へ入れます。

### 中で何をしているのか

Albumentations は、複数の augmentation を順番に適用します。

例:

- `HorizontalFlip`
  - 左右反転
- `RandomBrightnessContrast`
  - 明るさやコントラストを変える
- `GaussianBlur`
  - 少しぼかす
- `Affine`
  - 回転, 平行移動, 拡大縮小
- `CoarseDropout`
  - 画像の一部を四角く隠す

これらを `Compose` でまとめて書きます。

### いろんなパターンを作れるのか

はい。ただし、完全に自由に自動生成するわけではありません。

正確には:

- 変換の種類は自分で指定する
- 各変換の値は指定した範囲の中でランダムに変わる

つまり、

- 各パターンを 1 枚ずつ手で用意する必要はない
- ただし「何をどこまで変えてよいか」は自分で書く必要がある

です。

たとえば:

```python
A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), p=0.8)
```

と書くと、

- 拡大縮小は `0.9` から `1.1`
- 回転は `-10` 度から `10` 度

の範囲でランダムに変換が起きます。

一言でいうと、

- 変換の種類は自分で決める
- 具体的な変形量はランダムに振れる

です。

### 実装上の注意

Albumentations の bbox 同期では、返ってくる `bbox_indices` が float になることがありました。

その場合は:

```python
ann_index = int(ann_index)
```

としてから `selected_annotations[ann_index]` に使う必要があります。

そうしないと:

- `TypeError: list indices must be integers or slices, not float`

になります。

## mask と keypoint

## mask

notebook では `segmentation` から mask をデコードして画像に重ねています。

理解ポイント:

- bbox は四角
- mask は物体の形
- 同じ annotation でも情報量が違う

## keypoint

`keypoint` は、物体の中の重要な点の座標です。

たとえば人なら:

- 鼻
- 目
- 肩
- 肘
- 手首
- 腰
- 膝
- 足首

のような点です。

つまり:

- `bbox`
  - 人全体の四角
- `keypoint`
  - 人の姿勢を表すための点の集合

です。

### notebook の keypoint 可視化で何をしているか

notebook では次をしています。

- COCO の `keypoints` を読む
- その点を画像の上に丸で描く
- `visibility` がある点だけ表示する

COCO の keypoint は、典型的には次の並びです。

```text
[x1, y1, v1, x2, y2, v2, ...]
```

- `x, y`
  - 点の位置
- `v`
  - その点が見えているかどうか

`v` の意味:

- `0`
  - ラベルなし
- `1`
  - 見えていないが位置情報はある
- `2`
  - 見えている

可視化では通常:

- `v > 0` の点を描く
- 必要なら点同士を線でつなぐ

ということをします。

理解ポイント:

- bbox とは別種のラベル
- flip 時は左右の点ラベル交換が必要
- visibility の扱いがある

### 何のために可視化するのか

目的:

- keypoint ラベルが正しい場所についているか確認する
- flip や crop で点が壊れていないか確認する
- 姿勢推定用データのイメージをつかむ

## 可視化の意味

前処理は、コードだけ見ても本当に合っているか分かりにくいことがあります。

可視化で確認すべきこと:

- bbox が物体からずれていないか
- crop で bbox が消えすぎていないか
- pad で bbox の位置だけずれているか
- mask が正しい形で重なっているか
- keypoint が左右反転で壊れていないか

## PyTorch tensor 化

最後に notebook では PyTorch 入力の直前も確認しています。

やること:

- `HWC` から `CHW` に変換
- `torch.tensor` に変える

理解ポイント:

- OpenCV / NumPy と PyTorch では次元順が違う

## COCO で前処理確認をやる利点

COCO を使うと、一つのデータセットで次を触れます。

- bbox
- mask
- person keypoint

そのため `Milestone 1` の入口としてはかなり使いやすいです。

ただし分類専用データではないので:

- confusion matrix
- class imbalance を分類文脈で深く見る

にはあまり向きません。

## notebook と知識メモの役割分担

notebook は次に向いています。

- before / after を見る
- 実際に画像を変換する
- bbox, mask, keypoint のずれを見る

一方で notebook は外部編集との相性が少し不安定です。

そのため、この `processing.md` には:

- notebook で理解した内容
- 会話で補足した内容
- 見た目で混乱しやすかった点

を集約しています。

## この段階で覚えておきたい要点

- `bbox` は `[x, y, width, height]`
- `segmentation` は物体の形に近い領域情報
- `bbox` と `segmentation` は役割が違うので両方使う
- 幾何変換では画像だけでなくラベルも追従させる必要がある
- `resize` は倍率変換
- `crop` は中央を残して周辺を切り落とす処理
- `pad` は余白を足すので bbox は平行移動
- `flip` は鏡映変換
- `affine`, `perspective` は自由度が上がるほどラベル追従が難しい
- `threshold`, `denoise`, `morphology` は画像内容を整理する前処理
- `normalize` はモデル入力用の数値変換
- Albumentations は image とラベルの同期変換を安全に書きやすい
- COCO では bbox, mask, keypoint の入口まで確認できる
