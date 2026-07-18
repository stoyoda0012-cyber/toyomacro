# Toyomacro HDF5 provenance schema 設計 (I/O enrichment Phase B-1)

作成日: 2026-07-18 / 承認・判断反映: 2026-07-18 (§12 参照)
対象: Phase A (commit `79cc185`) で reader が保持するようになった provenance を、
既存 Toyomacro HDF5 と後方互換のまま永続化するための schema 設計。

背景資料: UQ backbone 監査ノート (以下「監査」)。ローカル作業ツリーでは
`docs/uncertainty_backbone_audit.md` に置かれた**未追跡の内部資料**であり、
リポジトリには収録されない。本書の判断に必要な事実は本文へ引用済みのため、
同資料がなくても本書は単体で読める (「監査 §N」の参照は出典表示であり
必須依存ではない)。

---

## 0. 形式の位置づけ

本書で扱う形式の関係を先に固定する。

- **PXT / IBW / VAMAS / NPL / SES**: Toyomacro が読む「上流の入力形式」。
  装置・装置付属ソフトウェア側が出力するファイルであり、Toyomacro はその
  読者にすぎない。
- **`source_format`**: HDF5 へ変換する前に、どの上流入力形式から読んだかを
  示すラベル (`scienta_pxt` 等)。**メーカー公式 schema への準拠認証や、
  メーカーとの提携・承認を意味しない**。Toyomacro reader が解釈できた範囲の
  自己申告である。
- **Toyomacro HDF5**: 装置メーカーが定めた形式ではなく、MATLAB 版 Toyomacro
  時代から使い続けている **Toyomacro ローカルな中間・保存形式**。schema の
  決定権も互換性責任も本プロジェクトにある。
- **`/provenance`**: 上流入力から読んだ事実 (hv、pass energy、次元構造など) と、
  Toyomacro reader / importer がその後に行った変換の記録。上流形式の再現では
  なく「何をどう読んだか」の記録である。

したがって §9 末尾の互換性確認も「メーカー形式との互換性」ではなく、
**従来の Toyomacro MATLAB ローカル HDF5 consumer との互換性**を指す。

---

## 1. 現行 schema の事実

実装から確認した事実のみを記す (根拠は `ファイル:行`)。

### 1.1 コア layout

| object | shape / dtype | 根拠 |
|---|---|---|
| `/specdata` | `(n_spectra+1, n_energy)` float32、row 0 = energy | `src/toyomacro/io/schema.py:36,43`、`writers/hdf5_writer.py:118-123` |
| `/fitpara` | `(n_spectra, maxcomp, 9)` float32 | `schema.py:37`、`hdf5_writer.py:126-131` |
| `/otherpara` | `(10, n_spectra)` float32 | `schema.py:38`、`hdf5_writer.py:134-139` |
| `/xytdata` | `(3, n_spectra)` float32 | `schema.py:39`、`hdf5_writer.py:142-147` |
| `/misc` | group。scalar/str dataset の集合 | `hdf5_writer.py:322-337` |

### 1.2 misc の既存内容と sentinel

- 既定 key: `xdata_str, ydata_str, tdata_str, numberofslice, fermienergy, bindingenergysign, sodeconv, maxcomp` (`schema.py:105-125`)。
- importer が追記: `source_file` (入力ファイル名)、`dim_shape`、`dim0_is_angle` (`io/importer.py`)。
- `fermienergy` は **0.0 が「未知」の legacy sentinel** (`schema.py:121`、Phase A で明示コメント化。`importer.py` の misc 構築部)。
- `dim0_is_angle` は Phase A 後、reader の `dimension_roles[1] == "emission_angle"` から導出される (`importer.py:369-378`)。チャンネル数や拡張子だけからの angle 推定は廃止済み。

### 1.3 圧縮ファイルの追加 dataset

`compress_h5_file_streaming` は `fitpara_compressed` / `otherpara_compressed` /
`xytdata_compressed` / `specdata_uint16` (+`specdata_uint16_energy`) を生成する
(`io/compression.py:740-797`)。つまり **既存 consumer は既に「root に未知の
dataset が増えたファイル」を日常的に扱っており、名前指定アクセスで動いている**。

### 1.4 version の現状

- `ToyomacroSchema.VERSION = "1.1.0"` は定数として存在するが、**リポジトリ内の
  どのコードもファイルへ書き込んでいない** (使用箇所は定義行 `schema.py:33` のみ。
  `grep -rn "VERSION" src/toyomacro/io/` で確認)。
- したがって既存ファイルはすべて「version 無記録」であり、これが後方互換規則の
  出発点になる (§8)。
- root attrs としては crash recovery marker `_toyomacro_incomplete` /
  `_toyomacro_last_batch` (旧 `_voigtfit_*`) だけが使われている
  (`io/writers/streaming_writer.py:383-384`、`io/recovery.py:196-201`)。

### 1.5 Phase A で in-memory に存在する provenance (永続化対象)

`SpectrumMetadata` (`io/readers/base_reader.py:111-131`):
`source_format`, `source_format_version`, `source_region_index`, `region`,
`excitation_energy: float|None`, `pass_energy: float|None`,
`intensity_semantics`, `intensity_unit`, `original_shape`, `dimension_roles`,
`vendor_metadata` (JSON-safe dict、deep copy 済)。
`RawSpectrumData.transforms: tuple[ReaderTransform, ...]`
(`name/parameters/source/reason`、`to_dict()` で JSON-safe dict 化)。

---

## 2. 推奨 HDF5 layout

```
/                                   (root)
  @toyomacro_schema_version = "1.2.0"      # 新規: file schema version (§4)
  /specdata, /fitpara, /otherpara, /xytdata, /misc   # 既存、不変更
  /provenance                              # 新規 group (additive)
    @schema_version        = "1.0"         # provenance 独自 version
    @source_format         = "scienta_pxt"
    @source_format_version = "3"           # 未知なら attr 自体を省略
    @source_region_index   = 0             # int64
    @region_name           = "Si2p"        # "" = ファイルに名前なし
    @excitation_energy_eV  = 1486.6        # float64。未知なら省略
    @pass_energy_eV        = 50.0          # float64。未知なら省略
    @intensity_semantics   = "unknown"     # 固定語彙 (§3.1)
    @intensity_unit        = "unknown"
    @acquisition_mode      = "Swept"       # 明示値がある場合のみ ("" は省略)
    @datetime              = "2026-01-15T10:30:00"  # ISO 8601。opt-in のみ (§3)
    @original_shape        = [401, 128]    # int64 1D attr。() なら省略
    @dimension_roles       = ["energy", "emission_angle"]  # vlen-str 1D attr
    @transform_history_incomplete = True   # JSON 化失敗 drop があった時のみ (§6.2)
    @dropped_transform_records    = 1      # 同上 (int64)
    transform_history      # dataset (n_records,) vlen-str UTF-8 JSON (§6)
                           # 履歴が空なら dataset 自体を作らない
    vendor_metadata        # dataset (1,) vlen-str JSON。opt-in 時のみ (§7)
      @policy_version = "1.0"
      @allowlist      = ["Detector Mode", ...]
  /uncertainty                             # ★予約のみ。B-1 では作らない (§4.3)
```

### 2.1 「scalar は attr、可変長は dataset」の根拠

単なる好みではなく、既存の複製経路 2 本の実装差から導かれる:

1. **圧縮経路は group を丸ごと verbatim コピーする。**
   `compress_h5_file_streaming` の Pass 1 は
   `{specdata, fitpara, otherpara, xytdata}` 以外の全 root オブジェクトを
   `f_in.copy()` し、root attrs も明示コピーする (`io/compression.py:727-738`)。
   → `/provenance` は attrs/dataset とも無傷で通る。
2. **repack 経路は dataset を「作り直す」。** `repack_file` は group と
   group attrs を verbatim 再作成する (`voigtfit/tools/repack_h5.py:302-307`,
   root attrs は 269-270) が、dataset は `repack_dataset` を通り、
   - `target_dtype=None` でも **float64 dataset を float32 へ自動降格する**
     (`repack_h5.py:174-179`)、
   - scalar dataset (shape `()`) は `get_optimal_chunks` が `()` を返し
     (`repack_h5.py:89`)、`create_dataset(chunks=())` が失敗し得る、
   - shape `(0,)` の空 dataset も chunk 不能。

   → **float scalar を dataset として置くと repack で精度劣化または破損リスク**。
   attr なら両経路とも verbatim。よって scalar provenance は全て group attr とし、
   dataset は「1 次元・要素数 ≥ 1・文字列型」に限定する。transform 履歴が空の
   ときに dataset を省略するのは `(0,)` chunk 不能への対処でもある。

この知見は将来の uncertainty schema にも波及する: **float64 の covariance を
dataset で保存すると、`repack_h5` の既定 (`force_float32=True`) が黙って
float32 化する**。B-2 以降の設計で必ず再考すること (§12-7)。

---

## 3. field 定義表

省略可 (optional) の表現は全 field 共通で「**attr/dataset の不存在 = 未知**」
(§5)。0 / NaN / 空文字への暗黙変換は行わない (例外: `region_name` の "" は
「ファイルが名前を持たない」という Phase A reader の読取事実であり sentinel ではない)。

| field | HDF5 path | 型 | shape | optional | 意味 |
|---|---|---|---|---|---|
| provenance version | `/provenance@schema_version` | str (vlen UTF-8) | scalar | 必須 (group があれば必ず) | provenance group の schema version。"major.minor" |
| source format | `/provenance@source_format` | str | scalar | 必須 | `scienta_pxt` / `igor_ibw` / `vamas` / `npl` / `ses_txt` / `text_columns` / `unknown` |
| source format version | `/provenance@source_format_version` | str | scalar | 省略可 | PXT wave v3 / IBW v5 の "3"/"5"、SES の "1.3.1"、VAMAS 識別行など。文字列固定 (数値化しない) |
| source region index | `/provenance@source_region_index` | int64 | scalar | 必須 | 元ファイル内の region 番号 (0-based)。SES はファイル内 `[Region N]` の N−1 |
| region name | `/provenance@region_name` | str | scalar | 必須 ("" 可) | 装置ファイルの region 名。"" = 記録なし |
| excitation energy | `/provenance@excitation_energy_eV` | float64 | scalar | 省略可 | hv [eV]。**省略 = ファイルに記録なし**。misc/fermienergy の 0.0 sentinel と独立に真値を保持 |
| pass energy | `/provenance@pass_energy_eV` | float64 | scalar | 省略可 | pass energy [eV]。省略 = 記録なし |
| intensity semantics | `/provenance@intensity_semantics` | str | scalar | 必須 | `raw_counts` / `count_rate` / `corrected_intensity` / `arbitrary` / `unknown` (Phase A `IntensitySemantics` と同一語彙) |
| intensity unit | `/provenance@intensity_unit` | str | scalar | 必須 | 単位文字列。判定不能は `unknown` |
| acquisition mode | `/provenance@acquisition_mode` | str | scalar | 省略可 (**明示値のみ**) | "Swept"/"Fixed" 等。Phase A metadata が空文字なら attr を作らない |
| datetime | `/provenance@datetime` | str (ISO 8601) | scalar | 省略可 (**明示 opt-in のみ**) | 測定日時。間接識別子になり得るため**既定非保存**。`ImportConfig.persist_datetime=True` のときのみ書く |

**B-1 で保存しない field (B-1 監査での補正)**: `n_sweeps` / `n_slices` /
`lens_mode` は標準 provenance へ**保存しない**。理由: Phase A metadata は
「ファイルに 1 と記録されていた」と「不明時の構造既定値 1」を区別できず、
NPL/VAMAS の `lens_mode="Angular"` は legacy 固定値であり取得事実と保証
できない。**将来これらを保存するには、Phase A `SpectrumMetadata` に
「ファイルから明示取得できたか」の marker (per-field provenance flag) を
追加する必要がある** (Phase A 側の将来課題として申し送り)。
`HDF5Provenance` には将来拡張用 field として残すが、不存在は `None` で表現し、
既定値 1 を復元しない。
| history incomplete | `/provenance@transform_history_incomplete` | bool | scalar | 省略可 (drop 発生時のみ) | transform 履歴に書込時 drop があったことの機械可読 flag (§6.2) |
| dropped records | `/provenance@dropped_transform_records` | int64 | scalar | 同上 | drop された record 数 |
| original shape | `/provenance@original_shape` | int64 | (ndim,) | 省略可 | 上流入力ファイル内の配列構造 (flatten 前)。`()` は省略 |
| dimension roles | `/provenance@dimension_roles` | vlen-str | (ndim,) | 省略可 | `energy` / `emission_angle` / `position` / `frame` / `sweep` / `time` / `unknown`。original_shape と同数。確定できない軸は `unknown` のまま保存 (書込側で角度へ再解釈しない) |
| transform history | `/provenance/transform_history` | vlen-str UTF-8 | (n_records,) | 省略可 (空履歴 = 省略) | 適用順の JSON record 列 (§6) |
| vendor metadata | `/provenance/vendor_metadata` | vlen-str UTF-8 | (1,) | 省略可 (opt-in のみ) | 許可リスト通過分の JSON object (§7) |
| vendor policy version | `/provenance/vendor_metadata@policy_version` | str | scalar | vendor dataset があれば必須 | 保存ポリシー version |
| vendor allowlist | `/provenance/vendor_metadata@allowlist` | vlen-str | (n_keys,) | 同上 | 書込時に適用した許可リストの記録 |

備考:

- attr 名に単位 (`_eV`) を含めるのは、misc の無単位 float 群 (fermienergy 等) で
  単位の曖昧さが監査対象になった反省から (監査 §8)。
- `dimension_roles` の語彙は Phase A `DIMENSION_ROLES` (`base_reader.py:41-50`)
  と一対一。将来語彙を増やす場合は provenance minor version を上げ、読側は
  未知語を `unknown` 扱いで受理する。
- 1 ファイル = 1 region (importer が multi-region を region ごとに分割する現行
  仕様。`importer.py:ensure_h5`) を前提に、provenance は file-level group とする。
  複数 region を 1 ファイルへ入れる将来案が出た場合は major version を上げる。

---

## 4. versioning 規則

3 つの version を分離する。

### 4.1 Toyomacro file schema version (採用: root attr)

- root attr `@toyomacro_schema_version` へ保存する (§12 判断 1)。値は
  `ToyomacroSchema.VERSION` をそのまま書く。コア dataset 群の layout を指す。
- **現状この定数は未永続 (§1.4) なので、書き始めること自体が新規動作**。
  additive 変更 (provenance 導入 + version 永続化開始) の節目として定数を
  `"1.1.0"` → **`"1.2.0"`** へ bump し、新規ファイルには
  `toyomacro_schema_version = "1.2.0"` を書く。
- **version 無記録の既存ファイルは legacy 1.1 系として正常に読む** (§4.4)。
  コア layout は 1.1 系から不変なので、読側にコード分岐は不要。
- 既知の穴: `StreamingFitparaWriter._copy_metadata_from_input` は root の
  オブジェクトだけをコピーし **root attrs をコピーしない**
  (`streaming_writer.py:387-399`)。fit 出力を `new_file` で作ると root attr が
  落ちる。実装で root attrs 継承を追加する (§10)。ただし crash-recovery
  marker (`_toyomacro_*` / `_voigtfit_*`) は per-file 状態なので継承しない。
  `/provenance` 側は group ごとコピーされるため影響しない。

### 4.2 provenance schema version

- `/provenance@schema_version`、初版 "1.0"。file schema と独立に進める。
- 互換規則: 同 major = additive 互換 (読側は未知 attr / 未知 dataset /
  未知 transform name を黙って無視してよい。ただし「無視した」ことを
  診断ログに残せる実装を推奨)。major 不一致 = 既知 field のみ best-effort 読取
  + warning。

### 4.3 uncertainty schema version (予約のみ)

- 名前空間 `/uncertainty` と、その attr `@schema_version` を**本書で予約する**。
  B-1 では group を作らず、仮 schema も定義しない (監査 §5 v1 の
  `UncertaintyResult` serialization が入るときに初版を切る)。
- 予約の含意: (a) `/provenance` に UQ 量を置かない、(b) 将来の UQ 書込が
  provenance version を bump させない、(c) §2.1 の repack float32 化問題を
  uncertainty 設計の必須検討事項として引き継ぐ。

### 4.4 version 無記録ファイルの扱い

| 状態 | 解釈 |
|---|---|
| root attr なし (既存全ファイル) | **legacy 1.1 系**として正常に読む。エラー・warning にしない |
| `/provenance` なし | provenance 未知。読側 API は None を返す (§8) |
| `/provenance` あり・`@schema_version` なし | 破損または手作業ファイル。warning を出し既知 field を best-effort 読取 |
| `@schema_version` major > 対応版 | warning + 既知 field のみ読取。例外にしない |

---

## 5. optional 値の表現

規則は 1 つ: **「未知 = その attr/dataset を書かない」**。

- 書込側: Python `None` の field は attr を作らない。`original_shape == ()`、
  `dimension_roles == ()`、`transforms == ()` も同様に省略。
- 読側: 不存在 → `None` (tuple field は `()`) へ復元。0 / NaN / "" への
  変換は行わない。これで Phase A の in-memory 表現
  (`excitation_energy: float | None`) と HDF5 表現が往復で一致する。
- 明示的に禁止する表現: `fermienergy = 0.0` 型の magic number、NaN sentinel
  (`json.dumps(allow_nan=False)` とも整合しない)、空文字 sentinel。
- 例外は `region_name` ("" は読取事実、§3) と `intensity_semantics` /
  `intensity_unit` (「わからない」を `unknown` という**値**で表現する設計を
  Phase A から継承。省略と区別され、省略 = 「provenance 書込者が semantics
  という概念自体を知らない旧版」を意味する)。

---

## 6. transform serialization

### 6.1 2 案の比較

**案 A: HDF5 展開型** — transform ごとに `transform_000_energy_scale_conversion/`
のような subgroup を作り、parameter を typed dataset/attr として展開する。

- 利点: h5dump / HDFView / MATLAB `h5read` で JSON parser なしに個別値を読める。
  数値が HDF5 native 型で残る。
- 欠点:
  - `ReaderTransform.parameters` は任意 key の JSON-safe dict (Phase A 設計)。
    展開型は **parameter 語彙が増えるたびに HDF5 schema 表面が増殖**し、
    versioning 対象が際限なく広がる。
  - 順序を name prefix (`000_`) で encode する必要があり、挿入・比較が煩雑。
  - 小さい object の多量生成で複製経路が遅くなり、repack の float64→float32
    降格 (§2.1) が数値 parameter を黙って劣化させる。
  - 古い reader に「未知 transform を無視させる」には group 単位 skip の
    規約が別途必要。

**案 B: JSON record 型 (推奨)** — `/provenance/transform_history` を
shape `(n_records,)` の vlen-str dataset とし、要素 i に record i の JSON object
を 1 つ入れる。

- 利点:
  - **順序 = 配列 index** で自明に保存される (reader 変換 → importer 変換の
    適用順)。
  - `ReaderTransform.to_dict()` (`base_reader.py`) と 1:1 対応。追加の型変換層が
    不要で、Phase A の deep-copy 済 JSON-safe 制約をそのまま流用できる。
  - schema evolution が record 単位で閉じる: 未知の `name` や未知 key を持つ
    record は「読めるが解釈しない」だけで済み、他 record の解釈に影響しない。
  - 文字列 dataset なので repack の dtype 降格を受けない (`repack_h5.py:205-206`
    の 1D コピーで素通り)。
  - MATLAB は `jsondecode(h5read(...))` の 1 行で読める。
- 欠点: HDF5 ツール単体で parameter を query できない。書込時に JSON-safe
  validation が必須。float の桁は JSON 文字列経由 (`json.dumps` は float64 を
  可逆表現するため実害なし)。

**推奨: 案 B。** 案 A の利点 (ツール可読性) は inspect 用ヘルパ
(`voigtfit/tools/inspect_h5.py` への表示追加など) で代替できるが、案 A の
schema 増殖は取り返しがつかない。

### 6.2 record 仕様

```json
{"name": "energy_scale_conversion",
 "parameters": {"from": "Kinetic", "to": "Binding",
                 "excitation_energy_eV": 1486.6,
                 "formula": "E_out = hv - E_in"},
 "source": "importer",
 "reason": "requested energy_scale=BE"}
```

- 必須 key: `name` (str)。準必須: `parameters` (object, 省略時 `{}`),
  `source` (str), `reason` (str) — `ReaderTransform.to_dict()` は常に 4 key を
  出すので実質必須。
- **読側規約: 未知の top-level key と未知の `name` は無視して続行する**
  (古い reader が新しい transform を含むファイルを読める条件)。record 単位の
  version key は置かない。語彙追加は provenance minor version で通知する。
- 書込順序: `RawSpectrumData.transforms` の tuple 順 (= 適用順) を保存。
- encoding: UTF-8。`json.dumps(..., ensure_ascii=False, allow_nan=False)`。
- validation: 書込前に record 単位で `json.dumps` を試行する。失敗した record は
  **偽の placeholder record へ置換しない** (§12 補正)。warning を出して drop し、
  `/provenance@transform_history_incomplete = True` と
  `@dropped_transform_records = <drop 数>` を書いて履歴が不完全であることを
  機械可読に明示する。読側 `HDF5Provenance` は両 flag をそのまま公開する。
  全 record が drop された場合は dataset を作らず flag のみが残る。

---

## 7. vendor metadata policy

**既定: 非永続。** Phase A の in-memory `vendor_metadata` は import 後も
HDF5 へ書かない。これが本設計の default であり、監査の privacy 指摘
(試料名・ユーザー名・パス) と 8/1 公開境界に対する最も安全な既定である。

opt-in 設計 (実装は B-1 実装フェーズ):

1. **opt-in 方式**: `ImportConfig` に
   `persist_vendor_metadata: bool = False` と
   `vendor_metadata_allowlist: tuple[str, ...] = ()` を追加。
   flag が True **かつ** allowlist が非空のときだけ書く。
   「全 key 保存」モードは B-1 では提供しない。
   **allowlist の初期値は空** (§12 判断 3) — flag を立てても、保存したい key を
   明示するまで何も書かれない。
2. **許可リスト方式**: 保存対象 = `vendor_metadata.keys() ∩ allowlist`
   (完全一致、大文字小文字区別)。適用した allowlist 自体を
   `@allowlist` attr に記録し、後から「何が落とされ得たか」を判別可能にする。
3. **`unparsed_notes` の扱い**: key=value 形式でない自由記述行 (Phase A で
   PXT の `vendor_metadata["unparsed_notes"]` に保持) は、**opt-in や
   allowlist 指定に関わらず B-1 では永続化しない**。key 単位でフィルタ
   できない自由テキストは許可リストの前提が成立しないため。
4. **privacy リスクの列挙** (allowlist 設計時の注意):
   ファイルパス、ユーザー名 (`User` 系 key)、試料識別子 (`Sample` 系 key)、
   測定日時、装置シリアル。なお **既存 schema の `misc/source_file` は
   入力ファイル名を既に保存しており** (`importer.py` misc 構築部)、ファイル名に
   試料名が含まれる運用では既存の露出点である。B-1 の変更対象外だが、
   公開データ配布時の注意点として明記する。
5. **保存ポリシー version**: `@policy_version = "1.0"`。allowlist の意味論
   (完全一致・非永続既定・unparsed 除外) を変えるときに bump。
6. **JSON-safe validation**: 値は str / int / float / bool / null と
   その list / dict のみ (Phase A `vendor_metadata` の deep-copy 制約と同じ)。
   非適合値は warning + その key を落とす (transform と異なり、vendor は
   opt-in の補助情報なので置換 record は作らない)。

---

## 8. 旧ファイルの読込規則

読側 API は素の dict でなく、**小さな型付き `HDF5Provenance` (frozen
dataclass)** を返す (§12 補正)。field は §3 の表に対応し、加えて
`transform_history_incomplete` / `dropped_transform_records` を公開する。
エントリポイントは `toyomacro.io.provenance.read_provenance()` と
`LazySpectrum.get_provenance()`。規則:

1. `/provenance` group がない → `None` を返す。warning なし
   (既存全ファイルが該当する正常系)。
2. group はあるが `@schema_version` がない → warning + best-effort 読取。
3. 各 attr の不存在 → その field は `None` / `()` (§5)。
4. `transform_history` の parse 不能 record → その record を診断情報付きで
   skip し、他 record は返す。
5. 未知 attr・未知 dataset・未知 transform name → 無視して続行 (§4.2, §6.2)。
6. どの経路でも **provenance の不備で配列読込 (specdata 等) を失敗させない**。
   provenance は常に「読めれば得、読めなくても本体は動く」付帯情報とする。

---

## 9. 互換性の根拠 (consumer 別)

「additive group だから安全」を仮定せず、consumer ごとに確認した。

| consumer | アクセス様式 | `/provenance` 追加時の挙動 | 根拠 |
|---|---|---|---|
| `LazySpectrum` | 固定 path (`specdata`)。misc のみ group 内 iterate | 影響なし | `io/readers/lazy_spectrum.py:113-141` (open)、434-458 (`get_misc` は misc 内のみ)、473-487 (`_get_dataset_info` は specdata のみ) |
| `HDF5Cache` | 固定 key (`specdata`/`specdata_uint16`/`misc`/`xytdata`) | 影響なし | `io/hdf5_cache.py:354-402` (`_scan_file`)、553-648 (`_get_file`、`misc["dim_shape"]` 等の名前指定) |
| 圧縮 (`compress_h5_file_streaming`) | 既知 4 dataset 以外の root object を group ごと copy + root attrs copy | **provenance・root attr とも保持** | `io/compression.py:727-738` |
| 圧縮 standard mode | 同上 Pass 1 + 名前指定 copy | 同上 | `compression.py:798-802` |
| repack (`repack_h5.repack_file`) | root/group attrs verbatim、dataset は再作成 | attr 群は保持。**dataset は 1D 文字列に限れば保持** (float scalar は §2.1 の理由で不採用) | `voigtfit/tools/repack_h5.py:269-270, 302-309` (attrs/group)、174-179 (float64→float32 降格)、89 + 202-204 (scalar chunk 問題)、205-206 (1D は素通り) |
| `StreamingFitparaWriter` (fit 出力の派生 h5) | 入力の非 fitpara root object を group ごと copy | **provenance group は保持**。root attrs は現状コピーされない (§4.1 の穴、実装時修正) | `io/writers/streaming_writer.py:387-399` |
| `h5io_fast` 変換 | dataset/group を名前指定 copy | 保持 | `voigtfit/h5io_fast.py:678-683` |
| recovery (`get_file_status`) | root を iterate するが `isinstance(obj, h5py.Dataset)` filter で group を除外 | 影響なし (provenance は datasets 一覧に出ないだけ) | `io/recovery.py:240-248` |
| `fit_dirty_map` | 固定 key (`fitpara`/`otherpara`/`fitpara_compressed`) | 影響なし | `io/fit_dirty_map.py:109-156` |
| voigtfit `integration` / `h5io` / `prefetch_pipeline` / `spectra_generator` | 固定 path。misc は key 存在確認後の名前指定 | 影響なし | `voigtfit/integration.py:447-468` (`'fermienergy' in misc`) ほか |
| fitting/depth reader (`depth/fitpara_reader.py`) | 固定 key の named reader 群 | 影響なし | `depth/fitpara_reader.py:57-288` |
| GUI/API (ローカル層) | `HDF5Cache`/`LazySpectrum` 経由 + `misc/source_file` 名前指定 | 影響なし | `gui/widgets/data_browser_panel.py:429-444`、`api/routes/browser.py:151-187` |
| `matlab_bridge` | Toyomacro schema ではない独自交換形式 (`/Y`,`/energy`) | 対象外 | `voigtfit/matlab_bridge.py:12-28` |

### 未知 root group を含むファイル・version 属性のない旧ファイル

- 上表のとおり Python consumer に root 全 iterate + 型仮定の経路はない
  (唯一の root iterate である recovery は Dataset filter 済)。
- また §1.3 のとおり圧縮ファイルで「コア以外の root dataset」は既に流通して
  おり、追加 object への耐性は実運用で担保されている。
- 旧ファイル (version 無記録) は §4.4 / §8 の規則で読む。

### compressed HDF5 での provenance 保持

- `compress_h5_file_streaming` の Pass 1 (`compression.py:727-738`) が
  `/provenance` を group ごと copy するため、**uint16+LZ4 圧縮ファイルでも
  provenance は無傷で保持される**。importer の `compress=True` 経路
  (`importer.py` step 8) は非圧縮ファイルへ書いた後にこの関数を通すので、
  書込フックは HDF5Writer 側 1 箇所で足りる。
- 実装時に「圧縮前後で provenance が bit 同一」の round-trip テストを置く (§11)。

### 従来の Toyomacro MATLAB ローカル HDF5 consumer (未確認事項)

ここで言う互換性は**メーカー形式との互換性ではなく**、同じローカル HDF5
schema を読む従来の MATLAB 版 Toyomacro / DepthProfiler consumer との互換性
である (§0)。その MATLAB 側読込コードは本リポジトリに存在せず、
**今回検証できていない**。`h5read` の明示 path 読込であれば未知 group は
無害のはずだが、以下を確認項目とする: provenance 付きファイルを MATLAB 版
Toyomacro で開き、(a) 既存読込が全て通る、(b) `Drlt`/`Dtrn` 書き戻しが
provenance を消さない、の 2 点。

---

## 10. 実装時の変更予定ファイル

| ファイル | 変更 |
|---|---|
| `src/toyomacro/io/schema.py` | `VERSION = "1.2.0"` bump、`ATTR_SCHEMA_VERSION`、`PATH_PROVENANCE`、`PROVENANCE_SCHEMA_VERSION = "1.0"` |
| `src/toyomacro/io/provenance.py` (新規) | `HDF5Provenance` (frozen dataclass)、`write_provenance()` / `read_provenance()`、`ProvenanceWarning`。attr 名の単一定義点 |
| `src/toyomacro/io/writers/hdf5_writer.py` | `create()` で root attr 書込、`write_provenance()` delegate 追加 |
| `src/toyomacro/io/importer.py` | `import_file` で `write_provenance` 呼出 (既定 ON)、`ImportConfig` へ `persist_datetime` / `persist_vendor_metadata` / `vendor_metadata_allowlist` 追加 |
| `src/toyomacro/io/readers/lazy_spectrum.py` | `get_provenance() -> HDF5Provenance \| None` (読込規則 §8) |
| `src/toyomacro/io/writers/streaming_writer.py` | `_copy_metadata_from_input` に root attrs 継承追加 (§4.1、recovery marker 除外) |
| `src/toyomacro/io/__init__.py` | `HDF5Provenance` / `read_provenance` / `write_provenance` / `ProvenanceWarning` export |
| `tests/test_provenance_schema.py` (新規) | §11 のテスト |
| `tests/test_readers_synthetic.py` | importer E2E に provenance 検証を追加 |
| `docs/` | 本書を実装結果で更新 |

reader 群 (`pxt_reader` 等)・core dataset・misc・GUI/Web は変更しない。

---

## 11. 必要なテスト一覧

1. **書込 round-trip**: 合成 PXT → import → `/provenance` 全 field が
   metadata と一致 (attr 型・値・順序)。
2. **省略規則**: hv/PE 不明の合成ファイル → 該当 attr が**存在しない**こと
   (0.0/NaN が書かれていないこと) + 読側で None 復元。
3. **transform 履歴**: KE→BE + flatten を含む import → record 数・順序・
   parameters が `ImportResult.metadata["transforms"]` と一致。空履歴で
   dataset が存在しないこと。
4. **未知 transform 耐性**: 手書きで未知 name の record を含むファイル →
   読側が例外なく既知 record を返す。
5. **圧縮保持**: `compress_h5_file_streaming` 前後で provenance の
   **意味的同一性** (`read_provenance` の結果が等価) を検証 (full / standard
   両モード)。ファイル全体の bit 一致は要求しない (§12 補正)。
6. **repack 保持**: `repack_file` 前後で同じく `read_provenance` 結果の
   意味的同一性を検証 (1D 文字列 dataset が通ること)。
7. **StreamingFitparaWriter 派生**: 入力 h5 の provenance が `new_file` /
   `temp_then_move` 出力へ引き継がれること + root attr コピー修正の検証。
8. **旧ファイル**: `/provenance` なしファイルで `get_provenance() is None`、
   既存 LazySpectrum/HDF5Cache/recovery テストが全て green のまま。
9. **version 規則**: `@schema_version` 欠落 → warning + best-effort。
   major 不一致 → warning + 既知 field のみ。
10. **vendor opt-in**: 既定で書かれない / opt-in + allowlist で交差分のみ /
    `unparsed_notes` が常に落ちる / `@policy_version`・`@allowlist` が付く。
11. **JSON-safe validation**: 非 serializable parameter を含む transform →
    import は成功、warning、当該 record は drop、
    `@transform_history_incomplete=True` + `@dropped_transform_records` が
    書かれ、placeholder record が**存在しない**こと。
11b. **datetime opt-in**: 既定 import で `@datetime` が存在しない /
    `persist_datetime=True` で ISO 8601 文字列が書かれる。
12. **既存スイート回帰**: フル pytest + ruff (Phase A と同基準)。

---

## 12. 設計判断の結果 (2026-07-18 承認済み)

| # | 論点 | 判断 |
|---|---|---|
| 1 | file schema version の置き場所 | **root attr**。新規ファイルは `"1.2.0"`、version 無記録の既存ファイルは legacy 1.1 系として正常に読む |
| 2 | provenance 書込の既定 | **ON** |
| 3 | vendor allowlist 初期値 | **空** (flag を立てても、key を明示するまで何も書かれない) |
| 4 | 取得条件の追加保存 | 当初判断は `lens_mode` / `acquisition_mode` / `n_sweeps` / `n_slices` を標準 attr で保存。**B-1 監査で補正**: `n_sweeps` / `n_slices` / `lens_mode` は取得事実と保証できないため保存対象から除外 (§3)。`acquisition_mode` は明示値がある場合のみ保存。**`datetime` は privacy (間接識別子) のため既定非保存・明示 opt-in** (`persist_datetime`) |
| 5 | uncertainty 名前空間 | `/uncertainty` を**文書上のみ**予約。B-1 では group を作らない |
| 6 | 合成データ生成側の provenance | B-1 対象外のまま (将来の generator 改修時に判断) |
| 7 | repack の float64→float32 自動降格 (`repack_h5.py:174-179`) | B-2 以降の uncertainty schema 設計での必須検討事項として申し送り (covariance を float64 dataset で持つ場合に黙って劣化するため) |

承認時の設計補正 (反映済み):

- 読込 API は素の dict でなく型付き `HDF5Provenance` を返す (§8)。
- JSON 化できない transform は placeholder record へ置換しない。warning +
  `@transform_history_incomplete` + `@dropped_transform_records` で履歴欠落を
  明示する (§6.2)。
- 圧縮/repack テストはファイル全体の bit 一致ではなく、provenance 内容の
  意味的同一性 (読側 API の比較) を検証する (§11)。
- 形式の位置づけ (§0): Toyomacro HDF5 はローカル形式であり、`source_format`
  はメーカー公式 schema への準拠認証を意味しない。互換性確認の対象は
  「従来の Toyomacro MATLAB ローカル HDF5 consumer」。

B-1 監査での追加補正 (コミット前レビュー、反映済み):

- `read_provenance()` は既知 field の型・shape・JSON 不正で例外を出さず、
  `ProvenanceWarning` + 安全な既定値の best-effort 読込とする (int/float attr
  の文字列・配列・非有限値、`original_shape`/`dimension_roles` の型不正、
  `transform_history`/`vendor_metadata` の非 dataset・scalar・空・不正 dtype、
  `/provenance` が group でない場合の warning + None)。`MemoryError` 等の
  システム例外までは握り潰さない。
- `n_sweeps` / `n_slices` / `lens_mode` の標準保存を取りやめ (§3、上表 4)。
- `LazySpectrum.get_provenance()` に `HDF5Provenance | None` の戻り値注釈。
- 背景資料 (UQ 監査ノート) は未追跡の内部資料であり、本書から必須依存として
  参照しない (冒頭の背景資料注記)。

---

## 13. Phase B-1 に含めない項目

- `/uncertainty` group の実装・仮 schema (名前空間予約のみ、§4.3)
- raw count dataset、dead-time 補正、detector covariance、Poisson 観測モデル
  (§5 の境界: B-1 は `intensity_semantics`/`intensity_unit`/`unknown` まで。
  以降は監査 v1 (raw count 保持・観測モデル一元化) と同期して別フェーズで判断)
- plugin / entry point / xpsuncertainty adapter
- 既存 dataset・misc の変更 (fermienergy sentinel の是正を含む — 現状維持)
- vendor 自由記述 (`unparsed_notes`) の永続化
- MATLAB 側 writer / reader の変更
- GUI / Web での provenance 表示
- reader (`pxt_reader` 等) の変更 — Phase A の出力をそのまま永続化する

---

## 14. 実装結果 (2026-07-18、B-1 監査補正込み)

§10 の予定どおり実装し、コミット前の B-1 監査補正 (§12 末尾) を反映した。

- `io/schema.py`: `VERSION = "1.2.0"`、`ATTR_SCHEMA_VERSION`、
  `PATH_PROVENANCE`、`PROVENANCE_SCHEMA_VERSION = "1.0"`。
- `io/provenance.py` (新規): `HDF5Provenance` (frozen dataclass)、
  `write_provenance()` / `read_provenance()`、`ProvenanceWarning`。
  attr 名の定義はこのモジュールに集約。読側は既知 field の malformed
  metadata に対し warning + 安全既定値の best-effort (§12 補正)。
  `n_sweeps` / `n_slices` / `lens_mode` は書かず、読側でも既定値 1 を
  復元しない (不存在 = `None`)。`acquisition_mode` は明示値のみ書く。
- `io/writers/hdf5_writer.py`: `create()` が root attr を stamp、
  `write_provenance()` delegate。
- `io/importer.py`: provenance 書込は既定 ON。`ImportConfig` に
  `persist_datetime` / `persist_vendor_metadata` / `vendor_metadata_allowlist`
  (初期値空)。
- `io/readers/lazy_spectrum.py`: `get_provenance()`。
- `io/writers/streaming_writer.py`: root attrs 継承 (recovery marker
  `_toyomacro_*` / `_voigtfit_*` は除外)。
- `io/__init__.py`: `HDF5Provenance` 等の export。

テスト (2026-07-18 時点):

- 新規 `tests/test_provenance_schema.py` 23 件 + `test_readers_synthetic.py`
  へ E2E 1 件追加 (§11 の 1〜11b をカバー)。圧縮 (full/standard)・repack・
  StreamingFitparaWriter 派生の 3 経路すべてで `read_provenance` の意味的
  同一性を実測確認。
- フルスイート 1481 passed / 93 skipped / 3 failed — 失敗 3 件は変更範囲外の
  `test_compression.py` 性能閾値 (既知の全体負荷フレーク、単独実行 22 passed)。
- `ruff check .` clean。

未確認のまま残る事項: 従来の Toyomacro MATLAB ローカル HDF5 consumer での
読込確認 (§9 の 2 点)。

---

## 付記: 監査文書からの主な修正点

1. 監査 §5 v1-1 は「`UncertaintyResult` + HDF5 schema 追加」を一括で
   挙げていたが、本設計では **provenance (取得事実) と uncertainty (推定結果)
   の名前空間・version を分離**し、B-1 は前者のみとした。
2. 監査は複製経路 (圧縮・repack・streaming 派生) の挙動を扱っていなかった。
   本調査で判明した repack の float64 降格・scalar chunk 問題を根拠に、
   scalar field を dataset でなく **attr** に置く layout へ修正した (§2.1)。
3. streaming writer の root attrs 非コピーという既存の穴 (§4.1) を新規に
   特定し、file version の置き場所判断 (§12-1) の材料にした。
