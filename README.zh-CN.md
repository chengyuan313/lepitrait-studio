# LepiTrait Studio

LepiTrait Studio 是面向标准化蝴蝶与蛾类标本照片的本地科研工作台。项目以 LEPY 的形态和颜色性状输出为核心参考，所有自动结果都保留质控状态、方法版本和人工复核入口。

## 当前版本

- 标准标本图像质量检查；
- 浅色统一背景下的透明分割基线；
- 基于比例尺的像素/毫米形态指标；
- 校准图像的 CIELAB 颜色统计；
- 标本整图标签区域裁切、独立标签近照上传与本地 OCR；
- 馆藏号、采集日期、模式状态和学名的保守解析与人工校正；
- LEPY 与 BioCLIP 物种识别适配层；
- JSON/CSV 导出；
- Streamlit GUI。

内置分割算法仅用于开发和界面联调，不能替代经过验证的 LEPY 模型。正式研究需要接入固定版本的 LEPY，并使用人工测量数据评估误差。

## 启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

标签 OCR 需要本机安装 Tesseract。Windows PowerShell 可执行：

```powershell
winget install --id UB-Mannheim.TesseractOCR
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

重新启动应用后，在 `Label record` 页面检查建议裁切范围并点击 `Run label OCR`。历史手写标签必须人工复核，OCR 文本不会进入物种识别模型。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 科学边界

物种识别只接收统一方式拍摄的标本照片。分类器输入必须移除标签、比例尺和色卡，避免模型通过文字、馆藏编号或拍摄批次识别物种。自然环境中的活体照片不属于本项目范围。

数据收集前请先阅读：

- `docs/imaging_sop.md`：标准化拍摄规范；
- `docs/data_dictionary.md`：数据字段；
- `docs/ocr_benchmark.md`：首批真实图片 OCR 验证结果；
- `training/README.md`：物种分类训练方案。
