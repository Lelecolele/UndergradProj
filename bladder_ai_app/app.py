import gradio as gr
import torch
import timm
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import zipfile
import tempfile

# --- Device setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model definitions ---
class TumorClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0, global_pool="avg")
        in_features = self.backbone.num_features
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 1)
        )
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class UNetPP(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None
        )
    def forward(self, x):
        return self.model(x)

class ConvNeXtClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.6, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='')
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.backbone.num_features, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# --- Load models ---
classification_model = TumorClassifier().to(device)
classification_model.load_state_dict(torch.load('models/classification_model.pth', map_location=device))
classification_model.eval()

try:
    segmentation_model = UNetPP().to(device)
    segmentation_model.load_state_dict(torch.load('models/segmentation_model.pth', map_location=device))
    segmentation_model.eval()
except Exception as e:
    print(f"Segmentation model loading failed: {e}")
    segmentation_model = None

grading_model = ConvNeXtClassifier().to(device)
grading_model.load_state_dict(torch.load('models/grading_model.pth', map_location=device))
grading_model.eval()

# --- Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Language dictionary ---
language_texts = {
    "中文": {
        "title": "膀胱癌AI诊断平台",
        "upload_label": "上传图片或压缩包",
        "predict_button": "开始预测",
        "result_label": "预测结果"
    },
    "English": {
        "title": "Bladder Cancer AI Diagnostic Platform",
        "upload_label": "Upload Images or Zip",
        "predict_button": "Start Prediction",
        "result_label": "Prediction Results"
    }
}

# --- Single image prediction ---
def predict_single(img_pil, lang):
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_out = classification_model(img_tensor)
        prob = torch.sigmoid(cls_out).item()
        label = "肿瘤" if lang == "中文" and prob > 0.5 else "正常" if lang == "中文" else "Tumor" if prob > 0.5 else "Normal"

        if prob > 0.5 and segmentation_model is not None:
            seg_out = segmentation_model(img_tensor)
            mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_binary).convert("RGB").resize(img_pil.size)
        else:
            mask_img = Image.new('RGB', img_pil.size)

        if prob > 0.5:
            grade_out = grading_model(img_tensor)
            grade_prob = torch.softmax(grade_out, dim=1)
            grade_label = "高级别" if lang == "中文" and torch.argmax(grade_prob).item() == 1 else "低级别" if lang == "中文" else "High Grade" if torch.argmax(grade_prob).item() == 1 else "Low Grade"
        else:
            grade_label = "-" if lang == "中文" else "-"

    overlay = np.array(img_pil) * 0.7 + np.array(mask_img) * 0.3
    overlay = overlay.astype(np.uint8)

    tumor_prob_percent = prob * 100
    normal_prob_percent = (1 - prob) * 100

    if lang == "中文":
        description = f"分类结果: {label}\n肿瘤概率: {tumor_prob_percent:.1f}%\n正常概率: {normal_prob_percent:.1f}%\n分级: {grade_label}"
    else:
        description = f"Classification: {label}\nTumor probability: {tumor_prob_percent:.1f}%\nNormal probability: {normal_prob_percent:.1f}%\nGrading: {grade_label}"

    return (overlay, description)

# --- Main predict function ---
def predict(files, lang):
    if not files:
        return []
    results = []
    for file in files:
        if file.name.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            for img_name in os.listdir(temp_dir):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(temp_dir, img_name)
                    img_pil = Image.open(img_path).convert('RGB')
                    results.append(predict_single(img_pil, lang))
        else:
            img_pil = Image.open(file).convert('RGB')
            results.append(predict_single(img_pil, lang))
    return results

# --- Language switch function ---
def switch_language(selected_lang):
    texts = language_texts[selected_lang]
    return (
        gr.update(value=f"<h1 style='font-weight:bold; font-size:32px;'>{texts['title']}</h1>"),
        gr.update(label=texts["upload_label"]),
        gr.update(value=texts["predict_button"]),
        gr.update(label=texts["result_label"])
    )

# --- Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        title = gr.Markdown("<h1 style='font-weight:bold; font-size:32px;'>膀胱癌AI诊断平台</h1>")
        lang_choice = gr.Radio(["中文", "English"], value="中文", label="", interactive=True, container=False)

    with gr.Row():
        with gr.Column():
            upload = gr.Files(label="上传图片或压缩包", file_types=["image", ".zip"], interactive=True)
            predict_btn = gr.Button("开始预测")
        with gr.Column():
            output_gallery = gr.Gallery(label="预测结果", columns=2, object_fit="contain")

    predict_btn.click(
        fn=predict,
        inputs=[upload, lang_choice],
        outputs=output_gallery
    )

    lang_choice.change(
        fn=switch_language,
        inputs=lang_choice,
        outputs=[title, upload, predict_btn, output_gallery]
    )

# --- Launch ---
demo.launch(share=True)
