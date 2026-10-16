import os, cv2, numpy as np, torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATASET_PATH = "dataset"
IMG_SIZE     = 224
CLASSES      = ["glioma", "meningioma", "pituitary", "notumor"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Dropout(0.4), nn.Linear(256, 4)
)
model.load_state_dict(torch.load("resnet_brain_tumor.pth", map_location=DEVICE))
model.eval().to(DEVICE)

features_out = []
def hook(module, input, output):
    features_out.append(output.squeeze().cpu().numpy())
model.avgpool.register_forward_hook(hook)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

all_features, all_labels = [], []
print("Extracting features...")
for label_idx, class_name in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, "Testing", class_name)
    for fname in os.listdir(folder)[:100]:   # 100 per class = fast
        img = cv2.imread(os.path.join(folder, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        features_out.clear()
        with torch.no_grad():
            model(tensor)
        all_features.append(features_out[0])
        all_labels.append(label_idx)

print("Running t-SNE (this takes ~1 min)...")
features_2d = TSNE(n_components=2, perplexity=30,
                   random_state=42).fit_transform(np.array(all_features))

colors = ['#378ADD','#1D9E75','#BA7517','#993556']
fig, ax = plt.subplots(figsize=(10, 8))
for i, name in enumerate(CLASSES):
    mask = np.array(all_labels) == i
    ax.scatter(features_2d[mask,0], features_2d[mask,1],
               c=colors[i], label=name, alpha=0.7, s=30)

ax.set_title("t-SNE — Decision Boundary (ResNet18 Feature Space)", fontsize=14)
ax.legend(fontsize=12)
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.savefig("tsne_boundary.png", dpi=150)
plt.show()
print("Saved as tsne_boundary.png")