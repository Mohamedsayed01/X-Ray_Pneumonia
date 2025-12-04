from torchvision import transforms,models
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)

model.load_state_dict(torch.load(r'C:\Users\windows10\Documents\GitHub\X-Ray_Pneumonia\models\xray_resnet18.pth', map_location=device
))
model = model.to(device )
model.eval()
def infer (img):
    image_model1 = Image.open(img_path).convert('L')
    img_trans = transforms.ToTensor()(image_model1).unsqueeze(0).to(device)
    pred = model(img_trans)
    pred_class = torch.argmax(pred).item()
    final_pred = le.inverse_transform([pred_class]) # le is a LabelEncoder instance
    return final_pred

img_path = r'C:\Users\windows10\Documents\GitHub\X-Ray_Pneumonia\Dataset\images\person100_bacteria_2931.jpeg'
img = Image.open(img_path)
infer(img)