import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from neural_style_transfer.constants import DEVICE
from neural_style_transfer.model.feature_extractor import FeatureExtractor
from neural_style_transfer.trainer.trainer import Trainer
from neural_style_transfer.utils import get_image, denormalize_img

# Preprocessing
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    transforms.Resize((200, 200))
                                    ])

content_img = get_image('images/image.png', img_transform)
style_img = get_image('images/style.jpg', img_transform)
generated_img = content_img.clone()  # or nn.Parameter(torch.FloatTensor(content_img.size()))
generated_img.requires_grad = True


optimizer = torch.optim.Adam([generated_img], lr=0.003, betas=(0.5, 0.999))
encoder = FeatureExtractor().to(DEVICE)

trainer = Trainer(optimizer)

trainer.train(model=encoder, epochs=500, content_img=content_img, style_img=style_img, generated_img=generated_img)
for p in encoder.parameters():
    p.requires_grad = False

inp = generated_img.detach().cpu().squeeze()
inp = denormalize_img(inp)
plt.imshow(inp)
