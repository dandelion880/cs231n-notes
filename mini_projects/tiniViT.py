import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

# 第一步，将输入图像划分为固定大小的patches
class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 32, 
                 patch_size = 8, 
                 in_channels = 3, 
                 embed_dim = 64
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用卷积完成patch切分，并线性映射到embed_dim维度
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.proj(x) # x: [B, 64, 4, 4]
        x = x.flatten(2) # x: [B, 64, 16]
        x = x.transpose(1, 2) # x: [B, 16, 64]
        return x
    
# 第二步，Transformer Encoder
class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim = 64, 
                 num_heads = 4, 
                 mlp_ratio = 4.0, 
                 dropout = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads, 
            dropout = dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, 16, 64]
        # self-attention
        x_norm = self.norm1(x) # x_norm: [B, 16, 64]
        attn_output, attn_weights = self.attn(
            x_norm, 
            x_norm, 
            x_norm,
            need_weights = False
        ) # attn_output: [B, 16, 64]

        x = x + attn_output # 残差连接
        x = x + self.mlp(self.norm2(x)) # MLP

        return x
    
# 第三步，ViT模型
class TiniViT(nn.Module):
    def __init__(self,
                 img_size = 32,
                 patch_size = 8,
                 in_channels = 3,
                 embed_dim = 64,
                 num_classes = 10,
                 depth = 4,
                 num_heads = 4,
                 dropout = 0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size = img_size,
            patch_size = patch_size,
            in_channels = in_channels,
            embed_dim = embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 分类token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # 位置编码
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = 4.0,
                dropout = dropout
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        B = x.size(0)
        x = self.patch_embed(x) # x: [B, 16, 64]

        cls_tokens = self.cls_token.expand(B, -1, -1) # cls_tokens: [B, 1, 64]
        x = torch.cat((cls_tokens, x), dim=1) # x: [B, 17, 64]
        x = x + self.pos_embed # 添加位置编码
        x = self.dropout(x)

        x = self.blocks(x) # x: [B, 17, 64]
        x = self.norm(x) # x: [B, 17, 64]
        cls_output = x[:, 0] # 分类token的输出: [B, 64]
        logits = self.head(cls_output) # logits: [B, num_classes]

        return logits
    
# 第四步，训练与评估
def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(
        model,
        test_loader,
        criterion,
        device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

# 第五步，主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "dataset"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = TiniViT(
        img_size=32,
        patch_size=8,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
if __name__ == "__main__":
    main()