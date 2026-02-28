"""
AI模型安全实战 - 对抗攻击代码
环境要求: Python 3.9
依赖安装: pip install torch torchvision matplotlib numpy pillow
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelSecurityAttack:
    """
    模型安全攻击类 - 包含FGSM和PGD攻击方法
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.transform = None
        self.classes = None

    def load_model(self):
        """
        加载预训练模型并分析结构
        """
        print("=" * 50)
        print("1. 加载预训练模型")
        print("=" * 50)

        # 加载预训练的ResNet18模型
        self.model = resnet18(pretrained=True)
        self.model.eval()  # 设置为评估模式
        self.model.to(self.device)

        # 分析模型结构
        print("\n模型结构分析:")
        print(f"模型类型: {type(self.model).__name__}")
        print(f"总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # 打印主要层
        print("\n主要网络层:")
        layers = list(self.model.children())
        for i, layer in enumerate(layers[:5]):  # 只显示前5层
            print(f"  Layer {i}: {type(layer).__name__}")

        # 定义数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ImageNet类别标签
        self.load_imagenet_classes()

        return self.model

    def load_imagenet_classes(self):
        """
        加载ImageNet类别标签
        """
        # 简化版类别标签（实际使用时可以加载完整列表）
        self.classes = [str(i) for i in range(1000)]
        try:
            # 尝试下载ImageNet类别标签
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            import urllib.request
            with urllib.request.urlopen(url) as f:
                self.classes = [line.decode('utf-8').strip() for line in f.readlines()]
        except:
            print("警告: 无法下载ImageNet类别标签，使用数字索引")

    def preprocess_image(self, image_path):
        """
        预处理输入图像
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 保存原始图像用于可视化
        original_image = image.copy()

        # 预处理
        input_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        input_tensor = input_tensor.to(self.device)

        return input_tensor, original_image

    def predict(self, input_tensor):
        """
        模型预测
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence, probabilities

    def fgsm_attack(self, image_tensor, epsilon, targeted=False, target_class=None):
        """
        FGSM攻击方法

        参数:
            image_tensor: 输入图像张量
            epsilon: 扰动强度
            targeted: 是否为目标攻击
            target_class: 目标类别（目标攻击时需要）

        返回:
            adversarial_image: 对抗样本
        """
        print("\n" + "=" * 50)
        print("2. 执行FGSM攻击")
        print("=" * 50)

        # 克隆原始图像并设置requires_grad
        adversarial_image = image_tensor.clone().detach().requires_grad_(True)

        # 获取原始预测
        original_pred, original_conf, _ = self.predict(image_tensor)
        print(f"原始预测: 类别 {original_pred} ({self.classes[original_pred]})")
        print(f"原始置信度: {original_conf:.4f}")

        # 前向传播
        outputs = self.model(adversarial_image)

        if targeted:
            # 目标攻击：最小化目标类别的损失
            target = torch.tensor([target_class]).to(self.device)
            loss = nn.CrossEntropyLoss()(outputs, target)
        else:
            # 非目标攻击：最大化真实类别的损失
            original_label = torch.tensor([original_pred]).to(self.device)
            loss = -nn.CrossEntropyLoss()(outputs, original_label)

        # 反向传播获取梯度
        self.model.zero_grad()
        loss.backward()

        # 获取梯度符号
        data_grad = adversarial_image.grad.data
        sign_data_grad = data_grad.sign()

        # 生成对抗样本
        if targeted:
            adversarial_image = adversarial_image - epsilon * sign_data_grad
        else:
            adversarial_image = adversarial_image + epsilon * sign_data_grad

        # 裁剪到有效范围
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        # 预测对抗样本
        adv_pred, adv_conf, _ = self.predict(adversarial_image)
        print(f"\nFGSM攻击结果 (epsilon={epsilon}):")
        print(f"对抗样本预测: 类别 {adv_pred} ({self.classes[adv_pred]})")
        print(f"对抗样本置信度: {adv_conf:.4f}")
        print(f"攻击成功: {adv_pred != original_pred}")

        return adversarial_image.detach()

    def pgd_attack(self, image_tensor, epsilon, alpha, num_iter, targeted=False, target_class=None):
        """
        PGD攻击方法（迭代式）

        参数:
            image_tensor: 输入图像张量
            epsilon: 最大扰动强度
            alpha: 步长
            num_iter: 迭代次数
            targeted: 是否为目标攻击
            target_class: 目标类别（目标攻击时需要）

        返回:
            adversarial_image: 对抗样本
        """
        print("\n" + "=" * 50)
        print("3. 执行PGD攻击")
        print("=" * 50)

        # 获取原始预测
        original_pred, original_conf, _ = self.predict(image_tensor)
        print(f"原始预测: 类别 {original_pred} ({self.classes[original_pred]})")
        print(f"原始置信度: {original_conf:.4f}")

        # 初始化对抗样本
        adversarial_image = image_tensor.clone().detach().requires_grad_(True)

        for i in range(num_iter):
            # 前向传播
            outputs = self.model(adversarial_image)

            if targeted:
                # 目标攻击
                target = torch.tensor([target_class]).to(self.device)
                loss = nn.CrossEntropyLoss()(outputs, target)
            else:
                # 非目标攻击
                original_label = torch.tensor([original_pred]).to(self.device)
                loss = -nn.CrossEntropyLoss()(outputs, original_label)

            # 反向传播
            self.model.zero_grad()
            loss.backward()

            # 获取梯度符号
            data_grad = adversarial_image.grad.data
            sign_data_grad = data_grad.sign()

            # 更新对抗样本
            with torch.no_grad():
                if targeted:
                    adversarial_image = adversarial_image - alpha * sign_data_grad
                else:
                    adversarial_image = adversarial_image + alpha * sign_data_grad

                # 投影到epsilon球内
                perturbation = adversarial_image - image_tensor
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                adversarial_image = torch.clamp(image_tensor + perturbation, 0, 1)

            adversarial_image.requires_grad_(True)

            # 每5次迭代打印一次进度
            if (i + 1) % 5 == 0:
                current_pred, current_conf, _ = self.predict(adversarial_image)
                print(f"迭代 {i + 1}/{num_iter}: 预测类别 {current_pred}, 置信度 {current_conf:.4f}")

        # 最终预测
        adv_pred, adv_conf, _ = self.predict(adversarial_image)
        print(f"\nPGD攻击结果 (epsilon={epsilon}, alpha={alpha}, iter={num_iter}):")
        print(f"对抗样本预测: 类别 {adv_pred} ({self.classes[adv_pred]})")
        print(f"对抗样本置信度: {adv_conf:.4f}")
        print(f"攻击成功: {adv_pred != original_pred}")

        return adversarial_image.detach()

    def tensor_to_image(self, tensor):
        """
        将张量转换回图像
        """
        # 反标准化
        tensor = tensor.squeeze(0).cpu().detach()

        # 如果张量已经标准化，需要反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean

        # 转换为numpy数组并调整维度
        image = tensor.numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        return Image.fromarray(image)

    def visualize_results(self, original_tensor, fgsm_tensor, pgd_tensor, original_image, save_path):
        """
        可视化结果
        """
        # 获取预测结果
        original_pred, original_conf, _ = self.predict(original_tensor)
        fgsm_pred, fgsm_conf, _ = self.predict(fgsm_tensor)
        pgd_pred, pgd_conf, _ = self.predict(pgd_tensor)

        # 计算扰动
        fgsm_perturbation = (fgsm_tensor - original_tensor).abs().mean().item() * 1000
        pgd_perturbation = (pgd_tensor - original_tensor).abs().mean().item() * 1000

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'原始图像\n预测: {self.classes[original_pred]}\n置信度: {original_conf:.3f}')
        axes[0, 0].axis('off')

        fgsm_image = self.tensor_to_image(fgsm_tensor)
        axes[0, 1].imshow(fgsm_image)
        axes[0, 1].set_title(f'FGSM对抗样本\n预测: {self.classes[fgsm_pred]}\n置信度: {fgsm_conf:.3f}')
        axes[0, 1].axis('off')

        pgd_image = self.tensor_to_image(pgd_tensor)
        axes[0, 2].imshow(pgd_image)
        axes[0, 2].set_title(f'PGD对抗样本\n预测: {self.classes[pgd_pred]}\n置信度: {pgd_conf:.3f}')
        axes[0, 2].axis('off')

        # 第二行：扰动可视化
        # 计算扰动图像
        fgsm_perturb = torch.abs(fgsm_tensor - original_tensor).squeeze(0).cpu()
        fgsm_perturb = fgsm_perturb.permute(1, 2, 0).numpy()
        fgsm_perturb = np.mean(fgsm_perturb, axis=2) * 10  # 增强可视化效果

        pgd_perturb = torch.abs(pgd_tensor - original_tensor).squeeze(0).cpu()
        pgd_perturb = pgd_perturb.permute(1, 2, 0).numpy()
        pgd_perturb = np.mean(pgd_perturb, axis=2) * 10

        axes[1, 0].imshow(np.zeros_like(fgsm_perturb), cmap='gray')
        axes[1, 0].set_title('无扰动')
        axes[1, 0].axis('off')

        im1 = axes[1, 1].imshow(fgsm_perturb, cmap='hot')
        axes[1, 1].set_title(f'FGSM扰动 (强度: {fgsm_perturbation:.2f})')
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1])

        im2 = axes[1, 2].imshow(pgd_perturb, cmap='hot')
        axes[1, 2].set_title(f'PGD扰动 (强度: {pgd_perturbation:.2f})')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2])

        plt.suptitle('对抗攻击结果对比', fontsize=16)
        plt.tight_layout()

        # 保存图像
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n结果已保存到: {save_path}")

        # 单独保存对抗样本
        fgsm_image.save('fgsm_adversarial.jpg')
        pgd_image.save('pgd_adversarial.jpg')
        print("对抗样本已保存: fgsm_adversarial.jpg, pgd_adversarial.jpg")

    def calculate_metrics(self, original_tensor, adversarial_tensor):
        """
        计算攻击效果指标
        """
        # L2距离
        l2_distance = torch.norm(original_tensor - adversarial_tensor, p=2).item()

        # 无穷范数
        linf_distance = torch.norm(original_tensor - adversarial_tensor, p=float('inf')).item()

        # PSNR
        mse = torch.mean((original_tensor - adversarial_tensor) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        return {
            'L2距离': l2_distance,
            'Linf距离': linf_distance,
            'MSE': mse,
            'PSNR': psnr
        }


def main():
    # 主函数
    print("=" * 60)
    print("AI模型安全实战 - 对抗攻击演示")
    print("=" * 60)

    # 初始化攻击类
    attacker = ModelSecurityAttack()

    # 1. 加载模型
    model = attacker.load_model()

    # 2. 准备测试图像
    print("\n" + "=" * 50)
    print("准备测试图像")
    print("=" * 50)

    # 创建测试图像（如果不存在）
    test_image_path = 'test_image.jpg'
    if not os.path.exists(test_image_path):
        # 创建一个简单的测试图像（彩色渐变）
        image = Image.new('RGB', (224, 224), color='lightblue')
        # 添加一些图案使其更有意义
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 174, 174], fill='darkblue')
        draw.ellipse([80, 80, 144, 144], fill='yellow')
        image.save(test_image_path)
        print(f"创建测试图像: {test_image_path}")

    # 3. 预处理图像
    input_tensor, original_image = attacker.preprocess_image(test_image_path)

    # 4. 原始预测
    print("\n" + "=" * 50)
    print("原始图像预测")
    print("=" * 50)
    pred_class, confidence, _ = attacker.predict(input_tensor)
    print(f"预测类别: {pred_class} ({attacker.classes[pred_class]})")
    print(f"置信度: {confidence:.4f}")

    # 5. FGSM攻击
    fgsm_adv = attacker.fgsm_attack(
        image_tensor=input_tensor,
        epsilon=0.03,  # 扰动强度
        targeted=False
    )

    # 6. PGD攻击
    pgd_adv = attacker.pgd_attack(
        image_tensor=input_tensor,
        epsilon=0.03,  # 最大扰动
        alpha=0.005,  # 步长
        num_iter=20,  # 迭代次数
        targeted=False
    )

    # 7. 计算攻击指标
    print("\n" + "=" * 50)
    print("攻击效果指标")
    print("=" * 50)

    print("\nFGSM攻击指标:")
    fgsm_metrics = attacker.calculate_metrics(input_tensor, fgsm_adv)
    for metric, value in fgsm_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nPGD攻击指标:")
    pgd_metrics = attacker.calculate_metrics(input_tensor, pgd_adv)
    for metric, value in pgd_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 8. 可视化结果
    attacker.visualize_results(
        original_tensor=input_tensor,
        fgsm_tensor=fgsm_adv,
        pgd_tensor=pgd_adv,
        original_image=original_image,
        save_path='attack_results.png'
    )

    # 9. 额外实验：不同epsilon值的对比
    print("\n" + "=" * 50)
    print("epsilon参数影响实验")
    print("=" * 50)

    epsilons = [0.01, 0.03, 0.05, 0.1]
    results = []

    for eps in epsilons:
        adv = attacker.fgsm_attack(input_tensor, epsilon=eps, targeted=False)
        adv_pred, adv_conf, _ = attacker.predict(adv)
        metrics = attacker.calculate_metrics(input_tensor, adv)
        results.append({
            'epsilon': eps,
            'success': adv_pred != pred_class,
            'confidence': adv_conf,
            'perturbation': metrics['L2距离']
        })
        print(f"epsilon={eps}: 攻击成功={adv_pred != pred_class}, 置信度={adv_conf:.4f}")

    # 绘制epsilon影响曲线
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot([r['epsilon'] for r in results], [r['confidence'] for r in results], 'bo-')
    plt.xlabel('Epsilon')
    plt.ylabel('对抗样本置信度')
    plt.title('Epsilon对置信度的影响')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r['epsilon'] for r in results], [r['perturbation'] for r in results], 'ro-')
    plt.xlabel('Epsilon')
    plt.ylabel('L2扰动距离')
    plt.title('Epsilon对扰动大小的影响')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('epsilon_analysis.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("攻击完成！请查看生成的图像文件。")
    print("=" * 60)

if __name__ == "__main__":
    main()

