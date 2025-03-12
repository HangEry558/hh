# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()  # 加载数据集
X = iris.data  # 特征数据
y = iris.target  # 标签数据

# 数据标准化处理
scaler = StandardScaler()  # 初始化标准化器
X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # 拆分数据集

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 转换训练特征数据为张量
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 转换训练标签数据为张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 转换测试特征数据为张量
y_test_tensor = torch.tensor(y_test, dtype=torch.long)  # 转换测试标签数据为张量

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)  # 创建训练数据集
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 创建测试数据集
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 创建训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 创建测试数据加载器

# 定义模型
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()  # 初始化父类
        self.fc = nn.Sequential(  # 定义序列模型
            nn.Linear(4, 32),  # 输入层到第一个隐藏层，4个输入特征，32个神经元            nn.Linear(32, 16),  # 第一个隐藏层到第二个隐藏层，32个输入神经元，16个输出神经元
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 3)  # 第二个隐藏层到输出层，16个输入神经元，3个输出神经元（对应3个类别）
        )

    def forward(self, x):
        x = self.fc(x)  # 前向传播
        return x

# 初始化模型、损失函数和优化器
model = IrisNet()  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.SGD(model.parameters(), lr=0.9, momentum=0.9)  # 定义优化器

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据加载器
        data, target = data.to(device), target.to(device)  # 将数据移动到指定设备（GPU或CPU）
        optimizer.zero_grad()  # 清零梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if batch_idx % 10 == 9:  # 每10个批次打印一次信息
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 定义测试函数
def test(model, device, test_loader):
    model.eval()  # 设置模型为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测数
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:  # 遍历测试数据加载器
            data, target = data.to(device), target.to(device)  # 将数据移动到指定设备
            output = model(data)  # 前向传播
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测标签
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测数

    test_loss /= len(test_loader.dataset)  # 计算平均损失
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')  # 打印测试结果
    return accuracy  # 返回准确率
# 主函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
model.to(device)  # 将模型移动到指定设备

acc_list_test = []  # 初始化测试准确率列表
for epoch in range(1, 11):  # 训练10个epoch
    train(model, device, train_loader, optimizer, epoch)  # 训练模型
    acc_test = test(model, device, test_loader)  # 测试模型
    acc_list_test.append(acc_test)  # 记录测试准确率

# 可视化准确率
plt.plot(acc_list_test)  # 绘制准确率曲线
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Accuracy On TestSet')  # 设置y轴标签
plt.title('Iris Dataset Accuracy Over Epochs')  # 设置标题
plt.show()  # 显示图形

