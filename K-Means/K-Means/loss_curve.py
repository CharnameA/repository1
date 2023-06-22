import matplotlib.pyplot as plt

def plot_loss_curve(loss, algorithm_name):
    """
    绘制损失函数曲线
    参数：
        - loss: 损失函数值列表
        - algorithm_name: 算法名称
    """
    plt.plot(loss, label=f'{algorithm_name} Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
