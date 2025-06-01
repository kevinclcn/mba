import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filename):
    """加载数据并显示基本信息"""
    try:
        df = pd.read_excel(filename)
        print("="*60)
        print("数据集基本信息")
        print("="*60)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前10行数据:")
        print(df.head(10))
        print("\n数据类型:")
        print(df.dtypes)
        print("\n缺失值统计:")
        print(df.isnull().sum())
        print("\n数值变量描述统计:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def analyze_classifiers(df):
    """分析两个分类器的性能"""
    
    # 提取数据
    y_true = df['真实类别 '].values  # 注意列名后面有空格
    classifier1_proba = df['分类器1预测1的概率'].values
    classifier2_proba = df['分类器2预测1的概率'].values
    
    print("\n" + "="*60)
    print("分类器数据分析")
    print("="*60)
    print(f"样本总数: {len(y_true)}")
    print(f"正样本数量: {np.sum(y_true == 1)} ({np.mean(y_true == 1):.1%})")
    print(f"负样本数量: {np.sum(y_true == 0)} ({np.mean(y_true == 0):.1%})")
    
    print(f"\n分类器1预测概率分布:")
    print(f"  最小值: {classifier1_proba.min():.4f}")
    print(f"  最大值: {classifier1_proba.max():.4f}")
    print(f"  平均值: {classifier1_proba.mean():.4f}")
    print(f"  中位数: {np.median(classifier1_proba):.4f}")
    
    print(f"\n分类器2预测概率分布:")
    print(f"  最小值: {classifier2_proba.min():.4f}")
    print(f"  最大值: {classifier2_proba.max():.4f}")
    print(f"  平均值: {classifier2_proba.mean():.4f}")
    print(f"  中位数: {np.median(classifier2_proba):.4f}")
    
    return y_true, classifier1_proba, classifier2_proba

def plot_roc_curves(y_true, classifier1_proba, classifier2_proba):
    """绘制ROC曲线"""
    
    # 计算ROC曲线
    fpr1, tpr1, thresholds1 = roc_curve(y_true, classifier1_proba)
    fpr2, tpr2, thresholds2 = roc_curve(y_true, classifier2_proba)
    
    # 计算AUC
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr1, tpr1, color='blue', lw=2, 
             label=f'分类器1 (AUC = {auc1:.4f})')
    plt.plot(fpr2, tpr2, color='red', lw=2, 
             label=f'分类器2 (AUC = {auc2:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='随机分类器 (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-特异度 (False Positive Rate)', fontsize=12)
    plt.ylabel('敏感性 (True Positive Rate)', fontsize=12)
    plt.title('ROC曲线比较', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加性能注解
    plt.text(0.6, 0.2, f'AUC差异: {abs(auc1-auc2):.4f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return auc1, auc2, fpr1, tpr1, fpr2, tpr2

def find_optimal_thresholds(y_true, classifier1_proba, classifier2_proba, fpr1, tpr1, fpr2, tpr2):
    """找到最优阈值"""
    
    # 使用Youden指数找最优阈值
    def find_optimal_threshold(fpr, tpr, thresholds):
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        return thresholds[optimal_idx], youden_index[optimal_idx]
    
    # 重新计算以获得阈值
    fpr1, tpr1, thresholds1 = roc_curve(y_true, classifier1_proba)
    fpr2, tpr2, thresholds2 = roc_curve(y_true, classifier2_proba)
    
    optimal_threshold1, youden1 = find_optimal_threshold(fpr1, tpr1, thresholds1)
    optimal_threshold2, youden2 = find_optimal_threshold(fpr2, tpr2, thresholds2)
    
    print("\n" + "="*60)
    print("最优阈值分析")
    print("="*60)
    print(f"分类器1最优阈值: {optimal_threshold1:.4f} (Youden指数: {youden1:.4f})")
    print(f"分类器2最优阈值: {optimal_threshold2:.4f} (Youden指数: {youden2:.4f})")
    
    return optimal_threshold1, optimal_threshold2

def evaluate_at_threshold(y_true, proba, threshold, classifier_name):
    """在给定阈值下评估分类器"""
    y_pred = (proba >= threshold).astype(int)
    
    print(f"\n{classifier_name}性能评估 (阈值: {threshold:.4f}):")
    print(classification_report(y_true, y_pred, target_names=['负类(0)', '正类(1)']))
    
    return y_pred

def plot_confusion_matrices(y_true, y_pred1, y_pred2, threshold1, threshold2):
    """绘制混淆矩阵"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 分类器1混淆矩阵
    cm1 = confusion_matrix(y_true, y_pred1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['预测负类', '预测正类'],
                yticklabels=['实际负类', '实际正类'])
    axes[0].set_title(f'分类器1混淆矩阵\n(阈值: {threshold1:.4f})')
    
    # 分类器2混淆矩阵
    cm2 = confusion_matrix(y_true, y_pred2)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['预测负类', '预测正类'],
                yticklabels=['实际负类', '实际正类'])
    axes[1].set_title(f'分类器2混淆矩阵\n(阈值: {threshold2:.4f})')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_probability_distributions(y_true, classifier1_proba, classifier2_proba):
    """绘制预测概率分布"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 分类器1概率分布
    axes[0,0].hist(classifier1_proba[y_true==0], bins=30, alpha=0.7, label='负类', color='blue')
    axes[0,0].hist(classifier1_proba[y_true==1], bins=30, alpha=0.7, label='正类', color='red')
    axes[0,0].set_title('分类器1预测概率分布')
    axes[0,0].set_xlabel('预测概率')
    axes[0,0].set_ylabel('频数')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 分类器2概率分布
    axes[0,1].hist(classifier2_proba[y_true==0], bins=30, alpha=0.7, label='负类', color='blue')
    axes[0,1].hist(classifier2_proba[y_true==1], bins=30, alpha=0.7, label='正类', color='red')
    axes[0,1].set_title('分类器2预测概率分布')
    axes[0,1].set_xlabel('预测概率')
    axes[0,1].set_ylabel('频数')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 概率散点图
    axes[1,0].scatter(classifier1_proba, classifier2_proba, 
                     c=y_true, cmap='RdYlBu', alpha=0.6)
    axes[1,0].set_xlabel('分类器1预测概率')
    axes[1,0].set_ylabel('分类器2预测概率')
    axes[1,0].set_title('两分类器预测概率关系')
    axes[1,0].grid(True, alpha=0.3)
    
    # 概率差异
    prob_diff = classifier2_proba - classifier1_proba
    axes[1,1].hist(prob_diff, bins=30, alpha=0.7, color='green')
    axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_title('预测概率差异分布\n(分类器2 - 分类器1)')
    axes[1,1].set_xlabel('概率差异')
    axes[1,1].set_ylabel('频数')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_additional_metrics(y_true, classifier1_proba, classifier2_proba):
    """计算额外的性能指标"""
    
    # 计算平均精度(Average Precision)
    from sklearn.metrics import average_precision_score
    ap1 = average_precision_score(y_true, classifier1_proba)
    ap2 = average_precision_score(y_true, classifier2_proba)
    
    # 计算对数损失
    from sklearn.metrics import log_loss
    ll1 = log_loss(y_true, classifier1_proba)
    ll2 = log_loss(y_true, classifier2_proba)
    
    # 计算Brier分数
    from sklearn.metrics import brier_score_loss
    bs1 = brier_score_loss(y_true, classifier1_proba)
    bs2 = brier_score_loss(y_true, classifier2_proba)
    
    print("\n" + "="*60)
    print("额外性能指标")
    print("="*60)
    print(f"平均精度 (Average Precision):")
    print(f"  分类器1: {ap1:.4f}")
    print(f"  分类器2: {ap2:.4f}")
    print(f"  差异: {abs(ap1-ap2):.4f}")
    
    print(f"\n对数损失 (Log Loss, 越小越好):")
    print(f"  分类器1: {ll1:.4f}")
    print(f"  分类器2: {ll2:.4f}")
    print(f"  差异: {abs(ll1-ll2):.4f}")
    
    print(f"\nBrier分数 (越小越好):")
    print(f"  分类器1: {bs1:.4f}")
    print(f"  分类器2: {bs2:.4f}")
    print(f"  差异: {abs(bs1-bs2):.4f}")
    
    return ap1, ap2, ll1, ll2, bs1, bs2

def generate_comparison_report(auc1, auc2, ap1, ap2, ll1, ll2, bs1, bs2):
    """生成比较报告"""
    
    print("\n" + "="*60)
    print("分类器性能比较总结")
    print("="*60)
    
    metrics = {
        'AUC': (auc1, auc2, True),  # True表示越大越好
        'Average Precision': (ap1, ap2, True),
        'Log Loss': (ll1, ll2, False),  # False表示越小越好
        'Brier Score': (bs1, bs2, False)
    }
    
    better_count = {'分类器1': 0, '分类器2': 0}
    
    for metric_name, (val1, val2, higher_better) in metrics.items():
        if higher_better:
            winner = '分类器1' if val1 > val2 else '分类器2'
            difference = abs(val1 - val2)
        else:
            winner = '分类器1' if val1 < val2 else '分类器2'
            difference = abs(val1 - val2)
        
        better_count[winner] += 1
        
        print(f"{metric_name}:")
        print(f"  分类器1: {val1:.4f}")
        print(f"  分类器2: {val2:.4f}")
        print(f"  更好的: {winner} (差异: {difference:.4f})")
        print()
    
    # 总体结论
    if better_count['分类器1'] > better_count['分类器2']:
        overall_winner = '分类器1'
    elif better_count['分类器2'] > better_count['分类器1']:
        overall_winner = '分类器2'
    else:
        overall_winner = '两个分类器表现相当'
    
    print("总体结论:")
    print(f"  {overall_winner}在更多指标上表现更好")
    print(f"  分类器1胜出指标数: {better_count['分类器1']}")
    print(f"  分类器2胜出指标数: {better_count['分类器2']}")
    
    # 保存结果
    results = {
        'classifier1_metrics': {
            'AUC': float(auc1),
            'Average_Precision': float(ap1),
            'Log_Loss': float(ll1),
            'Brier_Score': float(bs1)
        },
        'classifier2_metrics': {
            'AUC': float(auc2),
            'Average_Precision': float(ap2),
            'Log_Loss': float(ll2),
            'Brier_Score': float(bs2)
        },
        'comparison': {
            'overall_winner': overall_winner,
            'classifier1_wins': int(better_count['分类器1']),
            'classifier2_wins': int(better_count['分类器2'])
        }
    }
    
    import json
    with open('classifier_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    
    filename = 'homework2-ROC.xlsx'
    
    # 1. 加载数据
    df = load_data(filename)
    if df is None:
        return
    
    # 2. 分析分类器数据
    y_true, classifier1_proba, classifier2_proba = analyze_classifiers(df)
    
    # 3. 绘制ROC曲线
    auc1, auc2, fpr1, tpr1, fpr2, tpr2 = plot_roc_curves(y_true, classifier1_proba, classifier2_proba)
    
    # 4. 找最优阈值
    threshold1, threshold2 = find_optimal_thresholds(y_true, classifier1_proba, classifier2_proba, 
                                                    fpr1, tpr1, fpr2, tpr2)
    
    # 5. 在最优阈值下评估
    y_pred1 = evaluate_at_threshold(y_true, classifier1_proba, threshold1, "分类器1")
    y_pred2 = evaluate_at_threshold(y_true, classifier2_proba, threshold2, "分类器2")
    
    # 6. 绘制混淆矩阵
    plot_confusion_matrices(y_true, y_pred1, y_pred2, threshold1, threshold2)
    
    # 7. 绘制概率分布
    plot_probability_distributions(y_true, classifier1_proba, classifier2_proba)
    
    # 8. 计算额外指标
    ap1, ap2, ll1, ll2, bs1, bs2 = calculate_additional_metrics(y_true, classifier1_proba, classifier2_proba)
    
    # 9. 生成比较报告
    generate_comparison_report(auc1, auc2, ap1, ap2, ll1, ll2, bs1, bs2)
    
    print("\n" + "="*60)
    print("分析完成！生成的文件:")
    print("="*60)
    print("- roc_curves_comparison.png: ROC曲线比较图")
    print("- confusion_matrices_comparison.png: 混淆矩阵比较图")
    print("- probability_distributions.png: 概率分布分析图")
    print("- classifier_comparison_results.json: 详细比较结果")

if __name__ == "__main__":
    main() 