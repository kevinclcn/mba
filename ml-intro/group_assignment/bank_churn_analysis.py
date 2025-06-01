import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, accuracy_score,
                           precision_score, recall_score, f1_score)
import warnings

class BankChurnAnalysis:
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据集"""
        print("正在加载数据集...")
        try:
            # 获取脚本文件所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 构建数据文件路径（相对于脚本文件位置）
            train_path = os.path.join(script_dir, 'BankChurners_train.xlsx')
            val_path = os.path.join(script_dir, 'BankChurners_validation.xlsx')
            test_path = os.path.join(script_dir, 'BankChurners_test.xlsx')
            
            self.train_data = pd.read_excel(train_path)
            self.val_data = pd.read_excel(val_path)
            self.test_data = pd.read_excel(test_path)
            
            print(f"训练集大小: {self.train_data.shape}")
            print(f"验证集大小: {self.val_data.shape}")
            print(f"测试集大小: {self.test_data.shape}")
            
            return True
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """探索性数据分析"""
        print("\n" + "="*60)
        print("探索性数据分析")
        print("="*60)
        
        # 数据结构探索
        print("\n0. 数据结构探索:")
        print(f"所有列名 (共{len(self.train_data.columns)}列):")
        for i, col in enumerate(self.train_data.columns):
            dtype = self.train_data[col].dtype
            unique_count = self.train_data[col].nunique()
            print(f"  {i+1:2d}. {col:<25} | 类型: {str(dtype):<10} | 唯一值数: {unique_count:>3}")
        
        # 分析类别型特征
        categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        print(f"\n指定的类别型特征分析:")
        for col in categorical_features:
            if col in self.train_data.columns:
                unique_vals = sorted(self.train_data[col].unique())
                print(f"  ✅ {col}: {unique_vals}")
            else:
                print(f"  ❌ {col}: 未找到此列")
        
        # 基本信息
        print("\n1. 数据集基本信息:")
        print(f"训练集形状: {self.train_data.shape}")
        print(f"数据类型分布:")
        print(self.train_data.dtypes.value_counts())
        
        # 目标变量分布
        print(f"\n2. 目标变量分布:")
        target_dist = self.train_data['Attrition_Flag'].value_counts()
        print(target_dist)
        print(f"流失率: {target_dist[1] / len(self.train_data):.2%}")
        
        # 缺失值检查
        print(f"\n3. 缺失值检查:")
        missing_values = self.train_data.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("无缺失值")
        
        # 数值变量描述统计
        print(f"\n4. 数值变量描述统计:")
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
        print(f"数值型列数: {len(numeric_cols)}")
        if len(numeric_cols) > 0:
            print(self.train_data[numeric_cols].describe())
        
        # 可视化
        self.create_eda_plots()
        
    def create_eda_plots(self):
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        """创建探索性数据分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标变量分布
        target_counts = self.train_data['Attrition_Flag'].value_counts()
        axes[0,0].pie(target_counts.values, labels=['未流失', '流失'], autopct='%1.1f%%')
        axes[0,0].set_title('客户流失分布')
        
        # 2. 年龄分布
        axes[0,1].hist([self.train_data[self.train_data['Attrition_Flag']==0]['Customer_Age'],
                        self.train_data[self.train_data['Attrition_Flag']==1]['Customer_Age']], 
                       bins=20, alpha=0.7, label=['未流失', '流失'])
        axes[0,1].set_title('年龄分布对比')
        axes[0,1].set_xlabel('年龄')
        axes[0,1].set_ylabel('频数')
        axes[0,1].legend()
        
        # 3. 信用额度分布
        axes[0,2].boxplot([self.train_data[self.train_data['Attrition_Flag']==0]['Credit_Limit'],
                           self.train_data[self.train_data['Attrition_Flag']==1]['Credit_Limit']])
        axes[0,2].set_title('信用额度分布对比')
        axes[0,2].set_xticklabels(['未流失', '流失'])
        axes[0,2].set_ylabel('信用额度')
        
        # 4. 交易金额与流失关系
        axes[1,0].scatter(self.train_data['Total_Trans_Amt'], self.train_data['Total_Trans_Ct'], 
                         c=self.train_data['Attrition_Flag'], alpha=0.6, cmap='RdYlBu')
        axes[1,0].set_title('交易金额与交易次数关系')
        axes[1,0].set_xlabel('总交易金额')
        axes[1,0].set_ylabel('总交易次数')
        
        # 5. 收入类别与流失关系
        income_churn = pd.crosstab(self.train_data['Income_Category'], 
                                  self.train_data['Attrition_Flag'], normalize='index')
        income_churn.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('收入类别与流失率关系')
        axes[1,1].set_xlabel('收入类别')
        axes[1,1].set_ylabel('比例')
        axes[1,1].legend(['未流失', '流失'])
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. 相关性热力图（选择部分数值变量）
        numeric_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                       'Total_Relationship_Count', 'Credit_Limit', 'Total_Trans_Amt', 
                       'Total_Trans_Ct', 'Attrition_Flag']
        corr_matrix = self.train_data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('特征相关性热力图')
        
        plt.tight_layout()
        # 获取脚本文件所在目录，保存到当前目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'eda_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_features(self, data, fit_scaler=False):
        """特征预处理 - 处理已存在的虚拟变量"""
        # 复制数据
        df = data.copy()
        
        # 选择特征列（排除目标变量和id）
        feature_cols = [col for col in df.columns if col not in ['Attrition_Flag', 'id']]
        
        print(f"\n特征预处理信息:")
        print(f"原始特征数: {len(feature_cols)}")
        
        # 定义原始的数字化类别特征（需要删除，因为已有虚拟变量）
        original_categorical = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        
        # 明确定义虚拟变量组（基于实际数据结构）
        dummy_groups = {
            'Gender': ['Gender_F', 'Gender_M'],
            'Education_Level': [
                'Education_Level_Unknown', 'Education_Level_Uneducated', 
                'Education_Level_High_School', 'Education_Level_College',
                'Education_Level_Graduate', 'Education_Level_Post_Graduate', 
                'Education_Level_Doctorate'
            ],
            'Marital_Status': [
                'Marital_Status_Unknown', 'Marital_Status_Single',
                'Marital_Status_Divorced', 'Marital_Status_Married'
            ],
            'Income_Category': [
                'Income_Category_Unknown', 'Income_Category_Less_than_40k',
                'Income_Category_40k_60k', 'Income_Category_60k_80k',
                'Income_Category_80k_120k', 'Income_Category_over120k'
            ],
            'Card_Category': [
                'Card_Category_Blue', 'Card_Category_Silver',
                'Card_Category_Gold', 'Card_Category_Platinum'
            ]
        }
        
        # 收集所有虚拟变量
        all_dummy_vars = []
        for group_vars in dummy_groups.values():
            all_dummy_vars.extend(group_vars)
        
        # 数值型特征（不是原始类别特征，也不是虚拟变量）
        numeric_features = [col for col in feature_cols 
                           if col not in original_categorical and col not in all_dummy_vars]
        
        print(f"\n特征分类结果:")
        print(f"  - 原始类别特征(将删除): {original_categorical}")
        print(f"  - 数值型特征数量: {len(numeric_features)}")
        print(f"  - 虚拟变量组数量: {len(dummy_groups)}")
        
        # 验证虚拟变量是否存在并显示
        existing_dummy_groups = {}
        for base_name, group_vars in dummy_groups.items():
            existing_vars = [col for col in group_vars if col in df.columns]
            if existing_vars:
                existing_dummy_groups[base_name] = existing_vars
                print(f"    {base_name}: {existing_vars} (共{len(existing_vars)}个)")
            else:
                print(f"    {base_name}: 未找到虚拟变量")
        
        # 处理虚拟变量组，删除一个以避免共线性
        final_features = numeric_features.copy()
        removed_features = []
        
        for base_name, group_vars in existing_dummy_groups.items():
            if len(group_vars) > 1:
                # 删除第一个变量作为参考类别
                reference_var = group_vars[0]
                selected_vars = group_vars[1:]
                removed_features.append(reference_var)
                
                print(f"\n处理虚拟变量组 '{base_name}':")
                print(f"  - 删除参考类别: {reference_var}")
                print(f"  - 保留变量: {selected_vars}")
                
                final_features.extend(selected_vars)
            else:
                # 只有一个变量的组，直接保留
                print(f"\n虚拟变量组 '{base_name}' 只有1个变量，直接保留: {group_vars}")
                final_features.extend(group_vars)
        
        print(f"\n最终特征选择结果:")
        print(f"  - 数值型特征: {len(numeric_features)}")
        print(f"  - 保留的虚拟变量: {len(final_features) - len(numeric_features)}")
        print(f"  - 总特征数: {len(final_features)}")
        print(f"  - 删除的原始类别特征: {len(original_categorical)}")
        print(f"  - 删除的参考虚拟变量: {len(removed_features)}")
        print(f"  - 删除的参考变量: {removed_features}")
        
        # 准备最终特征矩阵
        X_processed = df[final_features]
        
        print(f"\n数据质量检查:")
        print(f"  - 最终数据形状: {X_processed.shape}")
        print(f"  - 数据类型分布: {X_processed.dtypes.value_counts().to_dict()}")
        
        # 检查缺失值
        missing_count = X_processed.isnull().sum().sum()
        if missing_count > 0:
            print(f"  - 警告: 发现 {missing_count} 个缺失值，将用0填充")
            X_processed = X_processed.fillna(0)
        else:
            print(f"  - 无缺失值")
        
        # 确保所有特征都是数值型
        non_numeric_cols = X_processed.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"  - 警告: 非数值型列: {list(non_numeric_cols)}")
            for col in non_numeric_cols:
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        
        # 标准化特征
        if fit_scaler:
            print(f"  - 执行 fit_transform (训练集标准化)")
            X_scaled = self.scaler.fit_transform(X_processed)
            # 保存特征名称用于后续分析
            self.feature_names_ = final_features
            print(f"  - 保存了 {len(self.feature_names_)} 个特征名称")
        else:
            print(f"  - 执行 transform (验证/测试集标准化)")
            X_scaled = self.scaler.transform(X_processed)
        
        print(f"  - 标准化后形状: {X_scaled.shape}")
        
        # 获取目标变量（如果存在）
        if 'Attrition_Flag' in df.columns:
            y = df['Attrition_Flag']
            return X_scaled, y, final_features
        else:
            return X_scaled, None, final_features
    
    def train_models(self):
        """训练多个机器学习模型"""
        print("\n" + "="*60)
        print("模型训练")
        print("="*60)
        
        # 准备训练数据
        X_train, y_train, feature_names = self.prepare_features(self.train_data, fit_scaler=True)
        X_val, y_val, _ = self.prepare_features(self.val_data, fit_scaler=False)
        
        print(f"特征数量: {X_train.shape[1]}")
        print(f"训练样本数: {X_train.shape[0]}")
        print(f"验证样本数: {X_val.shape[0]}")
        
        # 定义模型
        models_config = {
            'Logistic_Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
            },
            'Decision_Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=42, n_estimators=100),
                'params': {'max_depth': [10, 15, None], 'min_samples_split': [2, 5]}
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.5, 1.0, 1.5]}
            },
            'Neural_Network_1Layer': {
                'model': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500),
                'params': {'alpha': [0.001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.01]}
            },
            'Neural_Network_2Layer': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
                'params': {'alpha': [0.001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.01]}
            },
            'Neural_Network_3Layer': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50, 25), random_state=42, max_iter=500),
                'params': {'alpha': [0.001, 0.01, 0.1], 'learning_rate_init': [0.001, 0.01]}
            }
        }
        
        # 训练和调参
        for name, config in models_config.items():
            print(f"\n训练模型: {name}")
            
            # 网格搜索调参
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 保存最佳模型
            best_model = grid_search.best_estimator_
            self.models[name] = best_model
            
            # 在验证集上评估
            y_pred = best_model.predict(X_val)
            y_pred_proba = best_model.predict_proba(X_val)[:, 1]
            
            # 计算性能指标
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'auc': roc_auc_score(y_val, y_pred_proba),
                'best_params': grid_search.best_params_
            }
            
            self.results[name] = metrics
            
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"验证集AUC: {metrics['auc']:.4f}")
            print(f"验证集准确率: {metrics['accuracy']:.4f}")
    
    def compare_models(self):
        """比较模型性能"""
        print("\n" + "="*60)
        print("模型性能比较")
        print("="*60)
        
        # 创建结果对比表
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.drop('best_params', axis=1)
        comparison_df = comparison_df.round(4)
        
        print("\n模型性能对比表:")
        print(comparison_df)
        
        # 找出最佳模型
        best_model_name = comparison_df['auc'].idxmax()
        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳AUC: {comparison_df.loc[best_model_name, 'auc']:.4f}")
        
        # 可视化模型比较
        self.plot_model_comparison(comparison_df)
        
        return best_model_name
    
    def plot_model_comparison(self, comparison_df):
        """绘制模型性能比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'auc']
        titles = ['准确率', '精确率', '召回率', 'AUC']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            values = comparison_df[metric].values
            models = comparison_df.index
            
            bars = ax.bar(range(len(models)), values, alpha=0.7)
            ax.set_title(f'{title}比较')
            ax.set_ylabel(title)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        # 获取脚本文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self):
        """绘制所有模型的ROC曲线"""
        plt.figure(figsize=(12, 8))
        
        # 准备验证数据
        X_val, y_val, _ = self.prepare_features(self.val_data, fit_scaler=False)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        for (name, model), color in zip(self.models.items(), colors):
            # 预测概率
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # 绘制ROC曲线
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} (AUC = {auc_score:.3f})')
        
        # 绘制随机分类器线
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='随机分类器 (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)')
        plt.ylabel('真阳性率 (True Positive Rate)')
        plt.title('ROC曲线比较')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 获取脚本文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_best_model(self, best_model_name):
        """在测试集上评估最佳模型"""
        print("\n" + "="*60)
        print(f"最佳模型测试集评估: {best_model_name}")
        print("="*60)
        
        # 准备测试数据
        X_test, y_test, _ = self.prepare_features(self.test_data, fit_scaler=False)
        
        # 获取最佳模型
        best_model = self.models[best_model_name]
        
        # 预测
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # 性能指标
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n测试集性能指标:")
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=['未流失', '流失']))
        
        # 混淆矩阵
        self.plot_confusion_matrix(y_test, y_pred, best_model_name)
        
        # 特征重要性（如果模型支持）
        self.plot_feature_importance(best_model, best_model_name)
        
        return test_metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['未流失', '流失'],
                   yticklabels=['未流失', '流失'])
        plt.title(f'{model_name} - 混淆矩阵')
        plt.xlabel('预测结果')
        plt.ylabel('实际结果')
        
        plt.tight_layout()
        # 获取脚本文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, model_name):
        """绘制特征重要性"""
        try:
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                print(f"模型 {model_name} 不支持特征重要性分析")
                return
            
            # 获取特征名称
            if hasattr(self, 'feature_names_') and len(self.feature_names_) == len(importances):
                feature_names = self.feature_names_
                print(f"使用真实特征名称 (共{len(feature_names)}个特征)")
            else:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
                print(f"使用默认特征名称 (共{len(feature_names)}个特征)")
            
            # 排序
            indices = np.argsort(importances)[::-1]
            
            # 选择前20个最重要的特征
            n_features = min(20, len(importances))
            
            # 获取Top特征的名称和重要性
            top_features = [(feature_names[i], importances[i]) for i in indices[:n_features]]
            
            print(f"\n{model_name} - 前{n_features}个重要特征:")
            for i, (name, importance) in enumerate(top_features, 1):
                print(f"  {i:2d}. {name:<25}: {importance:.4f}")
            
            plt.figure(figsize=(14, 8))
            bars = plt.bar(range(n_features), importances[indices[:n_features]], alpha=0.7)
            plt.title(f'{model_name} - 前{n_features}个重要特征')
            plt.xlabel('特征')
            plt.ylabel('重要性')
            
            # 设置x轴标签，旋转以避免重叠
            feature_labels = [feature_names[i] for i in indices[:n_features]]
            plt.xticks(range(n_features), feature_labels, rotation=45, ha='right')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            # 获取脚本文件所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, f'feature_importance_{model_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"绘制特征重要性时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_business_insights(self, best_model_name, test_metrics):
        """生成商业洞察和建议"""
        print("\n" + "="*60)
        print("商业洞察与应用建议")
        print("="*60)
        
        print(f"\n1. 模型性能评估:")
        print(f"   - 最佳模型: {best_model_name}")
        print(f"   - 测试集AUC: {test_metrics['auc']:.4f}")
        print(f"   - 测试集准确率: {test_metrics['accuracy']:.4f}")
        print(f"   - 测试集召回率: {test_metrics['recall']:.4f}")
        
        # 模型质量评估
        auc_score = test_metrics['auc']
        if auc_score >= 0.9:
            quality = "优秀"
        elif auc_score >= 0.8:
            quality = "良好"
        elif auc_score >= 0.7:
            quality = "可接受"
        else:
            quality = "需要改进"
        
        print(f"\n2. 模型质量评估: {quality}")
        
        print(f"\n3. 商业应用建议:")
        if auc_score >= 0.8:
            print("   ✅ 模型具备商业应用价值，建议部署使用")
            print("   ✅ 可用于识别高风险流失客户，实施针对性挽留策略")
            print("   ✅ 建议结合业务规则，设置合适的预测阈值")
        else:
            print("   ⚠️  模型性能有待提升，建议进一步优化后再考虑商业应用")
            print("   ⚠️  可考虑收集更多特征数据或尝试其他算法")
        
        print(f"\n4. 实施建议:")
        print("   - 建立模型监控机制，定期评估模型性能")
        print("   - 设置客户流失风险评分系统")
        print("   - 制定差异化的客户挽留策略")
        print("   - 定期重训练模型以适应数据分布变化")
        
        # 保存结果摘要
        self.save_results_summary(best_model_name, test_metrics)
    
    def save_results_summary(self, best_model_name, test_metrics):
        """保存结果摘要"""
        results_summary = {
            'best_model': best_model_name,
            'test_metrics': test_metrics,
            'all_model_results': self.results,
            'model_parameters': {name: model.get_params() for name, model in self.models.items()}
        }
        
        import json
        # 获取脚本文件所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, 'results_summary.json')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n结果摘要已保存到: {save_path}")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始银行客户流失预测分析...")
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 探索性数据分析
        self.exploratory_data_analysis()
        
        # 3. 训练模型
        self.train_models()
        
        # 4. 比较模型
        best_model_name = self.compare_models()
        
        # 5. 绘制ROC曲线
        self.plot_roc_curves()
        
        # 6. 测试集评估
        test_metrics = self.evaluate_best_model(best_model_name)
        
        # 7. 商业洞察
        self.generate_business_insights(best_model_name, test_metrics)
        
        print("\n分析完成！所有结果已保存到 group_assignment/ 文件夹")

def main():
    """主函数"""
    # 创建分析实例
    analyzer = BankChurnAnalysis()
    
    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 