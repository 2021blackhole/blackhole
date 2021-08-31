## Blackhole ml API
Blackhole ml是高性能机器学习引擎，支持类sklearn接口的常见机器学习算子。


### save_model
blackhole ml保存模型方法。
* 参数:
    * model: 待保存模型
    * file_path: str，
        模型的保存路径


### load_model
blackhole ml加载模型方法。
* 参数:
    * file_path: str，
        模型保存的路径


### load_file
blackhole ml加载数据文件方法，支持CSV、Parquet、ORC等多种数据格式的加载。
* 参数:
    * file_path : str，
        指定要导入的数据文件或文件夹的路径
    * header : 默认0，
        -1表示第一行是数据，0表示猜测，1表示第一行是标题
    * sep : 默认None，
        字段分隔符，若None，解析器将自动推测分隔符
    * col_names : list，
        列名
    * col_types : list，
        列数据类别
    * na_strings : list，
        缺失值表示
    * pattern : str，
        当file_path为文件夹时，设定匹配文件夹中文件名的正则表达式
    * skipped_columns : list，
        跳过的列名
    * custom_non_data_line_markers : str，
        如果导入文件中的某行以给定的字符串中的任何字符为开头，则不会导入该行,\
        空字符串表示导入所有行


### RandomForestClassifier
随机森林分类算法
* 参数:
    * n_estimators : int，默认50，
        随机森林中树的棵数
    * criterion : 默认'gini'，
        样本集切分的度量指标
    * max_depth : int，默认20，
        树的最大深度
    * min_samples_split : int，默认2，
        拆分内部节点所需的最小采样数
    * min_samples_leaf : int，默认1，
        叶节点所需的最小采样数
    * min_weight_fraction_leaf : float，默认0.0，
        叶节点上样本权重加权和的最小值
    * max_features : 默认'auto'，
        寻找最佳分裂时，考虑的特征个数的最大值
    * max_leaf_nodes : int，
        最大叶子节点数
    * min_impurity_decrease : 默认1e-05，
        如果节点分裂导致不纯度的减少大于或等于这个值，则该节点将被分裂，
    * min_impurity_split : float，默认None
        节点划分的最小不纯度
    * bootstrap : bool，默认True，
        生成树时，是否进行有放回的抽样
    * oob_score : bool，默认False，
        是否使用袋外样本评估泛化精度
    * n_jobs : int，默认1，
        并行的job数量
    * random_state : int，默认None，
        控制算法过程中的随机性
    * verbose : int，默认0，
        控制训练和预测时的verbose信息
    * warm_start : bool，默认False，
        若为True，复用之前的模型，增加更多元估计器，进行训练，
        若为False，则训练一个全新的随机森里模型
    * class_weight : {“balanced”, “balanced_subsample”}
        样本的权重
    * score_each_iteration : bool，默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * score_tree_interval : int，默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项，
    * balance_classes : bool，默认False，
        过采样少数类别，以平衡样本分布，这会增加样本规模，此选项只适用分类，
    * class_sampling_factors : float，
        指定每个类别在采样时的过采样／欠采样率
    * max_after_balance_size : 默认5.0，
        指定平衡类数量后，相对原数据规模的最大比例
    * max_hit_ratio_k : 默认0，
        指定用于命中率计算的最大预测数，仅适用于多分类，0禁用
    * nbins : int，默认20，
        指定要构建的直方图的bin的数量
    * nbins_top_level : int，默认1024，
        指定在根节点用于构建直方图的最小bin数，此数量将每级别减少2倍
    * nbins_cats :int，默认1024，
         指定要构建的直方图的最大bin数
    * stopping_rounds : int，默认0，
        指定stopping_rounds合数后，在stopping_metric没有改善的情况下将停止         训练，0禁用
    * stopping_metric : 指定用于提前停止训练的评判指标，默认'AUTO'，
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'AUCPR'、'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'   
    * stopping_tolerance : float，默认0.001
        指定相对误差，如果提升小于该值，则停止训练
    * max_runtime_secs : int，默认0,
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * build_tree_one_node : bool，默认False,
        指定是否在单个机器节点上运行，适用于小数据集，无网络开销
    * mtries : 指定每个级别随机选择的列数，
        如果-1，则对于分类问题为列数的平方根，回归问题为p/3，
        如果-2，则使用所有列，此选项的所有值为-1，-2和任何大于1的值，默认-1
    * sample_rate : float，默认0.632,
        指定行采样率
    * binomial_double_trees : bool，默认False,
        仅适用于二分类，构建两倍的树（每类一棵），启用此选项可以提高准确性，
        禁用则可以加快模型构建
    * col_sample_rate_change_per_level : 默认1,
        此选型指定根据树的深度来更改列采样率，可以是0.0到2.0的值
    * col_sample_rate_per_tree : 默认1.0,
        指定每棵树的列采样率
    * histogram_type : 默认'AUTO'，
        直方图的类型,可选:'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * categorical_encoding : 指定以下编码方案之一来处理分类特征
        (默认'AUTO')，可选：
        'AUTO'、'Enumenum_limited'、'OneHotExplicit'、'Binary'、
        'Eigen'、'LabelEncoder'、'SortByResponse'
    * custom_metric_func : 指定一个自定义评估函数，
        自定义评估函数
    * export_checkpoints_dir : 将生成的模型自动导出到
        指定的目录
    * check_constant_response : bool，默认False，
        检查响应列是否为常量值，启用下，若响应列为常量值，则引发异常，\
        如果禁用，无论响应列是否为恒定值，模型都将训练
    
* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data
    * predict_proba，包含一个参数：待预测数据：test_data


### RandomForestRegressor
随机森例回归算法
* 参数:
    * n_estimators : int，默认50，
        随机森林中树的棵数
    * criterion : 默认'mse'，
        样本集切分的度量指标
    * max_depth : int，默认20，
        树的最大深度
    * min_samples_split : int，默认2，
        拆分内部节点所需的最小采样数
    * min_samples_leaf : int，默认1，
        叶节点所需的最小采样数
    * min_weight_fraction_leaf : float，默认0.0，
        叶节点上样本权重加权和的最小值
    * max_features : 默认'auto'，
        寻找最佳分裂时，考虑的特征个数的最大值
    * max_leaf_nodes : int，
        最大叶子节点数
    * min_impurity_decrease : 默认1e-05，
        如果节点分裂导致不纯度的减少大于或等于这个值，则该节点将被分裂，
    * min_impurity_split : float，默认None，
        节点划分的最小不纯度
    * bootstrap : bool，默认True，
        生成树时，是否进行有放回的抽样
    * oob_score : bool，默认False，
        是否使用袋外样本评估泛化精度
    * n_jobs : int，默认1，
        并行的job数量
    * random_state : int，默认None，
        控制算法过程中的随机性
    * verbose : int，默认0，
        控制训练和预测时的verbose信息
    * warm_start : bool，默认False，
        若为True，复用之前的模型，增加更多元估计器，进行训练，
        若为False，则训练一个全新的随机森里模型
    * class_weight : {“balanced”, “balanced_subsample”}
        样本的权重
    * score_each_iteration : bool，默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * score_tree_interval : int，默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项，
    * balance_classes : bool，默认False，
        过采样少数类别，以平衡样本分布，这会增加样本规模，此选项只适用分类，
    * class_sampling_factors : float，
        指定每个类别在采样时的过采样／欠采样率
    * max_after_balance_size : 默认5.0，
        指定平衡类数量后，相对原数据规模的最大比例
    * max_hit_ratio_k : 默认0，
        指定用于命中率计算的最大预测数，仅适用于多分类，0禁用
    * nbins : int，默认20，
        指定要构建的直方图的bin的数量
    * nbins_top_level : int，默认1024，
        指定在根节点用于构建直方图的最小bin数，此数量将每级别减少2倍
    * nbins_cats :int，默认1024，
         指定要构建的直方图的最大bin数
    * stopping_rounds : int，默认0，
        指定stopping_rounds合数后，在stopping_metric没有改善的情况下将停止         训练，0禁用
    * stopping_metric : 指定用于提前停止训练的评判指标，默认'AUTO'，
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'AUCPR'、'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'   
    * stopping_tolerance : float，默认0.001，
        指定相对误差，如果提升小于该值，则停止训练
    * max_runtime_secs : int，默认0,
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * build_tree_one_node : bool，默认False,
        指定是否在单个机器节点上运行，适用于小数据集，无网络开销
    * mtries : 指定每个级别随机选择的列数，
        如果-1，则对于分类问题为列数的平方根，回归问题为p/3，
        如果-2，则使用所有列，此选项的所有值为-1，-2和任何大于1的值，默认-1
    * sample_rate : float，默认0.632,
        指定行采样率
    * binomial_double_trees : bool，默认False,
        仅适用于二分类，构建两倍的树（每类一棵），启用此选项可以提高准确性，
        禁用则可以加快模型构建
    * col_sample_rate_change_per_level : 默认1,
        此选型指定根据树的深度来更改列采样率，可以是0.0到2.0的值
    * col_sample_rate_per_tree : 默认1.0,
        指定每棵树的列采样率
    * histogram_type : 默认'AUTO'，
        直方图的类型,可选:'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * categorical_encoding : 指定以下编码方案之一来处理分类特征
        (默认'AUTO')，可选：
        'AUTO'、'Enumenum_limited'、'OneHotExplicit'、'Binary'、
        'Eigen'、'LabelEncoder'、'SortByResponse'
    * custom_metric_func : 指定一个自定义评估函数，
        自定义评估函数
    * export_checkpoints_dir : 将生成的模型自动导出到
        指定的目录
    * check_constant_response : bool，默认False，
        检查响应列是否为常量值，启用下，若响应列为常量值，则引发异常，
        如果禁用，无论响应列是否为恒定值，模型都将训练
    
* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### GradientBoostingClassifier
梯度提升数分类算法
* 参数:
    * loss : 默认'deviance'，
        损失函数，可选'deviance'，'exponential'
    * learning_rate : float，默认0.1，
        学习率
    * n_estimators : int，默认50，
        树的棵数
    * subsample : 默认1，
        控制每棵树随机采样的比例
    * criterion : 默认'friedman_mse'，
        样本集切分的度量指标可选'friedman_mse'、 'mse'、 'mae'
    * min_samples_split : 默认2，
        拆分内部节点所需的最小采样数
    * min_samples_leaf : 默认1，
        叶节点所需的最小采样数
    * min_weight_fraction_leaf : 默认0.0，
        叶节点上样本权重加权和的最小值
    * max_depth : int，默认5，
        树的最大深度
    * min_impurity_decrease : 默认1e-05，
        如果节点分裂导致不纯度的减少大于或等于这个值，则该节点将被分裂
    * min_impurity_split : float，默认None，
        节点划分的最小不纯度
    * init : 默认None，
        用来计算初始预测的估计器
    * random_state : 默认None，
        随机数种子
    * max_features : 默认None，
        寻找最佳分裂时，考虑的特征个数的最大值
    * verbose : 默认0，
        控制训练和预测时的verbose信息
    * max_leaf_nodes : int，默认None，
        最大叶子节点数
    * warm_start : bool，默认False，
        若为True，复用之前的模型，增加更多元估计器，进行训练，若为False，
        则训练一个全新的模型
    * validation_fraction : float，默认0.1，
        验证集的比例，用于early stopping
    * n_iter_no_change : 默认None，
        当n轮迭代没有效果提升时，提前终止训练
    * tol : float，默认0.001，
        进行early stopping时效果提升的阈值
    * score_each_iteration : bool，默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * score_tree_interval : 默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项
    * ignore_const_cols : bool，默认True，
        是否忽略常量训练列，因为无法从中获取任何信息
    * balance_classes : bool，默认False，
        过采样少数类别，以平衡样本分布，这会增加样本规模，此选项只适用分类
    * class_sampling_factors : float，
            指定每个类别在采样时的过采样／欠采样率
    * max_after_balance_size : 默认5.0，
        指定平衡类数量后，相对原数据规模的最大比例
    * max_hit_ratio_k : 默认0，
        指定用于命中率计算的最大预测数，仅适用于多分类，0禁用
    * nbins : int，默认20，
        指定要构建的直方图的bin的数量
    * nbins_top_level : int，默认1024，
        指定在根节点用于构建直方图的最小bin数，此数量将每级别减少2倍
    * nbins_cats : int，默认1024，
        指定要构建的直方图的最大bin数
    * stopping_metric : 默认'AUTO'，
        指定用于提前停止训练的评判指标可选:
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'
    * max_runtime_secs : 默认0，
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * build_tree_one_node : bool，默认False
        指定是否在单个机器节点上运行，适用于小数据集，无网络开销
    * col_sample_rate : float，默认1.0，
        列采样率
    * col_sample_rate_change_per_level : 默认1，
        此选型指定根据树的深度来更改列采样率，可以是0.0到2.0的值
    * col_sample_rate_per_tree : 指定每棵树的列采样率，默认1.0
    * histogram_type : 默认'AUTO'，
        直方图的类型,可选:'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * max_abs_leafnode_pred : 默认1.7976931348623157e308，
        允许每个叶子输出的最大增量步长，0则无约束
    * categorical_encoding : 默认'AUTO'，
        指定以下编码方案之一，来处理分类特征，
        'AUTO'、'Enumenum_limited'、'OneHotExplicit'、                       'Binary'、'Eigen'、'LabelEncoder'、'SortByResponse'          
    * custom_metric_func : 指定一个自定义评估函数，
        自定义评估函数
    * export_checkpoints_dir : 将生成的模型自动导出到指定的目录，
        指定目录
    * monotone_constraints : 表示单调约束的映射，
        +1表示递增约束，-1表示递减约束
    * check_constant_response : bool，默认False，
        检查响应列是否为常量值，启用下，若响应列为常量值，则引发异常，
        如果禁用，无论响应列是否为恒定值，模型都将训练
        
* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### GradientBoostingRegressor
梯度提升树回算法
* 参数:
    * loss : 默认'ls'，
        损失函数,可选: 'ls、'lad'、'huber'、'quantile'
    * learning_rate : float，默认0.1，
        学习率
    * n_estimators : int，默认50，
        树的棵数
    * subsample : 默认1，
        控制每棵树随机采样的比例
    * criterion : 默认'friedman_mse'，
        样本集切分的度量指标可选'friedman_mse'、 'mse'、 'mae'
    * min_samples_split : 默认2，
        拆分内部节点所需的最小采样数
    * min_samples_leaf : 默认1，
        叶节点所需的最小采样数
    * min_weight_fraction_leaf : 默认0.0，
        叶节点上样本权重加权和的最小值
    * max_depth : int，默认5，
        树的最大深度
    * min_impurity_decrease : 默认1e-05，
        如果节点分裂导致不纯度的减少大于或等于这个值，则该节点将被分裂
    * min_impurity_split : float，默认None，
        节点划分的最小不纯度
    * init : 默认None，
        用来计算初始预测的估计器
    * random_state : 默认None，
        随机数种子
    * max_features : 默认None，
        寻找最佳分裂时，考虑的特征个数的最大值
    * verbose : 默认0，
        控制训练和预测时的verbose信息
    * max_leaf_nodes : int，默认None，
        最大叶子节点数
    * warm_start : bool，默认False，
        若为True，复用之前的模型，增加更多元估计器，进行训练，若为False，
        则训练一个全新的模型
    * validation_fraction : float，默认0.1，
        验证集的比例，用于early stopping
    * n_iter_no_change : 默认None，
        当n轮迭代没有效果提升时，提前终止训练
    * tol : float，默认0.001，
        进行early stopping时效果提升的阈值
    * score_each_iteration : bool，默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * score_tree_interval : 默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项
    * ignore_const_cols : bool，默认True，
        是否忽略常量训练列，因为无法从中获取任何信息
    * balance_classes : bool，默认False，
        过采样少数类别，以平衡样本分布，这会增加样本规模，此选项只适用分类
    * class_sampling_factors : float，
            指定每个类别在采样时的过采样／欠采样率
    * max_after_balance_size : 默认5.0，
        指定平衡类数量后，相对原数据规模的最大比例
    * max_hit_ratio_k : 默认0，
        指定用于命中率计算的最大预测数，仅适用于多分类，0禁用
    * nbins : int，默认20，
        指定要构建的直方图的bin的数量
    * nbins_top_level : int，默认1024，
        指定在根节点用于构建直方图的最小bin数，此数量将每级别减少2倍
    * nbins_cats : int，默认1024，
        指定要构建的直方图的最大bin数
    * stopping_metric : 默认'AUTO'，
        指定用于提前停止训练的评判指标可选:
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'
    * max_runtime_secs : 默认0，
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * build_tree_one_node : bool，默认False
        指定是否在单个机器节点上运行，适用于小数据集，无网络开销
    * col_sample_rate : float，默认1.0，
        列采样率
    * col_sample_rate_change_per_level : 默认1，
        此选型指定根据树的深度来更改列采样率，可以是0.0到2.0的值
    * col_sample_rate_per_tree : 指定每棵树的列采样率，默认1.0
    * histogram_type : 默认'AUTO'，
        直方图的类型,可选:'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * max_abs_leafnode_pred : 默认1.7976931348623157e308，
        允许每个叶子输出的最大增量步长，0则无约束
    * categorical_encoding : 默认'AUTO'，
        指定以下编码方案之一，来处理分类特征，
        'AUTO'、'Enumenum_limited'、'OneHotExplicit'、                       'Binary'、'Eigen'、'LabelEncoder'、'SortByResponse'          
    * custom_metric_func : 指定一个自定义评估函数，
        自定义评估函数
    * export_checkpoints_dir : 将生成的模型自动导出到指定的目录，
        指定目录
    * monotone_constraints : 表示单调约束的映射，
        +1表示递增约束，-1表示递减约束
    * check_constant_response : bool，默认False，
        检查响应列是否为常量值，启用下，若响应列为常量值，则引发异常，
        如果禁用，无论响应列是否为恒定值，模型都将训练
        
* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### XGBClassifier
xgboost分类算法.
* 参数:
    * max_depth : int，默认6，
        树的最大深度
    * learning_rate : float，默认0.3，
        学习率
    * n_estimators : int，默认50，
        树的棵数
    * verbosity : 默认0，
        控制训练和预测时的verbose信息
    * objective : 默认'binary:logistic'，
        损失函数
    * booster : 默认'gbtree'，
        基模型的类型，可选项：'gbtree'、'gblinear'、'dark'
    * tree_method : 默认'auto'，'auto'，'exact'，'approx'，'hist'，
        'gpu_exact'，'gpu_hist'
    * n_jobs : int，默认1，
        并行的job数量
    * gamma : 默认0，
        节点进一步分裂所需的最小loss减少
    * min_child_weight : 默认1，
        如果建树过程中，分裂导致叶节点的权重和小于min_child_weight，
        则建树过程将放弃该次分裂
    * max_delta_step : 默认0，
        允许每个叶子输出的最大增量步长，0则无约束
    * subsample : 默认1，
        控制每棵树随机采样的比例
    * colsample_bytree : 默认1，
        构造每棵树时列的下采样比例
    * colsample_bylevel : 默认1，
        构造树时每个级别的列下采样比例
    * colsample_bynode : 默认1，
        构造树时每个节点分裂时下采样的比例
    * reg_alpha : 默认0，
        L1正则化项
    * reg_lambda : 默认1，
        L2正则化项
    * scale_pos_weight : 默认1，
        控制正负权重的平衡
    * base_score : float，默认0.5，
        所有样本的初始预测分数
    * random_state : int,
        随机种子
    * missing : 缺失值设置，
        缺失值设置
    * score_each_iteration : 默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * stopping_rounds : 默认0,
        指定stopping_rounds合数后，在stopping_metric没有改善的情况下
        将停止训练，0禁用
    * stopping_metric : ，默认'AUTO'，
        指定用于提前停止训练的评判指标可选项: 
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'AUCPR'、'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'
    * stopping_tolerance : float，默认0.001，
        指定相对差，如果提升小于该值，则停止训练
    * max_runtime_secs : 默认0，
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * categorical_encoding : 指定以下编码方案之一来处理分类特征，
        (默认'AUTO'):'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * export_checkpoints_dir : 将生成的模型自动导出到指定的目录，
        指定目录
    * monotone_constraints : 表示单调约束的映射，
        +1表示递增约束，-1表示递减约束
    * score_tree_interval : 默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项
    * max_bins : int，默认256，
        构建直方图的bin的最大数量
    * max_leaves : int，默认0，
        指定每棵树的最大叶子节点数
    * sample_type : 默认'uniform'，
        采样算法的类型可选项:'uniform' : 统一选择丢弃树；
        'weighted' : 按权重比例选择丢弃树
    * normalize_type : , 默认'tree'，
        归一化算法的类型可选项：'tree' : 新树的每棵树都有相同的权重；
        'forest' : 新树的丢弃树（森林）总权重相同
    * rate_drop : 默认0，
        丢弃比例
    * one_drop : bool，默认False，
        启用此标志后，在丢弃期间始终会丢弃至少一颗树
    * skip_drop : 默认0，
        在boosting迭代期间跳过丢弃过程的概率
    * grow_policy : 默认'depthwise'，
        控制将新节点添加到树中的方式可选项: 
        'depthwise' : 距离根节点最近的节点处分割;
        'lossguide' : 在损失变化最大的节点处分离
    * dmatrix_type : ，默认'auto'
        指定DMatrix的类型,可选项:'auto'、'dense'、'sparse'
    
* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data
    * predict_proba，包含一个参数：待预测数据：test_data


### XGBRegressor
xgboost 回归算法
* 参数:
    * max_depth : int，默认6，
        树的最大深度
    * learning_rate : float，默认0.3，
        学习率
    * n_estimators : int，默认50，
        树的棵数
    * verbosity : 默认0，
        控制训练和预测时的verbose信息
    * objective : 默认'reg:linear'，
        损失函数
    * booster : 默认'gbtree'，
        基模型的类型，可选项：'gbtree'、'gblinear'、'dark'
    * tree_method : 默认'auto'，'auto'，'exact'，'approx'，'hist'，
        'gpu_exact'，'gpu_hist'
    * n_jobs : int，默认1，
        并行的job数量
    * gamma : 默认0，
        节点进一步分裂所需的最小loss减少
    * min_child_weight : 默认1，
        如果建树过程中，分裂导致叶节点的权重和小于min_child_weight，
        则建树过程将放弃该次分裂
    * max_delta_step : 默认0，
        允许每个叶子输出的最大增量步长，0则无约束
    * subsample : 默认1，
        控制每棵树随机采样的比例
    * colsample_bytree : 默认1，
        构造每棵树时列的下采样比例
    * colsample_bylevel : 默认1，
        构造树时每个级别的列下采样比例
    * colsample_bynode : 默认1，
        构造树时每个节点分裂时下采样的比例
    * reg_alpha : 默认0，
        L1正则化项
    * reg_lambda : 默认1，
        L2正则化项
    * scale_pos_weight : 默认1，
        控制正负权重的平衡
    * base_score : float，默认0.5，
        所有样本的初始预测分数
    * random_state : int,
        随机种子
    * missing : 缺失值设置，
        缺失值设置
    * score_each_iteration : 默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * stopping_rounds : 默认0,
        指定stopping_rounds合数后，在stopping_metric没有改善的情况下
        将停止训练，0禁用
    * stopping_metric : ，默认'AUTO'，
        指定用于提前停止训练的评判指标可选项: 
        'AUTO'、'logloss'、'MSE'、'RMSE'、'MAE'、'RMSLE'、'AUC'、
        'AUCPR'、'lift_top_group'、'misclassification'、
        'mean_per_class_error'、'custom'、'custom_increasing'
    * stopping_tolerance : float，默认0.001，
        指定相对差，如果提升小于该值，则停止训练
    * max_runtime_secs : 默认0，
        模型训练允许的最大运行时间（以秒为单位），0禁用
    * categorical_encoding : 指定以下编码方案之一来处理分类特征，
        (默认'AUTO'):'AUTO'、'UniformAdaptive'、'Random'、
        'QuantilesGlobal'、'RoundRobin'
    * export_checkpoints_dir : 将生成的模型自动导出到指定的目录，
        指定目录
    * monotone_constraints : 表示单调约束的映射，
        +1表示递增约束，-1表示递减约束
    * score_tree_interval : 默认0，
        每间隔score_tree_interval数量进行模型打分，设置为0则禁用该选项
    * max_bins : int，默认256，
        构建直方图的bin的最大数量
    * max_leaves : int，默认0，
        指定每棵树的最大叶子节点数
    * sample_type : 默认'uniform'，
        采样算法的类型可选项:'uniform' : 统一选择丢弃树；
        'weighted' : 按权重比例选择丢弃树
    * normalize_type : , 默认'tree'，
        归一化算法的类型可选项：'tree' : 新树的每棵树都有相同的权重；
        'forest' : 新树的丢弃树（森林）总权重相同
    * rate_drop : 默认0，
        丢弃比例
    * one_drop : bool，默认False，
        启用此标志后，在丢弃期间始终会丢弃至少一颗树
    * skip_drop : 默认0，
        在boosting迭代期间跳过丢弃过程的概率
    * grow_policy : 默认'depthwise'，
        控制将新节点添加到树中的方式可选项: 
        'depthwise' : 距离根节点最近的节点处分割;
        'lossguide' : 在损失变化最大的节点处分离
    * dmatrix_type : ，默认'auto'
        指定DMatrix的类型,可选项:'auto'、'dense'、'sparse'

* 方法:
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### ExtraTreesClassifier
极度随机树分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### ExtraTreesRegressor
极度随机树回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### BaggingClassifier
集成学习树分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### BaggingRegressor
集成学习树回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### AdaBoostClassifier
adboost树分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### AdaBoostRegressor
adaboost树回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### LogisticRegression
逻辑回归分类算法
* 参数: 
    * penalty : 默认'l2'，
        正则化项选择，可选'l1'，'l2'，'elasticnet'
    * dual : bool，默认False
        是否将原问题转换为其对偶问题
    * tol : float，默认1e-4，
        判断提前终止迭代时，残差减少的阈值
    * C : float，默认1.0，
        正则化参数，指定正则化强度
    * fit_intercept : bool，默认True
        是否将截距／方差加入到决策模型中
    * intercept_scaling : 默认1，
        截距缩放
    * class_weight : 默认None，即正负样本权重一样，
        用来调整正负样本比例的参数
    * random_state : 默认None，
        随机化种子
    * solver : 默认'lbfgs'，
        用来指明损失函数的优化方法，可选项:
        'newton-cg'、'lbfgs'、'liblinear'、'sag'、'saga'
    * max_iter : int，默认100，
        算法收敛的最大迭代次数
    * multi_class : 默认'auto'，
        多分类处理方法选择，可选：'auto'、'ovr'、'multinomial'
    * verbose : 默认0，不输出，
        是否输出任务进程等相关信息
    * warm_start : bool，默认False，
        是否使用上次模型结果作为初始化状态
    * n_jobs : int，默认None，
         并行任务数量
    * l1_ratio : 默认None，
        当正则化方式设置为'elasticnet'时，加入的'l1'比例
    * ignore_const_cols : bool，默认True，
        是否忽略常量训练列，因为无法从中获取任何信息
    * score_each_iteration : bool，默认False，
        若启用此选项，可在模型训练的每次迭代期间评分
    * lambda_search : bool，默认False，
        指定是否启用lambda搜索
    * early_stopping : 在训练集或验证集上效果没有更多提升时，是否提前停止
        提前终止训练
    * standardize : bool，默认True，
        是否进行标准化转换（零均值和单位方差）
    * missing_values_handling : 默认'MeanImputation'，
        指定如何处理缺失值，可选：'Skip'、'MeanImputation'、'PlugValues'
    * plug_values : 默认None，
        当missing_values_handling='PlugValues'时，
        指定一个单列frame用于计算缺失值，
    * compute_p_values : 是否计算p值，默认False，
    * remove_collinear_columns : 默认False，
        指定在模型构建期间是否自动删除共线列，
        仅当没有正则化（lambda=0）时设置，
    * intercept : bool，默认True，
        是否在模型中包括常数项
    * non_negative : bool，默认False，
        是否强制系数具有非负值
    * objective_epsilon : 默认-1，
        如果目标值小于此阈值，则模型收敛
    * prior : 默认-1，
        指定p的优先级概率
    * lambda_min_ratio : 指定lambda_search的最小lambda，
        lambda_search搜索空间限制
    * obj_reg : 默认-1，
        指定目标值计算中的似然除法器
    * export_checkpoints_dir : 将生成的模型自动导出到指定的目录，
        指定目录
    * balance_classes : bool，默认False，
        平衡类别分布
    * max_runtime_secs : int，默认0，
        模型训练允许的最大运行时间（以秒为单位），0表示禁用
* 方法: 
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### LinearRegression
线性回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### ElasticNet
弹性回归分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### Lasso
Lasso回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### Ridge
岭回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### GaussianNB
贝叶斯分类算法
* 参数:
    * priors=None,
    * var_smoothing=1e-9,
    * model_id=None,
    * nfolds=0,
    * seed=-1,
    * fold_assignment='AUTO',
    * fold_column=None,
    * keep_cross_validation_models=True,
    * keep_cross_validation_predictions=False,
    * keep_cross_validation_fold_assignment=False,
    * training_frame=None,
    * validation_frame=None,
    * response_column=None,
    * ignored_columns=None,
    * ignore_const_cols=True,
    * score_each_iteration=False,
    * balance_classes=False,
    * class_sampling_factors=None,
    * max_after_balance_size=5.0,
    * max_confusion_matrix_size=20,
    * max_hit_ratio_k=0,
    * laplace=0.0,
    * min_sdev=0.001,
    * eps_sdev=0.0,
    * min_prob=0.001,
    * eps_prob=0.0,
    * compute_metrics=True,
    * max_runtime_secs=0.0,
    * export_checkpoints_dir=None
* 方法: 
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### KNeighborsClassifier
最近邻分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### KNeighborsRegressor
最近邻回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### MLPClassifier
神经网络分类算法
* 参数: 同sklearn
* 方法: 同sklearn


### MLPRegressor
神经网络回归算法
* 参数: 同sklearn
* 方法: 同sklearn


### SVC
支持向量机分类算法
* 参数: 
    * C=1.0,
    * kernel='rbf',
    * degree=3,
    * gamma='scale',
    * coef0=0.0,
    * shrinking=True,
    * probability=False,
    * tol=1e-3,
    * cache_size=200,
    * class_weight=None,
    * verbose=False,
    * max_iter=200,
    * decision_function_shape='ovr',
    * random_state=None,
    * model_id=None,
    * training_frame=None,
    * validation_frame=None,
    * response_column=None,
    * ignored_columns=None,
    * ignore_const_cols=True,
    * rank_ratio=-1.0,
    * positive_weight=1.0,
    * negative_weight=1.0,
    * disable_training_metrics=True,
    * sv_threshold=0.0001,
    * fact_threshold=1e-05,
    * feasible_threshold=0.001,
    * surrogate_gap_threshold=0.001,
    * mu_factor=10.0
* 方法: 
    * fit，包含两个参数： 特征数据：X，标签数据：y
    * predict，包含一个参数： 待预测数据：test_data


### PCA
主成分分析 (PCA).
* 参数:
    * n_components=1,
    * copy=True,
    * whiten=False,
    * svd_solver='auto',
    * tol=0.0,
    * iterated_power='auto',
    * random_state=None,
    * model_id=None,
    * training_frame=None,
    * validation_frame=None,
    * ignored_columns=None,
    * ignore_const_cols=True,
    * score_each_iteration=False,
    * transform=None,
    * pca_method='GramSVD',
    * pca_impl=None,
    * max_iterations=1000,
    * use_all_factor_levels=False,
    * compute_metrics=True,
    * impute_missing=False,
    * max_runtime_secs=0.0,
    * export_checkpoints_dir=None,
    
* 方法:
    * fit，包含一个参数： 训练数据：X
    * predict，包含一个参数： 待预测数据：test_data



### IsolationForest
孤立森林算法
* 参数
    * n_estimators=50,
    * max_samples='auto',
    * contamination='auto',
    * max_features=1.0,
    * bootstrap=False,
    * n_jobs=None,
    * behaviour='deprecated',
    * random_state=None,
    * verbose=0,
    * warm_start=False,
    * model_id=None,
    * training_frame=None,
    * score_each_iteration=False,
    * score_tree_interval=0,
    * ignored_columns=None,
    * ignore_const_cols=True,
    * max_depth=8,
    * min_rows=1.0,
    * max_runtime_secs=0.0,
    * build_tree_one_node=False,
    * mtries=-1,
    * sample_size=256,
    * sample_rate=-1.0,
    * col_sample_rate_change_per_level=1.0,
    * col_sample_rate_per_tree=1.0,
    * categorical_encoding='AUTO',
    * stopping_rounds=0,
    * stopping_metric='AUTO',
    * stopping_tolerance=0.01,
    * export_checkpoints_dir=None,
* 方法:
    * fit，包含一个参数： 训练数据：X
    * predict，包含一个参数： 待预测数据：test_data
   

### KMeans
K-Means聚类算法
* 参数:
    * n_clusters : int, 默认1，
        要形成的簇数以及要生成的质心数
    * init : {'Random', 'Furthest', 'PlusPlus'}，默认'Furthest'，
        初始化方法
    * n_init : int，默认10，
        使用不同质心种子运行k-means算法的时间
    * max_iter : int，默认10，
        最大迭代次数
    * tol : float， 默认1e-4，
        收敛条件的阈值
    * precompute_distances : {‘auto’, True, False}，默认'auto'，
        是否预计算距离
    * verbose : int，默认0，
        控制训练和预测时的verbose信息
    * random_state : int，默认None，
        随机数种子
    * copy_x : bool，
        默认True
    * n_jobs : int，默认None，
        并行的job数量
    * algorithm : {“auto”, “full”, “elkan”}，
        默认'auto'
    * ignore_const_cols : bool，默认True，
        是否忽略常量训练列，因为无法从中获取任何信息
    * score_each_iteration : bool，默认False，
        启用此选项可在模型训练的每次迭代期间评分
    * estimate_k : bool，
        默认False
    * user_points : 指定一个dataframe，
        其中每一行表示一个初始群集中心
    * max_runtime_secs : float，默认0，
        模型训练允许的最大运行时间（以秒为单位）0表示禁用
    * categorical_encoding : 默认'AUTO'，
        指定以下编码方案之一，来处理分类特征：
        'AUTO'、'Enumenum_limited'、'OneHotExplicit'、
        'Binary'、'Eigen'、'LabelEncoder'、'SortByResponse'
    * export_checkpoints_dir : str，
        将生成的模型自动导出到指定的目录
    
* 方法:
    * fit，包含一个参数： 训练数据：X
    * predict，包含一个参数： 待预测数据：test_data