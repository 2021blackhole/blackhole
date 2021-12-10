[TOC]


# 总体介绍	
Jarvis 数据科学引擎是一个单机大数据、高性能数据分析工具。满足您在百 G 级别数据量下，使用个人开发机/工作站完成数据分析、机器学习的工作。主要特性如下：

 - 单机性能极致优化，通过并行计算，数据索引，核外计算等“黑科技”，实现百 G 数据量级下，相比Spark 5 -10 倍的性能提升。
 - pip 一键安装，单机极简部署，无服务化，无需运维。
 - 同一套数据，提供 SQL like、Pandas like、SKLearn like 三套接口，无学习成本。
 - 支持对接 HDFS，HIVE 等多种数据源，支持 CSV、Parquet、ORC 等多种数据格式，支持多数据源的联合分析。

# SQL 功能介绍&使用手册
数据导入、数据查询、表操作等 CK 常见和最可能被用户用到的 基础功能

1. 快速入门案例
---------

**1.1.功能简介**
	Jarvis sql是一款面向大数据的、旨在提供对结构化数据使用SQL语句进行查询、分析、统计等功能的单机计算引擎，提供了数据导入\导出能力。

   **1.2 应用场景**
   主要应用场景包括如下：

（1）数据存储和查询

（2）业务数据数据统计

（3）业务行为统计和分析

（4）日志分析

（5）商业智能/多维度分析\查询

  **1.3 常用SQL分析场景举例**
		假设存在两份数据events.csv，users.csv，分别记录了用户的访问、下单和购买信息以及用户的个人信息，下面从不同的分析场景举例来说明如何使用SQL引擎进行常用的用户行为分析。

events.csv的数据格式如下：

    1,83,pay_order,2021-04-09 01:13:20,sp_19819,85.01
    2,1,pay_order,2021-04-08 09:52:18,sp_55012,50.97
    3,72,add_cart,2021-04-06 05:39:10,sp_1044,0.0
    4,73,visit,2021-04-07 14:28:30,sp_14826,0.0
    5,53,visit,2021-04-10 08:28:59,sp_38361,0.0
    6,94,pay_order,2021-04-05 22:19:46,sp_79211,5.19

...



users.csv的数据内容如下：

    1,2,z6s5g0duce,28,1,长春
    2,2,t+nhd2scbv,18,1,南京
    3,4,4u8dmz+xgh,27,1,长沙
    4,1,r4y5ti16j+,60,1,广州
    5,2,ilgt53hb7c,62,2,兰州

...



代码如下：

    import blackhole as bh
    schema_events = '''
    	event_id Int64,
    	user_id Int32, 
    	event Enum('visit'=1,'add_cart'=2,'pay_order'=3), 
    	time DateTime,
    	item_id String,
    	fee Float32 
    '''
    schema_users = '''
        user_id Int32,
        equip Enum('android'=1,'ios'=2,'wm'=3,'pc'=4),
        user_name String,
        age Int8,
        gender Enum('男'=1,'女'=2),
        city String
    '''
    format = 'CSV'
    table_events = 'events'
    table_users = 'users'



1.3.1 先清理环境

    drop_events_table = 'drop table if exists %s' % table_events
    drop_users_table = 'drop table if exists %s' % table_users
    bh.sql(drop_events_table)
    bh.sql(drop_users_table)
    sql = '''SELECT * FROM file('events.csv', 'CSV', "%s")''' % schema_events
    bh.sql(sql).show()

1.3.2 建表并导入数据

    bh.sql("create table if not exists {} ({}) Engine=MergeTree() order by tuple()".format(table_events, schema_events)).show()
    
    bh.sql("insert into table {} from infile '{}' format CSV".format(table_events, "/****/***/data/events.csv")).show()
    
    bh.sql("create table if not exists {} ({}) Engine=MergeTree() order by tuple()".format(table_users, schema_users)).show()
    
    bh.sql("insert into table {} from infile '{}' format CSV".format(table_users, "/****/***/data/users.csv")).show()

1.3.3 日访问量(PV统计)

    sql_pv = '''SELECT count() as "今日PV" FROM events as t
    WHERE toDate(t.time)=today() AND t.event='visit' '''
    bh.sql(sql_pv).show()



1.3.4 日活用户量(UV统计)

    sql_uv = '''SELECT count(DISTINCT t.user_id) as "今日UV" FROM events as t
    WHERE toDate(t.time)=today() AND t.event='visit' '''
    bh.sql(sql_uv).show()


1.3.5 最近7天日活

    sql_7days_uv = '''SELECT toString(toDate(t.time)) as "日期", count(DISTINCT t.user_id) as "当日UV" FROM events as t
    WHERE t.event='visit' AND toDate(t.time) BETWEEN today()-7 AND today()
    GROUP BY toDate(t.time)'''
    bh.sql(sql_7days_uv).show()



1.3.6 今天分时活跃数

    sql_time_uv = '''SELECT toHour(t.time) as "时段", count(DISTINCT t.user_id) as "小时UV" FROM events as t
    WHERE t.event='visit' AND toDate(t.time)=today()
    GROUP BY toHour(t.time)'''
    bh.sql(sql_time_uv).show()



1.3.7 查询每天上午 10 点至 11 点的下单用户数

    sql_10_11_up = '''SELECT toString(toDate(t.time)) as "日期", count(DISTINCT t.user_id) as "下单用户数" FROM events as t
    WHERE t.event='add_cart' AND EXTRACT(HOUR FROM t.time) IN (10,11)
    GROUP BY toDate(t.time)'''
    bh.sql(sql_10_11_up).show()


1.3.8 查询来自某个城市的用户有多少

    sql_city_users = '''SELECT t.city as "城市", count(t.user_id) as "用户数" FROM users as t
    GROUP BY t.city'''
    bh.sql(sql_city_users).show()



1.3.9 漏斗分析 visit（访问）—add_cart（下单）—pay_order（支付）（窗口期 48 小时且严格满足事件先后顺序

    sql_vap_analyze = ''' SELECT count(DISTINCT t.user_id) as "全流程用户数" FROM
    (
    SELECT user_id, windowFunnel(172800)(time, event='visit',event='add_cart',event='pay_order') as level
    FROM events
    GROUP BY user_id
    ) as t WHERE t.level=3 '''
    bh.sql(sql_vap_analyze).show()




1.3.10 统计连续3(n)天访问的用户数

    sql_3days_continues_visit = ''' SELECT count(t2.user_id) as "连续3天访问用户数" FROM
    (
    SELECT user_id, windowFunnel(3)(t.dt, runningDifference(t.dt)=1, runningDifference(t.dt)=1) as level FROM
    (
    SELECT user_id, toDate(time) as dt FROM events ORDER BY user_id, time
    ) as t GROUP BY t.user_id
    ) as t2 WHERE t2.level=2 '''
    bh.sql(sql_3days_continues_visit).show()



1.3.11 统计过去3天内浏览最多的3件商品

    sql_top_sp_in_3_days = ''' SELECT topK(3)(t.item_id) as res FROM events as t WHERE t.event='visit' AND toDate(now()) - toDate(t.time) <=3 '''
    bh.sql(sql_top_sp_in_3_days).show()




1.3.12 统计过去3天内消费最多的3位用户

    sql_top_user_in_3_days = ''' SELECT t.user_id, t.fees as res FROM
    (
    SELECT user_id, sum(fee) as fees from events WHERE toDate(now()) - toDate(events.time) <=3 GROUP BY user_id
    ) as t ORDER BY res DESC limit 3 '''
    bh.sql(sql_top_user_in_3_days).show()



以上例子几乎涵盖了绝大部分用户行为分析的场景，当然用法不止上面这些，还有一些变通的用法，这里这是举例说明如何使用SQL引擎

##2. 基本功能介绍
### 2.1 数据类型定义
#### 2.1.1 数值类型
**2.1.1.1 Int类型**
	
	固定长度的整数类型又包括有符号和无符号的整数类型。
	
	有符号整数类型
	Int8	1	[-2^7 ~2^7-1]
	Int16	2	[-2^15 ~ 2^15-1]
	Int32	4	[-2^31 ~ 2^31-1]
	Int64	8	[-2^63 ~ 2^63-1]
	Int128	16	[-2^127 ~ 2^127-1]
	Int256	32	[-2^255 ~ 2^255-1]
	
	无符号类型
	UInt8	1	[0 ~2^8-1]
	UInt16	2	[0 ~ 2^16-1]
	UInt32	4	[0 ~ 2^32-1]
	UInt64	8	[0 ~ 2^64-1]
	UInt256	32	[0 ~ 2^256-1]

**2.1.1.2 浮点类型**
单精度浮点数
Float32从小数点后第8位起会发生数据溢出

	Float32

双精度浮点数
Float32从小数点后第17位起会发生数据溢出

	Float64

**2.1.1.3 Decimal类型**
有符号的定点数，可在加、减和乘法运算过程中保持精度。此处提供了Decimal32、Decimal64和Decimal128三种精度的定点数，支持几种写法：

	Decimal(P, S)
	Decimal32(S)
	数据范围：( -1 * 10^(9 - S), 1 * 10^(9 - S) )
	Decimal64(S)
	数据范围：( -1 * 10^(18 - S), 1 * 10^(18 - S) )
	Decimal128(S)
	数据范围： ( -1 * 10^(38 - S), 1 * 10^(38 - S) )
	Decimal256(S)
	数据范围：( -1 * 10^(76 - S), 1 * 10^(76 - S) )
其中：P代表精度，决定总位数（整数部分+小数部分），取值范围是1～76
S代表规模，决定小数位数，取值范围是0～P
根据P的范围，可以有如下的等同写法：

	[ 1 : 9 ]	Decimal(9,2)	Decimal32(2)
	[ 10 : 18 ]	Decimal(18,2)	Decimal64(2)
	[ 19 : 38 ]	Decimal(38,2)	Decimal128(2)
	[ 39 : 76 ]	Decimal(76,2)	Decimal256(2)
注意点：不同精度的数据进行四则运算时，精度(总位数)和规模(小数点位数)会发生变化，具体规则如下：
精度对应的规则

	Decimal64(S1) 运算符 Decimal32(S2) -> Decimal64(S)
	Decimal128(S1) 运算符 Decimal32(S2) -> Decimal128(S）
	Decimal128(S1) 运算符 Decimal64(S2) -> Decimal128(S)
	Decimal256(S1) 运算符 Decimal<32|64|128>(S2) -> Decimal256(S)
可以看出：两个不同精度的数据进行四则运算时，结果数据以最大精度为准
规模(小数点位数)对应的规则
加法|减法：S = max(S1, S2)，即以两个数据中小数点位数最多的为准
乘法： S = S1 + S2(注意：S1精度 >= S2精度)，即以两个数据的小数位相加为准
除法： S = S1，即被除数的小数位为准

#### 2.1.2 字符串类型
**2.1.2.1 String**
字符串可以是任意长度的。它可以包含任意的字节集，包含空字节。因此，字符串类型可以代替其他 DBMSs 中的VARCHAR、BLOB、CLOB 等类型。

**2.1.2.2 FixedString**
	
固定长度的N字节字符串，一般在在一些明确字符串长度的场景下使用，声明方式如下：

	<column_name> FixedString(N)
	-- N表示字符串的长度
注意：FixedString使用null字节填充末尾字符。

#### 2.1.3 日期类型
时间类型分为DateTime、DateTime64和Date三类。需要注意的是目前没有时间戳类型，也就是说，时间类型最高的精度是秒，所以如果需要处理毫秒、微秒精度的时间，则只能借助UInt类型实现。

	Date 类型
用两个字节存储，表示从 1970-01-01 (无符号) 到当前的日期值。日期中没有存储时区信息。

	DateTime 类型
用四个字节（无符号的）存储 Unix 时间戳，允许存储与日期类型相同的范围内的值，最小值为 0000-00-00 00:00:00。时间戳类型值精确到秒（没有闰秒），时区使用启动客户端或服务器时的系统时区。

#### 2.1.4 布尔类型
当前版本没有单独的类型来存储布尔值。可以使用UInt8 类型，取值限制为0或 1。

#### 2.1.5 数组类型
Array(T)，由 T 类型元素组成的数组。T 可以是任意类型，包含数组类型。但不推荐使用多维数组，目前对多维数组的支持有限。

    bh.sql("SELECT array(1, 2) AS x, toTypeName(x)").show()	
    或者
    bh.sql("SELECT [1, 2] AS x, toTypeName(x)").show()
    -- 结果输出
    x      toTypeName(array(1, 2))
    -----  -------------------------
    [1,2]  Array(UInt8)


需要注意的是，数组元素中如果存在Null值，则元素类型将变为Nullable。

    bh.sql("SELECT array(1, 2, NULL) AS x, toTypeName(x)").show()
    -- 结果输出
    x           toTypeName(array(1, 2, NULL))
    ----------  -------------------------------
    [1,2,NULL]  Array(Nullable(UInt8))

另外，数组类型里面的元素必须具有相同的数据类型，否则会报异常

    bh.sql("SELECT array(1, 'a')").show()
    -- 报异常
    DB::Exception: There is no supertype for types UInt8, String because some of them are String/FixedString and some of them are not

#### 2.1.6 枚举类型
枚举类型通常在定义常量时使用，当前版本提供了Enum8和Enum16两种枚举类型。

-- 建表

	bh.sql("CREATE TABLE t_enum (x Enum8('hello' = 1, 'world' = 2)) Engine=Memory").show()


-- INSERT数据

    bh.sql("INSERT INTO t_enum VALUES ('hello'), ('world'), ('hello')").show()

-- 如果定义了枚举类型值之后，不能写入其他值的数据

    bh.sql("INSERT INTO t_enum values('a')").show()
    -- 报异常：Unknown element 'a' for type Enum8('hello' = 1, 'world' = 2)

#### 2.1.7 Tuple类型
Tuple(T1, T2, ...)，元组，与Array不同的是，Tuple中每个元素都有单独的类型，不能在表中存储元组（除了内存表）。它们可以用于临时列分组。在查询中，IN表达式和带特定参数的 lambda 函数可以来对临时列进行分组。

    bh.sql("SELECT tuple(1,'a') AS x, toTypeName(x)").show()
   --结果输出
    
    x        toTypeName(tuple(1, 'a'))
    -------  -----------------------------
    (1,'a')  Tuple(UInt8, String)

   -- 建表
    
    bh.sql("CREATE TABLE t_tuple(c1 Tuple(String,Int8))").show()
   -- INSERT数据

    bh.sql("INSERT INTO t_tuple VALUES(('jack',20))").show()
   -- 查询数据
    
    bh.sql("SELECT * FROM t_tuple").show()
   -- 结果输出
    
    c1
    -----------
    ('jack',20)
-- 如果插入数据类型不匹配，会报异常
    
    bh.sql("INSERT INTO t_tuple VALUES(('tom','xxx'))").show()
    Code: 6. DB::Exception: Cannot parse string 'xxx' as Int8:

#### 2.1.8 特殊数据类型
Nullable
Nullable类型表示某个基础数据类型可以是Null值。其具体用法如下所示：

-- 建表

    bh.sql("CREATE TABLE t_null(x Int8, y Nullable(Int8)) engine=Memory").show()

-- 写入数据

    bh.sql("INSERT INTO t_null VALUES (1, NULL), (2, 3)").show()
    bh.sql("SELECT x + y FROM t_null").show()

-- 结果

    plus(x, y)
    -------------
    NULL
    5


### 2.2 数据导入
#### 2.2.1.从外部文件load数据到数据表
SQL引擎支持将外部数据导入到本地数据表中，格式包括CSV、CSVWithNames和Parquet，目前只有Python接口，用法如下：

    import blackhole as bh
    
    data_path = './test_data.csv'
    table_name = 'test_table'
    
    bh.sql("insert into table test_table from infile './test_data.csv'  format CSV").show()

上述代码会将本地的test_data.csv文件数据导入到test_table表中，其中schema为'id Int64, name String'。注意：指定的schema必须与数据源的schema相同。
#### 2.2.2.使用INSERT方式插入数据
1）单行插入

    import blackhole as bh
    schema = 'id Int64, name String, price Float32'
    table_name = 'test_insert_table'
    bh.sql("create table if not exists {} ({}) Engine=MergeTree() order by tuple()".format(table_name, schema)).show()
    insert_sql = '''INSERT INTO %s VALUES(%d,'%s',%f)''' % (table_name, 0, 'apple', 12.37)
    bh.sql(insert_sql).show()


2）多行插入

    insert_sql = 'INSERT INTO %s VALUES ' % table_name
    insert_sql += "(0,'apple',12.37),(1,'orange','11.23'),(2,'strawberry',11.34)"
    
    bh.sql(insert_sql).show()


注意：String类型字段值的引号不能省略。

### 2.3 数据查询
#### 2.3.3 从数据表查询

    import blackhole as bh
    query_sql = 'select * from test_table'
    bh.sql(query_sql).show()

#### 2.3.3 从外部文件查询

    bh.sql("select * from file('./test.csv','CSV','id Int64, name String, price Float32')").show()

#### 2.3.4 查询子句
**DISTINCT**
	查询结果集在指定字段上只保留一行

**GROUP BY**
   GROUP BY 子句将 SELECT 查询结果转换为聚合模式，目前支持的聚合函数如下：

   sum–对指定字段进行求和，只能对数值型字段求和
   min–求指定字段在查询结果集中的最小值
   max–求指定字段在查询结果集中的最大值


**HAVING**
  允许过滤由 GROUP BY 生成的聚合结果. 它类似于 WHERE ，但不同的是 WHERE 在聚合之前执行，而 HAVING 在之后进行。

**INTO OUTFILE**
  SELECT query 将其输出重定向到本机上的指定文件。

**JOIN**
  连接两个表
  支持的联接类型 
  所有标准 SQL JOIN 支持类型:

	INNER JOIN，只返回匹配的行。
	LEFT OUTER JOIN，除了匹配的行之外，还返回左表中的非匹配行。
	RIGHT OUTER JOIN，除了匹配的行之外，还返回右表中的非匹配行。
	FULL OUTER JOIN，除了匹配的行之外，还会返回两个表中的非匹配行。
	CROSS JOIN，产生整个表的笛卡尔积, “join keys” 是 不 指定。
	JOIN 没有指定类型暗指 INNER. 关键字 OUTER 可以安全地省略。

**LIMIT**
	LIMIT m 允许选择结果中起始的 m 行。

**ORDER BY**
	ORDER BY 子句包含一个表达式列表，每个表达式都可以用 DESC （降序）或 ASC （升序）修饰符确定排序方向。 如果未指定方向, 默认是 ASC。

**SAMPLE**
	SAMPLE K
	这里 k 从0到1的数字（支持小数和小数表示法）。 例如, SAMPLE 1/2 或 SAMPLE 0.5。这里k实际是百分比。

示例：

    SELECT
    Title,
    count() * 10 AS PageViews
    FROM hits_distributed
    SAMPLE 0.1
    WHERE
    CounterID = 34
    GROUP BY Title
    ORDER BY PageViews DESC LIMIT 1000

**SAMPLE N**
这里 n 是足够大的整数。 例如, SAMPLE 10000000.

在这种情况下，查询在至少 n 行（但不超过这个）上执行采样。 例如, SAMPLE 10000000 在至少10,000,000行上执行采样。

**UNION ALL**
联合两个子查询的结果

示例：

    SELECT CounterID, 1 AS table, toInt64(count()) AS c
    FROM test.hits
    GROUP BY CounterID
    UNION ALL
    SELECT CounterID, 2 AS table, sum(Sign) AS c
    FROM test.visits
    GROUP BY CounterID
    HAVING c > 0

**WHERE**
指定条件对查询结果集进行过滤，支持OR，AND等条件组合。

**FROM INFILE**
将外部文件数据导入到本地表中

### 2.4 数据导出

#### 2.4.1 直接展示
示例：

    import blackhole as bh
    query_sql = 'select * from test_table'
    bh.sql(query_sql).show()

默认打印前100行

#### 2.4.2 导出到文件
示例：

    query_sql = "select * from test_table into outfile './temp.csv'  Format CSV"
    bh.sql(query_sql).show()

保存为csv文件

----------
## 3. SQLTable 与 DataFrame 互转

### 3.1 SQLTable 转换为 Jarvis Dataframe

    import blackhole as bh 
    df = bh.sql('select * from test_table').to_df()

查询结果被转换成了dataframe，然后可以直接对dataframe执行相关操作，比如：
	
    print(df.shape)
    print(df)

### 3.2 SQLTable 转换为Pandas Dataframe

	import blackhole as bh
	pdf=bh.sql('select * from test_table'').to_pandas()

查询结果被转换成了Pandas dataframe，然后可以直接对pdf执行Pandas的相关操作，比如：

    pdf.info()

### 3.3 Jarvis Dataframe 转换为 SQLTable

    import blackhole as bh
    from blackhole.sql.dataset import Dataset
    import pyarrow as pa
    arrow_obj = pa.Table.from_pydict({
                "A": [3, 7, 11, 4],
                "B": [True, False, True, False],
                "C": [1.1, -2.5, 23, 0],
                "D": [2, 3, 4, -1],
                "E": ['make', 'love', 'not', 'war'],
            }, schema = pa.schema([
                pa.field("A", pa.int64()),
                pa.field("B", pa.bool_()),
                pa.field("C", pa.float64()),
                pa.field("D", pa.int8()),
                pa.field("E", pa.string())
            ]))
    import blackhole.dataframe as ks
    df = ks.from_arrow(arrow_obj)
    ds = Dataset()
    table_name = ds.from_df(df)
    bh.sql("select * from {}".format(table_name)).show()

将一个dataframe对象转换成了一个sqltable，并返回了表名，然后可以通过表名使用sql对表进行查询。


​	
## 4.  Cache
### 4.1 Cache的作用
cache模块旨在通过缓存方式提升离线分析的性能，解决以下问题：
* 重复查询：在很多离线分析场景下，用户会消耗额外的时间和资源用于同一sql语句的重复查询。通过cache可快速复用之前查询的结果。
* 断点续跑：复杂查询，可能在后期故障退出，之前完成的大量计算需要重复执行。通过cache可以做阶段性的checkpoint，方便断点续跑。
* 远程数据缓存至本地：在很多离线分析场景下，数据可能是在远程存储上，频繁从远程拉取数据成本高。通过cache可将数据缓存至本地，提升离线分析性能。
* 同时，cache模块也支持对python函数的缓存

### 4.2 sql节点的cache操作

    import blackhole as bh
    
    t1 = bh.sql('SELECT * FROM table1')
    t2 = bh.sql('SELECT * FROM table2')
    
    # 缓存t1、t2节点
    t1.cache()
    t2.cache()
    
    # 后续sql的操作，在使用t1、t2节点时，会复用对应节点的缓存
    t3 = bh.sql('SELECT * from {t1} join on {t2}')
    t3.show()
    
    # cache的管理：支持LRU策略下的自动cache清除，同时也支持手动清除所有cache
    bh.uncache() # 手动清除所有cache的操作

### 4.3 函数节点的cache操作

    import blackhole as bh
    import time
    
    # 使用bh.cache()装饰器，可以使任意python函数具备cache能力
    @bh.cache()
    def expensive_cal(x, y):
        time.sleep(5)
        return x + y
    
    #首次运行expensive_cal函数，耗时5s+，同时结果被缓存
    expensive_cal(1, 1)
    
    # 再次运行expensive_val函数，直接返回缓存结果，耗时在毫秒级
    expensive_cal(1, 1)
    
    # 改变入参，重新运行expensive_cal，cache miss，耗时5s+，同时结果再次被缓存
    expensive_cal(1, 2)
    
    # 说明：当函数的入参、函数体、函数依赖的外部变量以及函数依赖的外部函数，任意一项发生变化时，则函数cahce会miss，并重新缓存，以保证函数结果的准确。
## 5.  Catalog
### 5.1 Catalog的定义
待分析的数据，可能分布在HIVE、HDFS、MySQL、BOS等不同的系统中。Catalog功能，可帮用户管理外部可用的数据库和表的目录结构，给用户一个全局的数据视图，在引入外部数据和联邦查询中有重要作用。
Catalog功能有三个功能：可以解耦数据提供方和使用方；可以简化数据使用流程；可以避免直接配置密码、避免安全问题。
Catalog表示一个数据来源，可以包含来自不同系统的数据库和表，比如customer_catalog表示分析用户购买行为相关的数据源，其中包含来自MySQL的order_db和HIVE的customer_history_info_db等，通过联合分析MySQL order_db.order_list和HIVE customer_history_info_db.show_list来分析用户的喜好。所有分析师，只需要**配置Catalog源**的信息（比如http server的访问信息），就能看到这些数据库和表，就可以：**展示Catalog包含的Databases和Tables**，**从Catalog源导入某Database或Table** 到Jarvis里做分析。
        
### 5.2 配置Catalog源
	        Catalog源的配置文件，是Jarvis的运行目录下的.jarvis/catalog_conf.json文件。首次启动，会产生空文件。用户可以配置不同类型（http/bos/local/hdfs等）的Catalog源(注意：目前只重点支持了http类型）。
	      .jarvis/catalog_conf.json内容：
	      
		[
		    {
		        "type": "http",
		        "name": "http_catalog",
		        "uri": "http://ip:8080/http_catalog/",
		        "user": "user",
		        "password": "cGFzc3dvcmQ="
		    }
		]

说明：Catalog源提前放置到Nginx服务的指定目录下（如http_catalog)，具体内容按catalog name/database name/table & conn_info.json三层来组织，具体目录内容由数据提供方来维护，使用方不需要了解这些细节： table里面写表的table的schema,  \t 作为name 和类型的分隔符，不同列间用\n分隔，比如

    #用户不需要关注如下细节：
    #catalog/database/table1:  
    col1 \t  Int32,
    col2 \t String
    
    # catalog/database/conn_info.json写这些table的连接信息，格式如:
    {
        "type": "BOS", #或HIVE/Iceberg/MySQL/..
        "host_port": "http://ip:port/",
        "format": {
              "taxi": "CSV",
    	      "iris": "Parquet"
        }
    }

### 5.3 展示Catalog包含的Databases和Tables
catalog在database上一级，SHOW命令：
		
		bh.sql("SHOW DATABASES FROM http_catalog") 
		bh.sql("SHOW TABLES FROM http_catalog.db1") 
		bh.sql("DESC http_catalog.db1.table1") 

### 5.4 从Catalog源导入Database或Table

		bh.sql("CREATE DATABASE mydb1 FROM http_catalog.db1")
		bh.sql("CREATE TABLE mytable1 FROM http_catalog.db1.table1')")

导入后，可以按本地库或表一样访问外部数据： （但数据仍然在远端数据源，可结合cache把远端数据源缓存在本地、来加快查询速度）
		
		bh.sql("SHOW TABLES FROM mydb1")
		bh.sql("SELECT * FROM mytable1")
		bh.sql("DROP mydb1")


## 6. 外部数据对接
### 6.1 DAE数据对接
DAE数据，可以认为是一种特殊形式的catalog，catalog内置在Jarvis中。使用方式如下：


    bh.sql("show catalogs").show()
    _ _ _ _ _
    dae
    
    bh.sql("show databases from dae").show()
    _ _ _ _ _
    db1
    db2
    
    bh.sql("create database my_db1 from dae.db1").show()
    _ _ _ _ _
    create my_db1 success
    
    #接下来，可以访问本地的my_db1中的table数据: *
    bh.sql("select * from my_db1.table1").show()
    
    #除了直接sql分析，也可以将数据“取数”到Pandas Dataframe或Blackhole Dataframe：
    df = bh.sql("select * from my_db1.table1").to_pandas()
    df = bh.sql("select * from my_db1.table1").to_df()
    
    #然后可以按python dataframe方式做计算
    df = df[df.col1 > 4]     
    
    *说明：除了create database 也可以单独创建一张表：bh.sql("create table my_tb1 from dae.db1.table1").show()


### 6.2 Iceberg对接
#### 6.2.1 iceberg-catalog支持
	可以通过catalog支持iceberg库、表的建立
#### 6.2.2 使用方式
   spark iceberg原生表：

    CREATE TABLE local.db.sc1 (
        id bigint,
        data string,
        category string)
    USING iceberg;

jarvis 对应的表建立语句：

	CREATE TABLE icsc1 (
		id Int64,
		data String, 
		category String) 
	ENGINE=Iceberg('hdfs://localhost:8020/user/hive/warehouse/', 'Parquet','db','sc1','HADOOP')

其中Iceberg中有五个参数
	param1：如果是FE模式，参数1代表 fe_ip:fe_rpc_port；如果是HADOOP模式，参数1代表warehouse url
	param2：存储文件格式
	param3：iceberg原生db名
	param4：iceberg原生table名
	param5：type，分为FE/HADOOP，其中FE是支持DAE、HADOOP为spark-hadoop模式表

#### 6.2.3 嵌套格式支持
目前支持iceberg的nested，map，array以及他们的组合格式，例子如下所示：
iceberg表1：

		CREATE TABLE local.db.s2 (
		    id bigint,
		    data string,
		    category string,
		    point struct<x:bigint,y:bigint>)
		USING iceberg;

jarvis表1：

		CREATE TABLE ics2 (
			id Int64, 
			data String, 
			category String, 
			point Tuple(x Int64, y Int64)) 
		ENGINE=Iceberg('hdfs://localhost:8020/user/hive/warehouse/', 'Parquet','db','s2','HADOOP')

支持复杂格式内部单一格式的非空，复杂格式本身非空不支持
	
iceberg表2：（map->Map)

	CREATE TABLE local.db.s3 (
	    id bigint,
	    data string,
	    category string,
	    point map<bigint,bigint>)
	USING iceberg;

jarvis表2：
	
	CREATE TABLE ics3 (
		id Int64, 
		data String,
		category String, 
		point Map(Int64, Int64)) 
	ENGINE=Iceberg('hdfs://localhost:8020/user/hive/warehouse/', 'Parquet','db','s3','HADOOP')

iceberg表3：（array->Array)

	CREATE TABLE local.db.s4 (
	    id bigint,
	    data string,
	    category string,
	    point array<bigint>)
	USING iceberg;
	
	CREATE TABLE ics4 (
		id Int64, 
		data String, 
		category String, 
		point Array(Int64)) 
	ENGINE=Iceberg('hdfs://localhost:8020/user/hive/warehouse/', 'Parquet','db','s4','HADOOP')

#### 6.2.4 查询优化相关
1.3.2版本会支持部分辅助数据下推


​	
